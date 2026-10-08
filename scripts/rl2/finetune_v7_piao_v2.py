#!/usr/bin/env python
"""bc_v7_base piao 微调 v2（修复版）。

相对 v1 的修复：
1. 数据集：分层抽样（scripts/build_finetune_indices_v2.py），piao ≤35%，
   chi/peng/gang/pass/听牌点都有代表，不再被 discard 淹没。
2. lr=1e-5（v1 的 1/10），1 epoch。
3. 掩码损失：动作 BCE 只在 candidates_mask==1 的头位置计算（与 bc_v7 原始
   训练口径一致），tile CE 用 candidates_tile_mask 屏蔽不可行牌。
4. KL 锚定：基座模型的 6 类动作分布由 scripts/precompute_base_probs_v2.py
   在独立推理进程中预计算（避免 WSL2 训练+推理混合导致的 CUDA 死锁），
   损失 = CE + kl_coef × KL(基座分布 ‖ 当前分布)，只在候选动作上归一化后计算。
   **piao 样本豁免 KL**（--piao-kl-weight 0）：KL 的目的是防遗忘，
   piao 点恰恰是要改变行为的地方（基座在那里 86% 预测 hu，KL 会对抗学习），
   因此 KL 只锚非 piao 样本。

用法:
    setsid .venv/bin/python scripts/finetune_v7_piao_v2.py > outputs/finetune_v2.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nnrl2.model_v7 import MahjongTransformerV7

OBS_KEY_MAP = {
    "obs_hand": "hand",
    "obs_discards": "discards",
    "obs_melds": "melds",
    "obs_action_history": "action_history",
    "obs_scores": "scores",
    "obs_god": "god",
    "obs_wall_remaining": "wall_remaining",
    "obs_turn": "turn",
    "obs_phase": "phase",
    "obs_target": "target",
    "obs_seat": "seat",
    "obs_dealer": "dealer",
}

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]
# 头顺序 (hu, gang, peng, chi, discard) → 动作 id
HEAD_TO_ACTION = [4, 3, 2, 1, 0]


def binary_logits_to_probs6(binary_logits: torch.Tensor) -> torch.Tensor:
    """5 个 sigmoid 头 → 6 类概率（优先级链式，与 predict_full 一致）。"""
    sig = torch.sigmoid(binary_logits)
    p_hu, p_gang, p_peng, p_chi, p_dis = sig.unbind(dim=1)
    rem = 1.0 - p_hu
    out = torch.zeros(sig.shape[0], 6, device=sig.device, dtype=sig.dtype)
    out[:, 4] = p_hu
    out[:, 3] = rem * p_gang; rem = rem * (1 - p_gang)
    out[:, 2] = rem * p_peng; rem = rem * (1 - p_peng)
    out[:, 1] = rem * p_chi;  rem = rem * (1 - p_chi)
    out[:, 0] = rem * p_dis;  rem = rem * (1 - p_dis)
    out[:, 5] = rem
    return out


class FinetuneDatasetV2:
    """从 audit npz 按索引采样，全量预加载到 GPU（含 candidates 掩码与基座分布）。

    不走 DataLoader：直接在预加载的 GPU 张量上批量索引，避免逐样本
    GPU 索引的开销（v1 的训练循环就是这么快的）。
    base_probs: (N,6) 基座模型分布，由 precompute_base_probs_v2.py 预计算。
    """

    def __init__(self, npz_path: str, indices: np.ndarray, device: str = "cuda",
                 base_probs_path: str = "data/finetune_v2_base_probs.npy",
                 piao_indices_path: str = "data/piao_indices.npy"):
        print(f"加载数据: {npz_path}", flush=True)
        d = np.load(npz_path, allow_pickle=True)
        n = len(indices)
        print(f"采样 {n} 条", flush=True)

        self.obs = {}
        for npz_key, model_key in OBS_KEY_MAP.items():
            data = d[npz_key][indices]
            dtype = np.float32 if data.dtype in (np.float32, np.float64) else np.int64
            self.obs[model_key] = torch.from_numpy(data.astype(dtype)).to(device)

        self.action_labels = torch.from_numpy(
            d["chosen_action_kind"][indices].astype(np.int64)).to(device)
        self.tile_labels = torch.from_numpy(
            d["chosen_tile"][indices].astype(np.int64)).to(device)
        # (N,6) 可行性掩码；(N,34) tile 可行性
        self.candidates_mask = torch.from_numpy(
            d["candidates_mask"][indices].astype(np.float32)).to(device)
        self.candidates_tile_mask = torch.from_numpy(
            d["candidates_tile_mask"][indices].astype(np.float32)).to(device)

        self.n = n
        self.device = device
        # 基座分布（与 indices 对齐，由独立进程预计算）
        self.base_probs = torch.from_numpy(
            np.load(base_probs_path).astype(np.float32)).to(device)
        assert self.base_probs.shape == (n, 6), \
            f"base_probs shape {self.base_probs.shape} != ({n},6)"
        # piao 样本标记（用于 KL 豁免）
        piao_set = np.load(piao_indices_path)
        self.is_piao = torch.from_numpy(
            np.isin(indices, piao_set).astype(np.float32)).to(device)
        print(f"数据已加载到 {device}（含基座分布，piao 样本 {int(self.is_piao.sum())} 条）", flush=True)

    def __len__(self):
        return self.n

    def batch(self, idx: torch.Tensor):
        """直接批量取一个 batch（idx 为 GPU 上的 long 张量）。"""
        obs = {k: v[idx] for k, v in self.obs.items()}
        return (obs, self.action_labels[idx], self.tile_labels[idx],
                self.candidates_mask[idx], self.candidates_tile_mask[idx],
                self.base_probs[idx], self.is_piao[idx])


def load_model(model_path: str, device: str) -> MahjongTransformerV7:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    margs = ckpt.get("args", {})
    model = MahjongTransformerV7(
        d_model=margs.get("d_model", 128),
        nhead=margs.get("nhead", 8),
        num_layers=margs.get("num_layers", 4),
        dim_feedforward=margs.get("dim_feedforward", 512),
        dropout=margs.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型加载完成: {n_params / 1e6:.2f}M 参数 ({model_path})", flush=True)
    return model


def head_targets(action_labels: torch.Tensor, candidates_mask: torch.Tensor):
    """构造 (batch,5) target 与 head 级掩码。

    candidates_mask 是 (batch,6) 的动作可行性；头 h 对应动作 HEAD_TO_ACTION[h]。
    只有候选动作才计损失：候选且教师选了→1，候选但没选→0，非候选→掩掉。
    """
    bs = action_labels.shape[0]
    dev = action_labels.device
    target = torch.zeros(bs, 5, device=dev)
    mask = torch.zeros(bs, 5, dtype=torch.bool, device=dev)
    for h, act in enumerate(HEAD_TO_ACTION):
        mask[:, h] = candidates_mask[:, act] > 0.5
        target[:, h] = (action_labels == act).float()
    return target, mask


def compute_losses(model, obs, action_labels, tile_labels,
                   cand_mask, cand_tile_mask, base_p6, is_piao,
                   kl_coef: float, tile_loss_weight: float,
                   piao_kl_weight: float = 0.0,
                   piao_bce_weight: float = 1.0):
    """返回 (total, action_bce, tile_ce, kl)。base_p6 为预计算的基座 6 类分布。"""
    binary_logits, tile_logits, _ = model(obs)

    # --- 动作 BCE（掩码，只算候选头；piao 样本按 piao_bce_weight 加权）---
    target, hmask = head_targets(action_labels, cand_mask)
    bce_elem = F.binary_cross_entropy_with_logits(binary_logits, target, reduction="none")
    sw = (1.0 - is_piao) + is_piao * piao_bce_weight  # (batch,) 样本权重
    wmask = hmask.float() * sw.unsqueeze(1)
    action_loss = (bce_elem * wmask).sum() / wmask.sum().clamp(min=1.0)

    # --- tile CE（仅 discard 样本，且屏蔽不可行牌）---
    d_mask = action_labels == 0
    if d_mask.any():
        tl = tile_logits[d_mask]
        tm = cand_tile_mask[d_mask]
        tl = tl.masked_fill(tm < 0.5, float("-inf"))
        tile_loss = F.cross_entropy(tl, tile_labels[d_mask])
    else:
        tile_loss = torch.zeros((), device=binary_logits.device)

    # --- KL 锚定（基座分布 ‖ 当前分布，候选动作上归一化；piao 样本豁免）---
    if kl_coef > 0:
        cur_p6 = binary_logits_to_probs6(binary_logits)
        cm = cand_mask  # (batch,6) float
        bp = base_p6 * cm
        bp = bp / bp.sum(dim=1, keepdim=True).clamp(min=1e-8)
        cp = cur_p6 * cm
        cp = cp / cp.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # 逐样本 KL(base || cur)
        kl_per = (bp * (bp.clamp(min=1e-8).log() - cp.clamp(min=1e-8).log())).sum(dim=1)
        # piao 样本权重 piao_kl_weight（默认 0 = 豁免），非 piao 权重 1
        w = 1.0 - is_piao * (1.0 - piao_kl_weight)
        kl = (kl_per * w).sum() / w.sum().clamp(min=1.0)
    else:
        kl = torch.zeros((), device=binary_logits.device)

    total = action_loss + tile_loss_weight * tile_loss + kl_coef * kl
    return total, action_loss.detach(), tile_loss.detach(), kl.detach()


def save_checkpoint(model, args, epoch, out_path):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "args": {
            "d_model": model.d_model,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
        "finetuned_from": args.base_model,
        "finetune_args": vars(args),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)


def main():
    ap = argparse.ArgumentParser(description="bc_v7_base piao 微调 v2（掩码损失 + KL 锚定）")
    ap.add_argument("--base-model", default="runs/bc_v7_base.pt")
    ap.add_argument("--data", default="data/audit_v56_with_model.npz")
    ap.add_argument("--finetune-indices", default="data/finetune_indices_v2.npy")
    ap.add_argument("--out", default="runs/bc_v7_piao_finetuned_v2.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--kl-coef", type=float, default=0.5)
    ap.add_argument("--piao-kl-weight", type=float, default=0.0,
                    help="piao 样本的 KL 权重（0=完全豁免，1=与非 piao 同等锚定）")
    ap.add_argument("--piao-bce-weight", type=float, default=1.0,
                    help="piao 样本在动作 BCE 中的权重（>1 强化 piao 学习信号）")
    ap.add_argument("--tile-loss-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Args: {vars(args)}", flush=True)

    # 训练模型（基座分布已预计算，训练进程内不做任何 no_grad 推理）
    model = load_model(args.base_model, device)

    indices = np.load(args.finetune_indices)
    print(f"微调样本: {len(indices)}", flush=True)

    ds = FinetuneDatasetV2(args.data, indices, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    print(f"\n=== 开始微调 v2 ({args.epochs} epoch, lr={args.lr}, kl_coef={args.kl_coef}) ===", flush=True)
    model.train()
    t0 = time.perf_counter()
    tot = tot_a = tot_t = tot_k = 0.0
    nb = 0
    n = len(ds)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        for s in range(0, n - args.batch_size + 1, args.batch_size):
            idx = perm[s:s + args.batch_size]
            obs, act, tile, cm, ctm, bp6, isp = ds.batch(idx)
            loss, a, t, k = compute_losses(
                model, obs, act, tile, cm, ctm, bp6, isp,
                kl_coef=args.kl_coef, tile_loss_weight=args.tile_loss_weight,
                piao_kl_weight=args.piao_kl_weight,
                piao_bce_weight=args.piao_bce_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot += loss.item(); tot_a += a.item(); tot_t += t.item(); tot_k += k.item()
            nb += 1
            if nb % 10 == 0:
                print(f"  step {nb} loss={tot/nb:.4f} (bce={tot_a/nb:.4f} "
                      f"tile={tot_t/nb:.4f} kl={tot_k/nb:.4f}) "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n训练完成: {nb} steps, {dt:.1f}s, avg_loss={tot/nb:.4f}", flush=True)

    save_checkpoint(model, args, args.epochs, args.out)
    print(f"模型已保存: {args.out}", flush=True)

    # 训练日志摘要（供报告引用）
    summary = {
        "args": vars(args),
        "steps": nb,
        "train_seconds": dt,
        "avg_loss": tot / max(nb, 1),
        "avg_bce": tot_a / max(nb, 1),
        "avg_tile": tot_t / max(nb, 1),
        "avg_kl": tot_k / max(nb, 1),
    }
    log_path = Path(args.out).with_suffix(".train_log.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"训练日志: {log_path}", flush=True)


if __name__ == "__main__":
    main()
