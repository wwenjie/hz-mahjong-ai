#!/usr/bin/env python
"""bc_v7_base piao 过采样微调脚本（方案 B）。

用法:
    .venv/bin/python scripts/finetune_v7_piao.py [--epochs 5] [--batch-size 512] [--lr 1e-4]
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nnrl2.model_v7 import MahjongTransformerV7

# obs key 映射：npz 的 key → 模型 encoder 的 key
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


class FinetuneDataset(Dataset):
    """从 audit npz 按索引采样的微调数据集，全量预加载到 GPU。"""

    def __init__(self, npz_path: str, indices: np.ndarray, device: str = "cuda"):
        print(f"加载数据: {npz_path}")
        d = np.load(npz_path, allow_pickle=True)

        n = len(indices)
        print(f"采样 {n} 条")

        # 加载 obs 字段并映射 key
        self.obs = {}
        for npz_key, model_key in OBS_KEY_MAP.items():
            data = d[npz_key][indices]
            dtype = torch.float32 if data.dtype in (np.float32, np.float64) else torch.int64
            self.obs[model_key] = torch.from_numpy(data.astype(
                np.float32 if dtype == torch.float32 else np.int64
            )).to(device)

        # 标签
        self.action_labels = torch.from_numpy(
            d["chosen_action_kind"][indices].astype(np.int64)
        ).to(device)
        self.tile_labels = torch.from_numpy(
            d["chosen_tile"][indices].astype(np.int64)
        ).to(device)

        self.n = n
        self.device = device
        print(f"数据已加载到 {device}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        obs = {k: v[idx] for k, v in self.obs.items()}
        return obs, self.action_labels[idx], self.tile_labels[idx]


def collate_gpu(batch):
    """数据已在 GPU 上，直接 stack。"""
    obs_keys = batch[0][0].keys()
    obs = {}
    for k in obs_keys:
        obs[k] = torch.stack([b[0][k] for b in batch])
    actions = torch.stack([b[1] for b in batch])
    tiles = torch.stack([b[2] for b in batch])
    return obs, actions, tiles


def load_model(model_path: str, device: str) -> MahjongTransformerV7:
    """加载 bc_v7_base 模型。"""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = MahjongTransformerV7(
        d_model=args.get("d_model", 128),
        nhead=args.get("nhead", 8),
        num_layers=args.get("num_layers", 4),
        dim_feedforward=args.get("dim_feedforward", 512),
        dropout=args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型加载完成: {n_params / 1e6:.2f}M 参数")
    return model


def compute_loss(model, obs, action_labels, tile_labels):
    """计算微调损失。

    动作头：V7 的 5 个 sigmoid 二分类头。对微调，我们将 chosen_action_kind
    转为对应的二分类目标，计算 BCEWithLogits。
    Tile 头：34 类交叉熵（仅 discard 样本）。
    """
    binary_logits, tile_logits, _ = model(obs)

    # binary_logits: (batch, 5) — 顺序 (hu, gang, peng, chi, discard)
    # HEAD_ACTIONS = (4, 3, 2, 1, 0)
    HEAD_ACTIONS = [4, 3, 2, 1, 0]

    # 构造 5 头二分类标签：正类 = chosen_action_kind
    binary_targets = torch.zeros_like(binary_logits)
    for head_idx, action_id in enumerate(HEAD_ACTIONS):
        binary_targets[:, head_idx] = (action_labels == action_id).float()

    # pass (5) 样本：所有头都是负类（已经默认为 0）
    # 对所有头计算 BCE
    action_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)

    # Tile loss：仅 discard 样本
    discard_mask = action_labels == 0
    if discard_mask.any():
        tile_loss = F.cross_entropy(tile_logits[discard_mask], tile_labels[discard_mask])
    else:
        tile_loss = torch.tensor(0.0, device=binary_logits.device)

    total_loss = action_loss + tile_loss
    return total_loss, action_loss.item(), tile_loss.item()


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, desc=""):
    """评估：返回 top-1 动作准确率 + discard tile 准确率。"""
    model.eval()
    n = len(dataset)
    correct_action = 0
    correct_tile = 0
    n_discard = 0
    total_loss = 0.0
    n_batches = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_gpu)

    for obs, action_labels, tile_labels in loader:
        binary_logits, tile_logits, _ = model(obs)

        # 合成 6 类概率
        sig = torch.sigmoid(binary_logits)
        p_hu, p_gang, p_peng, p_chi, p_dis = sig.unbind(dim=1)
        rem = 1.0 - p_hu
        probs = torch.zeros(sig.shape[0], 6, device=device)
        probs[:, 4] = p_hu
        probs[:, 3] = rem * p_gang; rem *= (1 - p_gang)
        probs[:, 2] = rem * p_peng; rem *= (1 - p_peng)
        probs[:, 1] = rem * p_chi; rem *= (1 - p_chi)
        probs[:, 0] = rem * p_dis; rem *= (1 - p_dis)
        probs[:, 5] = rem

        pred_action = probs.argmax(dim=1)
        correct_action += (pred_action == action_labels).sum().item()

        # Tile accuracy (discard only)
        discard_mask = action_labels == 0
        if discard_mask.any():
            pred_tile = tile_logits[discard_mask].argmax(dim=1)
            correct_tile += (pred_tile == tile_labels[discard_mask]).sum().item()
            n_discard += discard_mask.sum().item()

        # Loss
        loss, _, _ = compute_loss(model, obs, action_labels, tile_labels)
        total_loss += loss.item()
        n_batches += 1

    action_acc = correct_action / n * 100
    tile_acc = correct_tile / n_discard * 100 if n_discard > 0 else 0
    avg_loss = total_loss / max(n_batches, 1)

    return {
        "action_acc": action_acc,
        "tile_acc": tile_acc,
        "loss": avg_loss,
        "n": n,
        "n_discard": n_discard,
    }


@torch.no_grad()
def evaluate_full_audit(model, npz_path, piao_indices, device, batch_size=1024):
    """在全量审计集上评估，重点看听牌点和 piao 点的一致率。"""
    print(f"\n=== 全量审计集评估 ===")
    d = np.load(npz_path, allow_pickle=True)
    n_total = len(d["chosen_action_kind"])
    print(f"总样本: {n_total}")

    # 预加载全部数据到 GPU（分批推理以避免 OOM）
    all_obs = {}
    for npz_key, model_key in OBS_KEY_MAP.items():
        data = d[npz_key]
        dtype = np.float32 if data.dtype in (np.float32, np.float64) else np.int64
        all_obs[model_key] = torch.from_numpy(data.astype(dtype)).to(device)

    all_action_labels = torch.from_numpy(d["chosen_action_kind"].astype(np.int64)).to(device)
    all_shanten = d["shanten"]
    model_top1_orig = d["model_top1_action"]  # 基座模型的 top1（存于 npz）

    # 分批推理
    model.eval()
    all_pred_actions = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        obs_batch = {k: v[start:end] for k, v in all_obs.items()}
        binary_logits, tile_logits, _ = model(obs_batch)

        sig = torch.sigmoid(binary_logits)
        p_hu, p_gang, p_peng, p_chi, p_dis = sig.unbind(dim=1)
        rem = 1.0 - p_hu
        probs = torch.zeros(sig.shape[0], 6, device=device)
        probs[:, 4] = p_hu
        probs[:, 3] = rem * p_gang; rem *= (1 - p_gang)
        probs[:, 2] = rem * p_peng; rem *= (1 - p_peng)
        probs[:, 1] = rem * p_chi; rem *= (1 - p_chi)
        probs[:, 0] = rem * p_dis; rem *= (1 - p_dis)
        probs[:, 5] = rem

        pred = probs.argmax(dim=1)
        all_pred_actions.append(pred.cpu())

        if (start // batch_size) % 100 == 0:
            print(f"  推理进度: {end}/{n_total}")

    all_pred = torch.cat(all_pred_actions).numpy()

    # 总体一致率（微调后模型 vs 引擎选择）
    overall_agree = (all_pred == all_action_labels.cpu().numpy()).mean() * 100

    # 听牌点（shanten=-1）一致率
    tingpai_mask = all_shanten == -1
    if tingpai_mask.sum() > 0:
        tingpai_agree = (all_pred[tingpai_mask] == all_action_labels.cpu().numpy()[tingpai_mask]).mean() * 100
    else:
        tingpai_agree = 0

    # piao 点一致率
    piao_mask = np.zeros(n_total, dtype=bool)
    piao_mask[piao_indices] = True
    piao_agree = (all_pred[piao_mask] == all_action_labels.cpu().numpy()[piao_mask]).mean() * 100

    # piao 点：模型 top1=hu 的分歧数
    piao_pred_hu = (all_pred[piao_mask] == 4).sum()  # 4 = hu
    piao_total = piao_mask.sum()
    piao_chosen = all_action_labels.cpu().numpy()[piao_mask]
    piao_engine_discard = (piao_chosen == 0).sum()  # 引擎选择 discard

    print(f"\n总体一致率: {overall_agree:.2f}%")
    print(f"听牌点一致率: {tingpai_agree:.2f}% ({tingpai_mask.sum()} 条)")
    print(f"Piao 点一致率: {piao_agree:.2f}% ({piao_total} 条)")
    print(f"Piao 点模型仍预测 hu: {piao_pred_hu}/{piao_total} ({piao_pred_hu/piao_total*100:.1f}%)")
    print(f"Piao 点引擎选择 discard: {piao_engine_discard}/{piao_total}")

    # 释放 GPU 内存
    del all_obs
    torch.cuda.empty_cache()

    return {
        "overall_agree": overall_agree,
        "tingpai_agree": tingpai_agree,
        "tingpai_count": int(tingpai_mask.sum()),
        "piao_agree": piao_agree,
        "piao_count": int(piao_total),
        "piao_pred_hu": int(piao_pred_hu),
        "piao_engine_discard": int(piao_engine_discard),
    }


def main():
    ap = argparse.ArgumentParser(description="bc_v7_base piao 过采样微调")
    ap.add_argument("--base-model", default="runs/bc_v7_base.pt")
    ap.add_argument("--data", default="data/audit_v56_with_model.npz")
    ap.add_argument("--finetune-indices", default="data/finetune_indices.npy")
    ap.add_argument("--piao-indices", default="data/piao_indices.npy")
    ap.add_argument("--out", default="runs/bc_v7_piao_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # 加载模型
    model = load_model(args.base_model, device)

    # 加载微调索引
    finetune_indices = np.load(args.finetune_indices)
    piao_indices = np.load(args.piao_indices)
    print(f"微调样本: {len(finetune_indices)}, piao 样本: {len(piao_indices)}")

    # 划分 train/val
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(finetune_indices))
    n_val = int(len(perm) * args.val_ratio)
    val_idx = finetune_indices[perm[:n_val]]
    train_idx = finetune_indices[perm[n_val:]]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 构建数据集
    train_ds = FinetuneDataset(args.data, train_idx, device)
    val_ds = FinetuneDataset(args.data, val_idx, device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_gpu, drop_last=True,
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练循环
    best_val_loss = float("inf")
    best_val_action_acc = 0.0
    patience = 2
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_action_acc": [], "val_tile_acc": []}

    print(f"\n=== 开始微调 ({args.epochs} epochs) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_tile_loss = 0.0
        n_batches = 0

        t0 = time.perf_counter()
        for obs, action_labels, tile_labels in train_loader:
            loss, a_loss, t_loss = compute_loss(model, obs, action_labels, tile_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_action_loss += a_loss
            epoch_tile_loss += t_loss
            n_batches += 1

        dt = time.perf_counter() - t0
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # 验证
        val_metrics = evaluate(model, val_ds, args.batch_size, device, "val")
        val_loss = val_metrics["loss"]
        val_action_acc = val_metrics["action_acc"]
        val_tile_acc = val_metrics["tile_acc"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_action_acc"].append(val_action_acc)
        history["val_tile_acc"].append(val_tile_acc)

        print(
            f"Epoch {epoch}/{args.epochs} ({dt:.1f}s) | "
            f"train_loss={avg_train_loss:.4f} (a={epoch_action_loss/n_batches:.4f} t={epoch_tile_loss/n_batches:.4f}) | "
            f"val_loss={val_loss:.4f} val_action_acc={val_action_acc:.2f}% val_tile_acc={val_tile_acc:.2f}%"
        )

        # 早停：val_loss 上升
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_action_acc = val_action_acc
            no_improve = 0
            # 保存最优模型
            save_checkpoint(model, args, epoch, val_metrics, args.out)
            print(f"  -> 保存最优模型 (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> 早停 (val_loss 连续 {patience} epoch 未改善)")
                break

    # 保存最终模型（如果不是最优）
    # 最优已在循环中保存
    print(f"\n最优 val_loss={best_val_loss:.4f}, val_action_acc={best_val_action_acc:.2f}%")

    # 重新加载最优模型进行全量评估
    print("\n加载最优模型进行全量审计集评估...")
    best_model = load_model(args.out, device)
    audit_metrics = evaluate_full_audit(best_model, args.data, piao_indices, device)

    # 保存微调日志
    log = {
        "args": vars(args),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_action_acc": best_val_action_acc,
        "audit_metrics": audit_metrics,
    }
    log_path = Path(args.out).with_suffix(".log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n微调日志已保存: {log_path}")

    print(f"\n=== 完成 ===")
    print(f"模型: {args.out}")
    print(f"日志: {log_path}")


def save_checkpoint(model, args, epoch, val_metrics, out_path):
    """保存模型 checkpoint。"""
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
        "val_metrics": val_metrics,
    }
    torch.save(ckpt, out_path)


if __name__ == "__main__":
    main()
