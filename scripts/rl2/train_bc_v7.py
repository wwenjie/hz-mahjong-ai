#!/usr/bin/env python
"""BC 训练 v7：Suphx 式独立二分类头 + tile 头，可行性掩码损失。

与 v6 的区别：
- 动作头从「6 类 softmax」改为「5 个独立 sigmoid 二分类头」（hu/gang/peng/chi/discard）
- 训练时只对「该决策点可行的动作」（avail mask）计算 BCE 损失：
  可行且教师选了 → 正样本；可行但教师没选 → 负样本；不可行 → 跳过
- tile 头不变：仅对 discard 样本计算 34 类交叉熵
- 类别不均衡：每个头内部用 pos_weight = 负/正 比例自动加权

数据要求：v7 数据集（含 avail 字段）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model_v7 import MahjongTransformerV7
from nnrl2.model import count_parameters

SCALAR_KEYS = {"god", "wall_remaining", "turn", "phase", "target", "seat", "dealer"}
ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]
HEAD_NAMES = ["hu", "gang", "peng", "chi", "discard"]
# 头索引 → 动作 id
HEAD_TO_ACTION = [4, 3, 2, 1, 0]


def load_to_gpu(npz_path, device, lazy=False):
    data = np.load(npz_path)
    obs = {}
    for k in data.keys():
        if k in ("x_flat", "y", "y_action", "y_tile", "avail"):
            continue
        arr = torch.from_numpy(np.ascontiguousarray(data[k]))
        # lazy=True 时保留在 CPU，训练循环中按需传 GPU
        obs[k] = arr if lazy else arr.to(device)
    y_action = torch.from_numpy(np.ascontiguousarray(data["y_action"])).long()
    y_tile = torch.from_numpy(np.ascontiguousarray(data["y_tile"])).long()
    avail = torch.from_numpy(np.ascontiguousarray(data["avail"])).bool()
    if not lazy:
        y_action = y_action.to(device)
        y_tile = y_tile.to(device)
        avail = avail.to(device)
    return obs, y_action, y_tile, avail


def merge_shards(shard_dir: str, out_path: str):
    shard_dir = Path(shard_dir)
    parts = sorted(shard_dir.glob("train_part*.npz"))
    if not parts:
        print(f"错误: {shard_dir} 下无分片文件", flush=True)
        sys.exit(1)
    print(f"合并 {len(parts)} 个分片 -> {out_path}", flush=True)
    merged = {}
    for p in parts:
        d = np.load(p)
        for k in d.keys():
            merged.setdefault(k, []).append(d[k])
        print(f"  {p.name}: {len(d['y_action'])} 样本", flush=True)
    final = {k: np.concatenate(v) for k, v in merged.items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **final)
    counts = np.bincount(final["y_action"], minlength=6)
    print(f"合并完成: {len(final['y_action'])} 样本", flush=True)
    print(f"动作分布: {'  '.join(f'{n}={c}' for n, c in zip(ACTION_NAMES, counts))}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="BC v7 Suphx 式二分类头训练")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--shard-dir", default="data/bc_v7_parts")
    ap.add_argument("--train-data", default="data/bc_v7_train.npz")
    ap.add_argument("--valid-ratio", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup-epochs", type=int, default=1, help="线性 warmup 到目标 lr 的 epoch 数")
    ap.add_argument("--tile-loss-weight", type=float, default=0.3, help="tile 头 loss 权重（默认 0.3，防止 34 类 softmax 主导梯度）")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default="runs/bc_v7.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--timeout-per-epoch", type=int, default=7200)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if args.merge_only or not Path(args.train_data).exists():
        merge_shards(args.shard_dir, args.train_data)
        if args.merge_only:
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    t0 = time.perf_counter()
    # lazy=True：数据保留在 CPU 内存，训练循环中每个 batch 再传 GPU，避免 142 万样本占满显存
    all_obs, all_action, all_tile, all_avail = load_to_gpu(args.train_data, device, lazy=True)
    n_all = len(all_action)
    n_valid = int(n_all * args.valid_ratio)
    perm = torch.randperm(n_all)  # CPU 索引
    valid_idx, train_idx = perm[:n_valid], perm[n_valid:]
    train_obs = {k: v[train_idx] for k, v in all_obs.items()}
    valid_obs = {k: v[valid_idx] for k, v in all_obs.items()}
    train_action, train_tile, train_avail = all_action[train_idx], all_tile[train_idx], all_avail[train_idx]
    valid_action, valid_tile, valid_avail = all_action[valid_idx], all_tile[valid_idx], all_avail[valid_idx]
    n_train = len(train_action)
    print(f"预加载: train {n_train} + valid {n_valid}，耗时 {time.perf_counter()-t0:.1f}s", flush=True)

    # 每个二分类头的 pos_weight（负/正），仅统计 avail 样本
    pos_weights = []
    for h_idx, act_id in enumerate(HEAD_TO_ACTION):
        mask = train_avail[:, act_id]
        pos = (train_action[mask] == act_id).sum().item()
        neg = mask.sum().item() - pos
        w = neg / max(pos, 1)
        pos_weights.append(min(w, 100.0))  # 防止极端权重
        print(f"  {HEAD_NAMES[h_idx]}: avail={mask.sum().item()} pos={pos} neg={neg} pos_weight={pos_weights[-1]:.2f}", flush=True)
    pos_weight_t = torch.tensor(pos_weights).to(device)

    model = MahjongTransformerV7(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    print(f"参数量: {count_parameters(model):,}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # warmup + cosine decay
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t.unsqueeze(0), reduction="none")
    tile_criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if False and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"resume: epoch {start_epoch} 起", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    def slice_obs(obs, idx):
        batch = {}
        for k, v in obs.items():
            val = v[idx]
            if k in SCALAR_KEYS and val.ndim == 2 and val.shape[1] == 1:
                val = val.squeeze(1)
            batch[k] = val.to(device, non_blocking=True)
        return batch

    def head_targets(y_action, avail):
        """构造 (batch, 5) 的 target 和 mask。"""
        batch_size = len(y_action)
        target = torch.zeros(batch_size, 5, device=y_action.device)
        mask = torch.zeros(batch_size, 5, dtype=torch.bool, device=y_action.device)
        for h_idx, act_id in enumerate(HEAD_TO_ACTION):
            mask[:, h_idx] = avail[:, act_id]
            target[:, h_idx] = (y_action == act_id).float()
        return target, mask

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        perm_t = torch.randperm(n_train)  # CPU 索引
        total_loss = 0.0
        total_bce = 0.0
        total_tloss = 0.0
        nb = 0
        total_batches = (n_train + args.batch_size - 1) // args.batch_size
        for i in range(0, n_train, args.batch_size):
            idx = perm_t[i:i + args.batch_size]
            b_action = train_action[idx].to(device, non_blocking=True)
            b_tile = train_tile[idx].to(device, non_blocking=True)
            b_avail = train_avail[idx].to(device, non_blocking=True)
            b_logits, t_logits, _ = model(slice_obs(train_obs, idx))
            target, mask = head_targets(b_action, b_avail)
            bce = bce_criterion(b_logits, target)  # (batch, 5)
            bce = (bce * mask.float()).sum() / mask.float().sum().clamp(min=1)
            d_mask = b_action == 0
            if d_mask.any():
                t_loss = tile_criterion(t_logits[d_mask], b_tile[d_mask])
            else:
                t_loss = torch.tensor(0.0, device=device)
            loss = bce + args.tile_loss_weight * t_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_bce += bce.item()
            total_tloss += t_loss.item()
            nb += 1
            if nb == 1 or nb % 10 == 0:
                print(f"  batch {nb}/{total_batches} loss {loss.item():.4f} (bce={bce.item():.4f} t={t_loss.item():.4f}) ({time.perf_counter()-t0:.1f}s)", flush=True)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        avg_loss = total_loss / nb
        dt = time.perf_counter() - t0

        if dt > args.timeout_per_epoch:
            print(f"警告: epoch {epoch+1} 耗时 {dt:.0f}s 超限，保存后退出", flush=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss,
                        "args": vars(args)}, out_path.parent / f"{out_path.stem}_ep{epoch+1:02d}_timeout.pt")
            sys.exit(2)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss {avg_loss:.4f} (bce={total_bce/nb:.4f} t={total_tloss/nb:.4f}) | {dt:.1f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss,
                        "args": vars(args)}, out_path)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss,
                    "args": vars(args)}, out_path.parent / f"{out_path.stem}_ep{epoch+1:02d}.pt")

        if (epoch + 1) % 5 == 0:
            eval_req = out_path.parent / f"{out_path.stem}_eval_req.txt"
            eval_req.write_text(f"{out_path.parent / f'{out_path.stem}_ep{epoch+1:02d}.pt'}\n")

    print(f"\n完成。最优 loss: {best_loss:.4f}", flush=True)
    print(f"模型已保存到: {out_path}", flush=True)


if __name__ == "__main__":
    main()
