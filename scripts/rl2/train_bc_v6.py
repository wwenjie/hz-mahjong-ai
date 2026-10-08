#!/usr/bin/env python
"""BC 训练 v6：双头模型（动作类型 6 类 + tile 34 类），类别加权损失。

解决 v5 单头网络被 discard 样本（77%）淹没导致学不会吃碰胡的问题。
设计依据 Suphx 分离模型范式（见 records/research_mahjong_rl_20261002.md）。

损失 = action_loss（类别加权） + tile_loss（仅 discard 样本，类别加权）
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

from nnrl2.model_v6 import MahjongTransformerV6
from nnrl2.model import count_parameters

SCALAR_KEYS = {"god", "wall_remaining", "turn", "phase", "target", "seat", "dealer"}
ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]


def load_to_gpu(npz_path, device):
    data = np.load(npz_path)
    obs = {}
    for k in data.keys():
        if k in ("x_flat", "y", "y_action", "y_tile"):
            continue
        arr = torch.from_numpy(np.ascontiguousarray(data[k]))
        obs[k] = arr.to(device)
    y_action = torch.from_numpy(np.ascontiguousarray(data["y_action"])).long().to(device)
    y_tile = torch.from_numpy(np.ascontiguousarray(data["y_tile"])).long().to(device)
    return obs, y_action, y_tile


def merge_shards(shard_dir: str, out_path: str):
    """合并 v6 分片为单个训练文件。"""
    shard_dir = Path(shard_dir)
    parts = sorted(shard_dir.glob("train_part*.npz"))
    if not parts:
        print(f"错误: {shard_dir} 下无分片文件", flush=True)
        sys.exit(1)
    print(f"合并 {len(parts)} 个分片 -> {out_path}", flush=True)
    merged = {}
    for i, p in enumerate(parts):
        d = np.load(p)
        for k in d.keys():
            if k not in merged:
                merged[k] = [d[k]]
            else:
                merged[k].append(d[k])
        print(f"  {p.name}: {len(d['y_action'])} 样本", flush=True)
    final = {k: np.concatenate(v) for k, v in merged.items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **final)
    counts = np.bincount(final["y_action"], minlength=6)
    print(f"合并完成: {len(final['y_action'])} 样本", flush=True)
    print(f"动作分布: {'  '.join(f'{n}={c}' for n, c in zip(ACTION_NAMES, counts))}", flush=True)


def compute_class_weights(y_action: torch.Tensor) -> torch.Tensor:
    """按频率倒数计算动作类别权重（稀有动作权重更高）。"""
    counts = torch.bincount(y_action, minlength=6).float()
    # 逆频率，归一化使平均权重为 1
    weights = counts.sum() / (counts.clamp(min=1) * len(counts))
    return weights


def main():
    ap = argparse.ArgumentParser(description="BC v6 双头训练")
    ap.add_argument("--merge-only", action="store_true", help="仅合并分片，不训练")
    ap.add_argument("--shard-dir", default="data/bc_v6_parts")
    ap.add_argument("--train-data", default="data/bc_v6_train.npz")
    ap.add_argument("--valid-ratio", type=float, default=0.05, help="验证集比例")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default="runs/bc_v6.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--timeout-per-epoch", type=int, default=600)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 固定输入形状，启用 cudnn autotuner 提速
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # 合并分片
    if args.merge_only or not Path(args.train_data).exists():
        merge_shards(args.shard_dir, args.train_data)
        if args.merge_only:
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    # 全量预加载
    t0 = time.perf_counter()
    all_obs, all_action, all_tile = load_to_gpu(args.train_data, device)
    n_all = len(all_action)
    # 划分训练/验证
    n_valid = int(n_all * args.valid_ratio)
    perm = torch.randperm(n_all, device=device)
    valid_idx, train_idx = perm[:n_valid], perm[n_valid:]
    train_obs = {k: v[train_idx] for k, v in all_obs.items()}
    valid_obs = {k: v[valid_idx] for k, v in all_obs.items()}
    train_action, train_tile = all_action[train_idx], all_tile[train_idx]
    valid_action, valid_tile = all_action[valid_idx], all_tile[valid_idx]
    n_train = len(train_action)
    print(f"预加载: train {n_train} + valid {n_valid}，耗时 {time.perf_counter()-t0:.1f}s", flush=True)

    # 类别权重
    action_weights = compute_class_weights(train_action)
    print(f"动作类别权重: {'  '.join(f'{n}={w:.2f}' for n, w in zip(ACTION_NAMES, action_weights.tolist()))}", flush=True)

    model = MahjongTransformerV6(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    print(f"参数量: {count_parameters(model):,}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    action_criterion = nn.CrossEntropyLoss(weight=action_weights)
    tile_criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
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
            batch[k] = val
        return batch

    @torch.no_grad()
    def evaluate(obs, y_action, y_tile, bs=2048):
        model.eval()
        n = len(y_action)
        act_correct = 0
        tile_correct = 0
        # 分动作类型的准确率
        per_action_correct = torch.zeros(6, dtype=torch.long, device=device)
        per_action_total = torch.zeros(6, dtype=torch.long, device=device)
        for i in range(0, n, bs):
            idx = torch.arange(i, min(i + bs, n), device=device)
            a_logits, t_logits, _ = model(slice_obs(obs, idx))
            a_pred = a_logits.argmax(-1)
            act_correct += (a_pred == y_action[idx]).sum().item()
            # tile 准确率仅统计 discard 样本
            d_mask = y_action[idx] == 0
            if d_mask.any():
                t_pred = t_logits.argmax(-1)
                tile_correct += (t_pred[d_mask] == y_tile[idx][d_mask]).sum().item()
            for a in range(6):
                m = y_action[idx] == a
                per_action_total[a] += m.sum()
                per_action_correct[a] += (a_pred[m] == a).sum() if m.any() else 0
        model.train()
        d_total = per_action_total[0].item()
        per_acc = {ACTION_NAMES[a]: (per_action_correct[a].item() / per_action_total[a].item() if per_action_total[a] > 0 else 0) for a in range(6)}
        return act_correct / n, (tile_correct / d_total if d_total > 0 else 0), per_acc

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        perm_t = torch.randperm(n_train, device=device)
        total_loss = 0.0
        total_aloss = 0.0
        total_tloss = 0.0
        nb = 0
        total_batches = (n_train + args.batch_size - 1) // args.batch_size
        for i in range(0, n_train, args.batch_size):
            idx = perm_t[i:i + args.batch_size]
            a_logits, t_logits, _ = model(slice_obs(train_obs, idx))
            # 动作类型损失（全部样本）
            a_loss = action_criterion(a_logits, train_action[idx])
            # tile 损失（仅 discard 样本）
            d_mask = train_action[idx] == 0
            if d_mask.any():
                t_loss = tile_criterion(t_logits[d_mask], train_tile[idx][d_mask])
            else:
                t_loss = torch.tensor(0.0, device=device)
            loss = a_loss + t_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_aloss += a_loss.item()
            total_tloss += t_loss.item()
            nb += 1
            if nb == 1 or nb % 10 == 0:
                print(f"  batch {nb}/{total_batches} loss {loss.item():.4f} (a={a_loss.item():.4f} t={t_loss.item():.4f}) ({time.perf_counter()-t0:.1f}s)", flush=True)
        scheduler.step()
        avg_loss = total_loss / nb
        dt = time.perf_counter() - t0

        if dt > args.timeout_per_epoch:
            print(f"警告: epoch {epoch+1} 耗时 {dt:.0f}s 超限，保存后退出", flush=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss,
                        "args": vars(args)}, out_path.parent / f"{out_path.stem}_ep{epoch+1:02d}_timeout.pt")
            sys.exit(2)

        # 训练期不跑 evaluate：evaluate 在训练循环中触发 WSL2/CUDA 死锁（train_bc_v3 已三次复现）。
        # 改为每 5 epoch 保存 checkpoint，训练结束后用 CPU 单独评估。
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss {avg_loss:.4f} (a={total_aloss/nb:.4f} t={total_tloss/nb:.4f}) | {dt:.1f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss,
                        "args": vars(args)}, out_path)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss,
                    "args": vars(args)}, out_path.parent / f"{out_path.stem}_ep{epoch+1:02d}.pt")

        # 每 5 epoch 写一次评估请求文件，供外部 CPU 评估脚本消费
        if (epoch + 1) % 5 == 0:
            eval_req = out_path.parent / f"{out_path.stem}_eval_req.txt"
            eval_req.write_text(f"{out_path.parent / f'{out_path.stem}_ep{epoch+1:02d}.pt'}\n")

    print(f"\n完成。最优 loss: {best_loss:.4f}", flush=True)
    print(f"模型已保存到: {out_path}", flush=True)


if __name__ == "__main__":
    main()
