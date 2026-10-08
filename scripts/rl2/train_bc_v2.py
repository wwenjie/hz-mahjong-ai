#!/usr/bin/env python
"""BC 训练 v2：合并 train/valid，用 train loss 判断收敛。

用法::

    PYTHONPATH=src uv run python scripts/train_bc_v2.py --epochs 30 --batch-size 64
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
from torch.utils.data import DataLoader, Dataset, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model import MahjongTransformer, count_parameters


class BCDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.obs = {}
        for key in data.keys():
            if key not in ("x_flat", "y"):
                self.obs[key] = data[key]
        self.y = data["y"]
        self.n = len(self.y)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        obs = {}
        for k, v in self.obs.items():
            val = v[idx]
            if val.ndim == 0:
                val = val.reshape(1)
            obs[k] = torch.from_numpy(val)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return obs, y


def collate_fn(batch):
    obs = {}
    scalar_keys = {"god", "wall_remaining", "turn", "phase", "target", "seat", "dealer"}
    for key in batch[0][0].keys():
        stacked = torch.stack([item[0][key] for item in batch])
        if key in scalar_keys and stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.squeeze(1)
        obs[key] = stacked
    y = torch.stack([item[1] for item in batch])
    return obs, y


def evaluate(model, dataloader, device):
    """评估模型在数据集上的 top-1 准确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for obs, y in dataloader:
            obs = {k: v.to(device) for k, v in obs.items()}
            y = y.to(device)
            policy_logits, _ = model(obs)
            pred = policy_logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/bc_train.npz")
    ap.add_argument("--valid-data", default="data/bc_valid.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default="runs/bc_v1.pt")
    ap.add_argument("--save-every-epoch", action="store_true", default=True,
                    help="每个 epoch 保存一份 checkpoint（默认开启）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 合并 train + valid
    train_ds = BCDataset(args.train_data)
    valid_ds = BCDataset(args.valid_data)
    full_ds = ConcatDataset([train_ds, valid_ds])
    train_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # 用 train 集的一部分做验证（避免 valid 集太小）
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"总样本: {len(full_ds)} (train {len(train_ds)} + valid {len(valid_ds)})")

    # 模型
    model = MahjongTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    print(f"参数量: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0

        for obs, y in train_loader:
            obs = {k: v.to(device) for k, v in obs.items()}
            y = y.to(device)

            policy_logits, _ = model(obs)
            loss = criterion(policy_logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        dt = time.perf_counter() - t0

        # 每 5 个 epoch 评估一次
        if (epoch + 1) % 5 == 0:
            train_acc = evaluate(model, train_eval_loader, device)
            print(f"Epoch {epoch+1:3d}/{args.epochs} | loss {avg_loss:.4f} | train acc {train_acc:.4f} | {dt:.1f}s")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | loss {avg_loss:.4f} | {dt:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "args": vars(args),
            }, out_path)

        # 每个 epoch 保存一份 checkpoint，防止中断后没有可用模型
        if args.save_every_epoch:
            epoch_path = out_path.parent / f"{out_path.stem}_ep{epoch+1:02d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "args": vars(args),
            }, epoch_path)

    # 最终评估
    final_acc = evaluate(model, train_eval_loader, device)
    print(f"\n完成。最优 train loss: {best_loss:.4f}，最终 train acc: {final_acc:.4f}")
    print(f"模型已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
