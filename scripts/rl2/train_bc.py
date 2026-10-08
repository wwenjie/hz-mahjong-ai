#!/usr/bin/env python
"""BC 训练脚本（Transformer）。

用法::

    PYTHONPATH=src uv run python scripts/train_bc.py --epochs 20 --batch-size 32
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model import MahjongTransformer, count_parameters


class BCDataset(Dataset):
    """BC 数据集。"""

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
            # 0-d array 需要 reshape 成 scalar
            if val.ndim == 0:
                val = val.reshape(1)
            obs[k] = torch.from_numpy(val)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return obs, y


def collate_fn(batch):
    """把 batch 的 dict 合并。scalar 字段 squeeze 成 (batch,)。"""
    obs = {}
    scalar_keys = {"god", "wall_remaining", "turn", "phase", "target", "seat", "dealer"}
    for key in batch[0][0].keys():
        stacked = torch.stack([item[0][key] for item in batch])
        if key in scalar_keys and stacked.ndim == 2 and stacked.shape[1] == 1:
            stacked = stacked.squeeze(1)  # (batch, 1) -> (batch,)
        obs[key] = stacked
    y = torch.stack([item[1] for item in batch])
    return obs, y


def evaluate(model, dataloader, device):
    """评估模型在 held-out 集上的准确率。"""
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
    ap = argparse.ArgumentParser(description="BC 训练（Transformer）")
    ap.add_argument("--train-data", default="data/bc_train.npz")
    ap.add_argument("--valid-data", default="data/bc_valid.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default="runs/bc_v0.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 数据
    train_ds = BCDataset(args.train_data)
    valid_ds = BCDataset(args.valid_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"训练样本: {len(train_ds)}, 验证样本: {len(valid_ds)}")

    # 模型
    model = MahjongTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    n_params = count_parameters(model)
    print(f"模型参数量: {n_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 损失
    criterion = nn.CrossEntropyLoss()

    # 训练
    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

        # 验证
        valid_acc = evaluate(model, valid_loader, device)
        dt = time.perf_counter() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss {avg_loss:.4f} | valid acc {valid_acc:.4f} | {dt:.1f}s")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_acc": valid_acc,
                "args": vars(args),
            }, out_path)
            print(f"  -> 保存最优模型 (acc={best_acc:.4f})")

    print(f"\n完成。最优验证准确率: {best_acc:.4f}")
    print(f"模型已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
