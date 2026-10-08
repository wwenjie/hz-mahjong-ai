#!/usr/bin/env python
"""BC v3 训练：多任务学习（动作 + tile + final_score 回归 + other_hands 预测）。

基于论文：
  - Suphx (arXiv:2003.13590) §3.3: global reward prediction 辅助任务
  - PerfectDou (arXiv:2203.16406): PTIE 完美信息作为辅助目标（不是输入）

数据: bc_oracle_train.npz（含 final_scores, other_hands, wall_counts, match_id）
模型: MahjongTransformerV7 + 辅助头（final_score 回归 + other_hands 预测）

用法::

    PYTHONPATH=src python scripts/train_bc_v3.py --train-data data/bc_oracle_train.npz
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

# head 顺序 → 动作 id（与 model_v7.HEAD_ACTIONS / train_bc_v7 一致）
HEAD_TO_ACTION = [4, 3, 2, 1, 0]  # hu, gang, peng, chi, discard
HEAD_NAMES = ("hu", "gang", "peng", "chi", "discard")


def head_targets(y_action, avail):
    """构造 (batch, 5) 的 target 和 mask（与 train_bc_v7.py 一致）。

    target[:, h] = 1 当且仅当教师选了 head h 对应的动作；
    mask[:, h] = 1 当且仅当该动作在此决策点可行（avail）。
    pass（y_action=5）不设头——所有头 target=0 即教师选择 pass。
    """
    batch_size = len(y_action)
    target = torch.zeros(batch_size, 5, device=y_action.device)
    mask = torch.zeros(batch_size, 5, dtype=torch.bool, device=y_action.device)
    for h_idx, act_id in enumerate(HEAD_TO_ACTION):
        mask[:, h_idx] = avail[:, act_id]
        target[:, h_idx] = (y_action == act_id).float()
    return target, mask


class MahjongV3MultiTask(nn.Module):
    """v7 主干 + 辅助头：final_score 回归 + other_hands 预测（PerfectDou PTIE）。"""

    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.backbone = MahjongTransformerV7(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        # 辅助头 1：final_score 回归（4 家净分）
        self.final_score_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )
        # 辅助头 2：other_hands 预测（3 家 × 34 牌分布）
        self.other_hands_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 3 * 34),
        )

    def forward(self, obs):
        # 主干 pooled 特征（复用 _backbone，不跑 heads 省算力）
        pooled = self.backbone._backbone(obs)  # (batch, d_model)
        # 主任务头
        binary_logits = torch.cat([h(pooled) for h in self.backbone.heads()], dim=1)  # (batch, 5)
        tile_logits = self.backbone.tile_head(pooled)  # (batch, 34)
        value = self.backbone.value_head(pooled)       # (batch, 1)
        # 辅助头
        final_score = self.final_score_head(pooled)    # (batch, 4)
        other_hands = self.other_hands_head(pooled).view(-1, 3, 34)  # (batch, 3, 34)
        return binary_logits, tile_logits, value, final_score, other_hands

    def load_v7_checkpoint(self, ckpt_path, device):
        """热启动：加载 v7 checkpoint 到 backbone。"""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"热启动: {ckpt_path} -> backbone")


def load_oracle_data(path: str):
    """加载 oracle 数据（含辅助标签）。"""
    d = np.load(path)
    obs_keys = [k for k in d.files if k not in ("y_action", "y_tile", "avail", "x_flat", "final_scores", "other_hands", "wall_counts", "match_id")]
    obs = {k: d[k] for k in obs_keys}
    y_action = d["y_action"]
    y_tile = d["y_tile"]
    avail = d["avail"]
    final_scores = d["final_scores"]  # (N, 4)
    other_hands = d["other_hands"]     # (N, 3, 34)
    return obs, y_action, y_tile, avail, final_scores, other_hands


def main():
    ap = argparse.ArgumentParser(description="BC v3 多任务训练（动作+tile+final_score+other_hands）")
    ap.add_argument("--train-data", default="data/bc_oracle_train.npz")
    ap.add_argument("--valid-ratio", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--tile-loss-weight", type=float, default=0.3)
    ap.add_argument("--final-score-weight", type=float, default=0.1, help="final_score 回归 loss 权重")
    ap.add_argument("--other-hands-weight", type=float, default=0.1, help="other_hands 预测 loss 权重")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default="runs/bc_v3.pt")
    ap.add_argument("--init-from", default=None, help="v7 checkpoint 路径（热启动）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    obs, y_action, y_tile, avail, final_scores, other_hands = load_oracle_data(args.train_data)
    n = len(y_action)
    print(f"样本: {n}, 特征键: {list(obs.keys())}")

    # 划分 train/valid
    idx = np.random.permutation(n)
    n_valid = int(n * args.valid_ratio)
    valid_idx, train_idx = idx[:n_valid], idx[n_valid:]

    def subset(indices):
        return (
            {k: v[indices] for k, v in obs.items()},
            y_action[indices], y_tile[indices], avail[indices],
            final_scores[indices], other_hands[indices],
        )

    train_data = subset(train_idx)
    valid_data = subset(valid_idx)

    # 模型
    model = MahjongV3MultiTask(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M")

    # 热启动（可选）
    if args.init_from:
        model.load_v7_checkpoint(args.init_from, device)

    # 优化器 + warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)

    # 每个二分类头的 pos_weight（负/正），仅统计 avail 样本（与 v7 一致，处理类别不均衡）
    tr_action_np, tr_avail_np = train_data[1], train_data[3]
    pos_weights = []
    for h_idx, act_id in enumerate(HEAD_TO_ACTION):
        m = tr_avail_np[:, act_id].astype(bool)
        pos = int((tr_action_np[m] == act_id).sum())
        neg = int(m.sum()) - pos
        pos_weights.append(neg / max(pos, 1))
        print(f"  {HEAD_NAMES[h_idx]}: avail={int(m.sum())} pos={pos} neg={neg} pos_weight={pos_weights[-1]:.2f}")
    pos_weight_t = torch.tensor(pos_weights, dtype=torch.float32, device=device)

    # loss
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t.unsqueeze(0), reduction="none")
    tile_criterion = nn.CrossEntropyLoss()
    huber_criterion = nn.HuberLoss(delta=1.0)   # final_score 回归（归一化后，Huber 抗离群）
    mse_criterion = nn.MSELoss()                # other_hands 牌计数回归（[0,4] 小数值，MSE 即可）

    FS_SCALE = 50.0  # final_scores 归一化系数（实测 std≈35.8，范围 [-96,242]）

    print(f"开始训练: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_bce = total_tloss = total_fs = total_oh = 0.0
        nb = 0
        t0 = time.perf_counter()

        # 打乱训练数据
        perm = np.random.permutation(len(train_data[1]))
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            b_obs = {k: torch.from_numpy(v[batch_idx]).to(device) for k, v in train_data[0].items()}
            b_action = torch.from_numpy(train_data[1][batch_idx]).to(device)
            b_tile = torch.from_numpy(train_data[2][batch_idx]).to(device)
            b_avail = torch.from_numpy(train_data[3][batch_idx]).to(device)
            b_fs = torch.from_numpy(train_data[4][batch_idx]).float().to(device)
            b_oh = torch.from_numpy(train_data[5][batch_idx]).float().to(device)

            b_logits, t_logits, value, pred_fs, pred_oh = model(b_obs)

            # 主任务：5 头二分类（head 顺序 hu/gang/peng/chi/discard，pass 无头）+ tile 34 分类
            target, mask = head_targets(b_action, b_avail)
            bce = bce_criterion(b_logits, target)
            bce = (bce * mask.float()).sum() / mask.float().sum().clamp(min=1)

            d_mask = b_action == 0
            if d_mask.any():
                t_loss = tile_criterion(t_logits[d_mask], b_tile[d_mask])
            else:
                t_loss = torch.tensor(0.0, device=device)

            # 辅助任务 1：final_score 回归（归一化 + Huber，防止大数值淹没主任务）
            fs_loss = huber_criterion(pred_fs / FS_SCALE, b_fs / FS_SCALE)

            # 辅助任务 2：other_hands 预测（MSE，牌计数回归）
            oh_loss = mse_criterion(pred_oh, b_oh)

            # 总 loss
            loss = (bce + args.tile_loss_weight * t_loss
                    + args.final_score_weight * fs_loss
                    + args.other_hands_weight * oh_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_tloss += t_loss.item()
            total_fs += fs_loss.item()
            total_oh += oh_loss.item()
            nb += 1

            if nb % 50 == 0:
                print(f"  batch {nb:4d} loss {total_loss/nb:.4f} (bce={total_bce/nb:.4f} t={total_tloss/nb:.4f} fs={total_fs/nb:.4f} oh={total_oh/nb:.4f}) ({time.perf_counter()-t0:.1f}s)")

        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        dt = time.perf_counter() - t0
        print(f"Epoch {epoch+1:2d}/{args.epochs} | loss {total_loss/nb:.4f} | {dt:.1f}s")

        # 保存 checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt = args.out.replace(".pt", f"_ep{epoch+1:02d}.pt")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1, "args": vars(args)}, ckpt)
            print(f"  保存 {ckpt}")

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs, "args": vars(args)}, args.out)
    print(f"训练完成 -> {args.out}")


if __name__ == "__main__":
    main()
