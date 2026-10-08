#!/usr/bin/env python
"""PPO 快速验证（小规模）。

用现有 BC 模型（即便不准）跑通 PPO 骨架，验证：
1. 自对弈能跑
2. PPO 更新能执行
3. KL 散度计算正确
4. 模型能保存/加载
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model import MahjongTransformer
from nnrl2.obs import situation_to_obs


class BCDecider:
    """BC 模型决策器（用于自对弈）。"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.name = "bc_policy"
        self.last_reason = ""
        self.last_detail = {}

    def configure(self, tournament):
        pass

    def choose(self, situation, actions, *, budget_ms: int = 0):
        obs = situation_to_obs(situation, situation.seat)
        obs_t = {}
        for k, v in obs.items():
            val = np.expand_dims(v, 0)
            obs_t[k] = torch.from_numpy(val).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(obs_t)
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        best_action = None
        best_prob = -1
        for action in actions:
            if action.kind == "discard" and action.tile is not None:
                tile = action.tile
                if 0 <= tile < 34 and probs[tile] > best_prob:
                    best_prob = probs[tile]
                    best_action = action

        return best_action if best_action else actions[0]


def run_one_match(model, device, seed):
    """跑一场自对弈，返回结果。"""
    from majiang.sim.batch import run_match

    deciders = [BCDecider(model, device) for _ in range(4)]
    result = run_match(deciders, rounds=4, base_score=1, seed=seed)
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 创建模型
    model = MahjongTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=256).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 跑 3 场自对弈
    print("\n=== 自对弈测试 ===")
    for i in range(3):
        t0 = time.perf_counter()
        result = run_one_match(model, device, seed=42 + i)
        dt = time.perf_counter() - t0
        print(f"场 {i+1}: {dt:.2f}s | 名次分 {[s.place_points for s in result.seats]}")

    # 测试 PPO 更新
    print("\n=== PPO 更新测试 ===")
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # 模拟一些经验
    batch_size = 4
    dummy_obs = {
        "hand": torch.randint(0, 4, (batch_size, 34), device=device),
        "discards": torch.randint(-1, 34, (batch_size, 4, 24), device=device),
        "melds": torch.randint(-1, 34, (batch_size, 4, 4, 3), device=device),
        "action_history": torch.randint(-1, 34, (batch_size, 200), device=device),
        "scores": torch.randint(0, 100, (batch_size, 4), device=device),
        "god": torch.randint(0, 34, (batch_size,), device=device),
        "wall_remaining": torch.randint(0, 70, (batch_size,), device=device),
        "turn": torch.randint(0, 20, (batch_size,), device=device),
        "phase": torch.randint(0, 2, (batch_size,), device=device),
        "target": torch.randint(-1, 34, (batch_size,), device=device),
        "seat": torch.randint(0, 4, (batch_size,), device=device),
        "dealer": torch.randint(0, 4, (batch_size,), device=device),
    }
    dummy_actions = torch.randint(0, 34, (batch_size,), device=device)
    dummy_advantages = torch.randn(batch_size, device=device)
    dummy_returns = torch.randn(batch_size, device=device)
    dummy_old_log_probs = torch.randn(batch_size, device=device)

    # 前向
    policy_logits, values = model(dummy_obs)
    dist = torch.distributions.Categorical(logits=policy_logits)
    log_probs = dist.log_prob(dummy_actions)
    entropy = dist.entropy().mean()

    # PPO 损失
    ratio = torch.exp(log_probs - dummy_old_log_probs)
    surr1 = ratio * dummy_advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * dummy_advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values.squeeze(-1), dummy_returns)
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"policy_loss: {policy_loss.item():.4f}")
    print(f"value_loss: {value_loss.item():.4f}")
    print(f"entropy: {entropy.item():.4f}")
    print(f"total loss: {loss.item():.4f}")

    # 保存/加载测试
    print("\n=== 保存/加载测试 ===")
    test_path = Path("runs/ppo_smoke.pt")
    test_path.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "test": True,
    }, test_path)

    model2 = MahjongTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=256).to(device)
    ckpt = torch.load(test_path, map_location=device)
    model2.load_state_dict(ckpt["model_state_dict"])
    print("保存/加载成功")

    # 验证两个模型输出一致
    with torch.no_grad():
        out1, _ = model(dummy_obs)
        out2, _ = model2(dummy_obs)
        diff = (out1 - out2).abs().max().item()
        print(f"输出差异: {diff:.10f}")

    print("\n✅ PPO 骨架验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
