#!/usr/bin/env python
"""PPO + KL 锚定训练（Mahjax 风格）。

用法::

    PYTHONPATH=src uv run python scripts/train_ppo.py --bc-model runs/bc_v0.pt --episodes 100

核心设计（对齐 Mahjax）：
- 策略 = BC 预训练的 Transformer
- PPO 裁剪目标 + KL(π‖π_BC) 正则
- 自对弈 4 座位，奖励 = 名次分（place_points）
- 价值头 = critic，GAE 优势估计
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model import MahjongTransformer
from nnrl2.obs import situation_to_obs


class PPOBuffer:
    """PPO 经验回放缓冲区。"""

    def __init__(self):
        self.obs_list = []
        self.action_list = []
        self.log_prob_list = []
        self.reward_list = []
        self.value_list = []
        self.done_list = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs_list.append(obs)
        self.action_list.append(action)
        self.log_prob_list.append(log_prob)
        self.reward_list.append(reward)
        self.value_list.append(value)
        self.done_list.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.reward_list)


class PPOTrainer:
    """PPO + KL 锚定训练器。"""

    def __init__(
        self,
        model: MahjongTransformer,
        bc_model: MahjongTransformer,
        *,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        device: str = "cuda",
    ):
        self.model = model
        self.bc_model = bc_model
        self.device = device

        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # BC 模型设为 eval 模式（不更新）
        self.bc_model.eval()
        for p in self.bc_model.parameters():
            p.requires_grad = False

    def compute_gae(self, rewards, values, dones):
        """计算 GAE 优势。"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, buffer: PPOBuffer, epochs: int = 4, batch_size: int = 32):
        """PPO 更新。"""
        if len(buffer) < batch_size:
            return {}

        # 计算 GAE
        advantages = self.compute_gae(buffer.reward_list, buffer.value_list, buffer.done_list)
        returns = [adv + val for adv, val in zip(advantages, buffer.value_list)]

        # 转 tensor
        obs_batch = self._collate_obs(buffer.obs_list)
        actions = torch.tensor(buffer.action_list, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(buffer.log_prob_list, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 标准化优势
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO epochs
        n = len(buffer)
        indices = np.arange(n)
        total_loss = 0
        total_kl = 0
        n_updates = 0

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                batch_obs = {k: v[idx] for k, v in obs_batch.items()}
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages_t[idx]
                batch_returns = returns_t[idx]

                # 前向
                policy_logits, values = self.model(batch_obs)
                dist = torch.distributions.Categorical(logits=policy_logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # KL 散度（对 BC 模型）
                with torch.no_grad():
                    bc_logits, _ = self.bc_model(batch_obs)
                kl_div = F.kl_div(
                    F.log_softmax(policy_logits, dim=-1),
                    F.softmax(bc_logits, dim=-1),
                    reduction="batchmean",
                )

                # PPO 裁剪损失
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # 总损失
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.kl_coef * kl_div
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_kl += kl_div.item()
                n_updates += 1

        buffer.clear()
        return {
            "loss": total_loss / max(n_updates, 1),
            "kl": total_kl / max(n_updates, 1),
        }

    def _collate_obs(self, obs_list):
        """把 obs list 合并成 batch。"""
        keys = obs_list[0].keys()
        result = {}
        for key in keys:
            values = [obs[key] for obs in obs_list]
            if isinstance(values[0], np.ndarray):
                result[key] = torch.from_numpy(np.stack(values)).to(self.device)
            else:
                result[key] = torch.tensor(values, device=self.device)
        return result


def run_episode(model, bc_model, buffer, device, seed):
    """跑一局自对弈，收集经验。"""
    from majiang.sim.batch import run_match
    from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

    # 用 BC 模型包装成 decider
    class BCDecider:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self.name = "bc_policy"

        def choose(self, situation, actions, *, budget_ms: int = 0):
            obs = situation_to_obs(situation, situation.seat)
            # 转 tensor
            obs_t = {k: torch.from_numpy(np.expand_dims(v, 0)).to(self.device) for k, v in obs.items()}
            policy_logits, value = self.model(obs_t)
            dist = torch.distributions.Categorical(logits=policy_logits)
            action = dist.sample()

            # 找对应的合法动作
            chosen_tile = action.item()
            for a in actions:
                if a.kind == "discard" and a.tile == chosen_tile:
                    return a
            # 如果没找到，返回第一个合法动作
            return actions[0] if actions else None

    deciders = [BCDecider(model, device) for _ in range(4)]
    result = run_match(deciders, rounds=8, base_score=1, seed=seed)

    # 记录奖励（名次分）
    for seat in range(4):
        buffer.reward_list.append(result.seats[seat].place_points)
        buffer.done_list.append(True)

    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO + KL 锚定训练")
    ap.add_argument("--bc-model", default="runs/bc_v0.pt")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="runs/ppo_v0.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载 BC 模型
    bc_model = MahjongTransformer().to(device)
    if Path(args.bc_model).exists():
        ckpt = torch.load(args.bc_model, map_location=device)
        bc_model.load_state_dict(ckpt["model_state_dict"])
        print(f"加载 BC 模型: {args.bc_model} (acc={ckpt.get('valid_acc', 0):.4f})")
    else:
        print(f"警告: BC 模型不存在 {args.bc_model}，使用随机初始化")

    # 创建策略模型（从 BC 初始化）
    model = MahjongTransformer().to(device)
    model.load_state_dict(bc_model.state_dict())

    # PPO 训练器
    trainer = PPOTrainer(model, bc_model, lr=args.lr, clip_eps=args.clip_eps, kl_coef=args.kl_coef, device=device)

    # 训练循环
    buffer = PPOBuffer()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(args.episodes):
        t0 = time.perf_counter()

        # 收集经验
        result = run_episode(model, bc_model, buffer, device, seed=args.seed + episode)

        # PPO 更新
        stats = trainer.update(buffer, batch_size=args.batch_size)

        dt = time.perf_counter() - t0
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | loss {stats.get('loss', 0):.4f} | KL {stats.get('kl', 0):.4f} | {dt:.1f}s")

        # 定期保存
        if episode % 50 == 0 and episode > 0:
            torch.save({
                "episode": episode,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
            }, out_path)

    torch.save({
        "episode": args.episodes,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, out_path)
    print(f"\n完成。模型已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
