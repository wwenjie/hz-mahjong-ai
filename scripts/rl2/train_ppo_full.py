#!/usr/bin/env python
"""PPO + KL 锚定训练（完整版）。

用法::

    PYTHONPATH=src uv run python scripts/train_ppo_full.py --bc-model runs/bc_v2.pt --episodes 200

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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model import MahjongTransformer
from nnrl2.obs import situation_to_obs


class PolicyDecider:
    """策略模型决策器（用于自对弈收集经验）。"""

    def __init__(self, model, device, deterministic=False):
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.name = "policy"
        self.last_reason = ""
        self.last_detail = {}
        # 记录本步的选择，供 buffer 收集
        self.last_obs = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        # 完整轨迹：每一步都记录
        self.traj_obs = []
        self.traj_action = []
        self.traj_log_prob = []
        self.traj_value = []

    def configure(self, tournament):
        pass

    def choose(self, situation, actions, *, budget_ms: int = 0):
        # 规则兜底：能胡必胡（麻将里几乎总是最优）。规则动作不记入轨迹——
        # 不是模型做的决策，PPO 只学模型自己的出牌选择。
        for action in actions:
            if action.kind == "hu":
                return action

        obs = situation_to_obs(situation, situation.seat)
        obs_t = {}
        for k, v in obs.items():
            val = np.expand_dims(v, 0)
            obs_t[k] = torch.from_numpy(val).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(obs_t)
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            log_probs = torch.log_softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        # 找合法出牌动作
        legal_discards = []
        for action in actions:
            if action.kind == "discard" and action.tile is not None:
                tile = action.tile
                if 0 <= tile < 34:
                    legal_discards.append((action, tile, probs[tile]))

        if not legal_discards:
            return actions[0] if actions else None

        if self.deterministic:
            # 选概率最高的
            best_action, best_tile, _ = max(legal_discards, key=lambda x: x[2])
            chosen_action, chosen_tile = best_action, best_tile
        else:
            # 按概率采样
            probs_legal = np.array([p for _, _, p in legal_discards])
            probs_legal = probs_legal / probs_legal.sum()
            idx = np.random.choice(len(legal_discards), p=probs_legal)
            chosen_action, chosen_tile, _ = legal_discards[idx]

        # 记录本步信息
        self.last_obs = obs
        self.last_action = chosen_tile
        self.last_log_prob = log_probs[chosen_tile]
        self.last_value = value.item()
        # 追加到完整轨迹
        self.traj_obs.append(obs)
        self.traj_action.append(chosen_tile)
        self.traj_log_prob.append(log_probs[chosen_tile])
        self.traj_value.append(value.item())

        return chosen_action


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


def compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=0.95):
    """计算 GAE 优势。"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


def collate_obs(obs_list, device):
    """把 obs list 合并成 batch。"""
    keys = obs_list[0].keys()
    result = {}
    for key in keys:
        values = [obs[key] for obs in obs_list]
        if isinstance(values[0], np.ndarray):
            result[key] = torch.from_numpy(np.stack(values)).to(device)
        else:
            result[key] = torch.tensor(values, device=device)
    return result


def collect_episode(model, device, seed, rounds=8):
    """跑一局自对弈，收集经验。"""
    from majiang.sim.batch import run_match

    buffer = PPOBuffer()
    deciders = [PolicyDecider(model, device, deterministic=False) for _ in range(4)]
    result = run_match(deciders, rounds=rounds, base_score=1, seed=seed)

    # 从每个 decider 的完整轨迹中提取经验
    for seat, decider in enumerate(deciders):
        reward = result.seats[seat].place_points
        # 每一步都作为一条经验，done=1 只在最后一步
        for t in range(len(decider.traj_obs)):
            is_last = (t == len(decider.traj_obs) - 1)
            buffer.add(
                obs=decider.traj_obs[t],
                action=decider.traj_action[t],
                log_prob=decider.traj_log_prob[t],
                reward=reward if is_last else 0.0,
                value=decider.traj_value[t],
                done=is_last,
            )

    return buffer, result


def ppo_update(model, bc_model, buffer, optimizer, device, *, clip_eps=0.2, kl_coef=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4, batch_size=32):
    """PPO 更新。"""
    if len(buffer) < batch_size:
        return {}

    # 计算 GAE
    advantages = compute_gae(buffer.reward_list, buffer.value_list, buffer.done_list)
    returns = [adv + val for adv, val in zip(advantages, buffer.value_list)]

    # 转 tensor
    obs_batch = collate_obs(buffer.obs_list, device)
    actions = torch.tensor(buffer.action_list, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_prob_list, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

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
            policy_logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=policy_logits)
            log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # KL 散度（对 BC 模型）
            with torch.no_grad():
                bc_logits, _ = bc_model(batch_obs)
            kl_div = F.kl_div(
                F.log_softmax(policy_logits, dim=-1),
                F.softmax(bc_logits, dim=-1),
                reduction="batchmean",
            )

            # PPO 裁剪损失
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

            # 总损失
            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + kl_coef * kl_div
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_kl += kl_div.item()
            n_updates += 1

    buffer.clear()
    return {
        "loss": total_loss / max(n_updates, 1),
        "kl": total_kl / max(n_updates, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO + KL 锚定训练")
    ap.add_argument("--bc-model", default="runs/bc_v2.pt")
    ap.add_argument("--episodes", type=int, default=200)
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
        print(f"加载 BC 模型: {args.bc_model} (train_loss={ckpt.get('train_loss', 0):.4f})")
    else:
        print(f"警告: BC 模型不存在 {args.bc_model}，使用随机初始化")

    # 创建策略模型（从 BC 初始化）
    model = MahjongTransformer().to(device)
    model.load_state_dict(bc_model.state_dict())

    # PPO 训练器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 训练循环
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(args.episodes):
        t0 = time.perf_counter()

        # 收集经验
        buffer, result = collect_episode(model, device, seed=args.seed + episode)

        # PPO 更新
        stats = ppo_update(
            model, bc_model, buffer, optimizer, device,
            clip_eps=args.clip_eps, kl_coef=args.kl_coef, batch_size=args.batch_size,
        )

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
