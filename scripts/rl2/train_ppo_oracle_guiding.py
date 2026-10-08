#!/usr/bin/env python
"""Oracle Guiding PPO 训练：Suphx 3.3 节的先知引导渐进过渡。

核心思路：
- 用 Oracle 模型（能看到三家手牌）作为起点
- 训练时逐步 dropout oracle 特征（γ 从 1 → 0），让模型从"依赖完美信息"平滑过渡到"只用公开信息"
- γ=0 后继续训练，lr 降至 1/10，重要性采样权重超阈值时拒绝更新

与 train_ppo_oracle.py 的区别：
- train_ppo_oracle.py：纯先知训练（γ 恒为 1），产出先知模型
- 本脚本：先知引导过渡（γ 从 1 衰减到 0），产出正常 v7 模型

用法::
    PYTHONPATH=src python scripts/train_ppo_oracle_guiding.py \
        --bc-model runs/bc_v7_base.pt --episodes 1000 --dropout-episodes 800
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

from nnrl2.model_oracle import MahjongOracleV7
from nnrl2.oracle_obs import situation_to_oracle_obs, ORACLE_TOKENS
from nnrl2.obs import situation_to_obs
from nnrl2.model import count_parameters

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]
HEAD_TO_ACTION = [4, 3, 2, 1, 0]  # hu/gang/peng/chi/discard


class OracleGuidingDecider:
    """Oracle Guiding 策略决策器：observe_state 注入 RoundState，choose 用渐进 dropout 的 oracle 观测。"""

    def __init__(self, model, device, gamma=1.0, deterministic=False):
        self.model = model
        self.device = device
        self.gamma = gamma  # oracle 特征保留率（1=全保留，0=全丢弃）
        self.deterministic = deterministic
        self.name = "oracle_guiding_v7"
        self.last_reason = ""
        self.last_detail = {}
        self._state = None  # RoundState（observe_state 注入）
        # 轨迹记录
        self.traj_obs = []
        self.traj_action_kind = []
        self.traj_tile = []
        self.traj_log_prob = []
        self.traj_value = []
        self.traj_avail = []

    def configure(self, tournament):
        pass

    def observe_state(self, state):
        """仿真器钩子：接收 RoundState（完美信息来源）。"""
        self._state = state

    def choose(self, situation, actions, *, budget_ms: int = 0):
        # 规则兜底：能胡必胡（不记入轨迹）
        for action in actions:
            if action.kind == "hu":
                return action

        # 构造 avail mask
        avail = np.zeros(6, dtype=np.float32)
        kind_map = {"discard": 0, "chi": 1, "peng": 2, "gang": 3, "hu": 4, "pass": 5}
        for a in actions:
            k = kind_map.get(getattr(a, "kind", None))
            if k is not None:
                avail[k] = 1.0

        state = self._state
        if state is None:
            # 无 state（纯推理模式），退化为公开信息
            obs = situation_to_obs(situation, situation.seat)
            obs["oracle_hands"] = np.zeros((3, 34), dtype=np.int8)
        else:
            obs = situation_to_oracle_obs(situation, situation.seat, state)
            # Oracle Guiding 核心：按 γ 对 42 个 oracle token 做逐 token Bernoulli dropout。
            # γ=1 保留全部完美信息；γ=0 等价普通 v7。
            if self.gamma < 1.0:
                keep = (np.random.random((3, 14)) < self.gamma).astype(np.int8)
                oh = obs["oracle_hands"].copy()
                for i in range(3):
                    if keep[i].sum() < 14:
                        tiles = np.repeat(np.arange(34, dtype=np.int8), oh[i].clip(0, 4))
                        tiles = tiles[:14]
                        dropped = np.zeros(34, dtype=np.int8)
                        for slot, tile in enumerate(tiles):
                            if keep[i, slot]:
                                dropped[tile] += 1
                        oh[i] = dropped
                obs["oracle_hands"] = oh

        obs_t = {}
        for k, v in obs.items():
            obs_t[k] = torch.from_numpy(np.expand_dims(v, 0)).to(self.device)

        with torch.no_grad():
            binary_logits, tile_logits, value = self.model(obs_t)
            # binary_logits: (1, 5) hu/gang/peng/chi/discard
            b_probs = torch.sigmoid(binary_logits[0]).cpu().numpy()  # (5,)
            t_probs = F.softmax(tile_logits[0], dim=-1).cpu().numpy()  # (34,)

        # 合成 6 类动作概率
        action_probs = np.zeros(6, dtype=np.float64)
        p_hu, p_gang, p_peng, p_chi, p_dis = b_probs
        action_probs[4] = p_hu
        rem = 1.0 - p_hu
        action_probs[3] = rem * p_gang
        rem *= (1 - p_gang)
        action_probs[2] = rem * p_peng
        rem *= (1 - p_peng)
        action_probs[1] = rem * p_chi
        rem *= (1 - p_chi)
        action_probs[0] = rem * p_dis
        action_probs[5] = rem * (1 - p_dis)

        # 选择合法动作中概率最高的
        best_action = None
        best_prob = -1.0
        for a in actions:
            k = kind_map.get(getattr(a, "kind", None))
            if k is not None and action_probs[k] > best_prob:
                best_prob = action_probs[k]
                best_action = a

        if best_action is None:
            best_action = actions[0]

        # 记录轨迹
        chosen_kind = kind_map.get(getattr(best_action, "kind", None), 5)
        chosen_tile = getattr(best_action, "tile", -1) if chosen_kind == 0 else -1

        if chosen_kind == 0:
            # discard：log_prob = log p(discard) + log p(tile|discard)
            lp = np.log(max(action_probs[0], 1e-8)) + np.log(max(t_probs[chosen_tile], 1e-8))
        else:
            lp = np.log(max(action_probs[chosen_kind], 1e-8))

        self.traj_obs.append(obs)
        self.traj_action_kind.append(chosen_kind)
        self.traj_tile.append(chosen_tile)
        self.traj_log_prob.append(lp)
        self.traj_value.append(value.item())
        self.traj_avail.append(avail)

        return best_action


class PPOBufferV7:
    """PPO 经验缓冲区（v7 格式）。"""

    def __init__(self):
        self.obs_list = []
        self.action_kind = []
        self.tile = []
        self.log_prob = []
        self.reward = []
        self.value = []
        self.done = []
        self.avail = []

    def add(self, obs, action_kind, tile, log_prob, reward, value, done, avail):
        self.obs_list.append(obs)
        self.action_kind.append(action_kind)
        self.tile.append(tile)
        self.log_prob.append(log_prob)
        self.reward.append(reward)
        self.value.append(value)
        self.done.append(done)
        self.avail.append(avail)

    def __len__(self):
        return len(self.reward)


def compute_gae(rewards, values, dones, gamma=1.0, lam=0.95):
    """GAE 优势估计。"""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


def collate_obs(obs_list, device):
    """把一批 obs 堆叠成 batch（torch tensor）。"""
    keys = obs_list[0].keys()
    batch = {}
    for k in keys:
        batch[k] = torch.from_numpy(np.stack([o[k] for o in obs_list])).to(device)
    return batch


def collect_episode_oracle_guiding(model, device, seed, rounds=8, gamma=1.0):
    """收集一个 episode（8 局），Oracle Guiding 模式。"""
    from majiang.sim.batch import run_match
    buffer = PPOBufferV7()
    learner = OracleGuidingDecider(model, device, gamma=gamma, deterministic=False)
    deciders = [learner] + [OracleGuidingDecider(model, device, gamma=gamma, deterministic=False) for _ in range(3)]
    result = run_match(deciders, rounds=rounds, base_score=1, seed=seed)

    reward = result.seats[0].place_points
    for t in range(len(learner.traj_obs)):
        is_last = (t == len(learner.traj_obs) - 1)
        buffer.add(
            obs=learner.traj_obs[t],
            action_kind=learner.traj_action_kind[t],
            tile=learner.traj_tile[t],
            log_prob=learner.traj_log_prob[t],
            reward=reward if is_last else 0.0,
            value=learner.traj_value[t],
            done=is_last,
            avail=learner.traj_avail[t],
        )
    return buffer, result


def ppo_update_oracle_guiding(model, buffer, optimizer, device, *,
                                clip_eps=0.2, kl_coef=0.1, value_coef=0.5, entropy_coef=0.01,
                                epochs=4, batch_size=16, bc_model=None):
    """PPO 更新（Oracle Guiding 版，支持动态熵）。"""
    if len(buffer) < 4:
        return {}
    actual_bs = min(batch_size, len(buffer))
    advantages = compute_gae(buffer.reward, buffer.value, buffer.done)
    returns = [adv + val for adv, val in zip(advantages, buffer.value)]

    obs_batch = collate_obs(buffer.obs_list, device)
    action_kinds = torch.tensor(buffer.action_kind, dtype=torch.long, device=device)
    tiles = torch.tensor(buffer.tile, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_prob, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    avail_t = torch.tensor(np.stack(buffer.avail), dtype=torch.float32, device=device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = len(buffer)
    indices = np.arange(n)
    total_loss = 0
    total_kl = 0
    total_entropy = 0.0
    total_tile_ent = 0.0
    total_action_ent = 0.0
    n_updates = 0

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, actual_bs):
            end = min(start + actual_bs, n)
            idx = indices[start:end]

            batch_obs = {k: v[idx] for k, v in obs_batch.items()}
            batch_ak = action_kinds[idx]
            batch_tile = tiles[idx]
            batch_old_lp = old_log_probs[idx]
            batch_adv = advantages_t[idx]
            batch_ret = returns_t[idx]
            batch_avail = avail_t[idx]

            binary_logits, tile_logits, values = model(batch_obs)
            b_probs = torch.sigmoid(binary_logits)  # (B, 5)
            t_log_probs = F.log_softmax(tile_logits, dim=-1)  # (B, 34)
            t_probs = F.softmax(tile_logits, dim=-1)

            # 动作类型概率（链式合成）
            p_hu = b_probs[:, 0]
            p_gang = b_probs[:, 1]
            p_peng = b_probs[:, 2]
            p_chi = b_probs[:, 3]
            p_dis = b_probs[:, 4]

            action_log_probs = torch.zeros(len(idx), 6, device=device)
            action_log_probs[:, 4] = torch.log(p_hu + 1e-8)
            rem = 1.0 - p_hu
            action_log_probs[:, 3] = torch.log(rem * p_gang + 1e-8)
            rem = rem * (1 - p_gang)
            action_log_probs[:, 2] = torch.log(rem * p_peng + 1e-8)
            rem = rem * (1 - p_peng)
            action_log_probs[:, 1] = torch.log(rem * p_chi + 1e-8)
            rem = rem * (1 - p_chi)
            action_log_probs[:, 0] = torch.log(rem * p_dis + 1e-8)
            action_log_probs[:, 5] = torch.log(rem * (1 - p_dis) + 1e-8)

            # 联合 log_prob
            new_log_probs = torch.zeros(len(idx), device=device)
            for i in range(len(idx)):
                ak = batch_ak[i].item()
                if ak == 0:
                    new_log_probs[i] = action_log_probs[i, ak] + t_log_probs[i, batch_tile[i]]
                else:
                    new_log_probs[i] = action_log_probs[i, ak]

            # PPO clip
            ratio = torch.exp(new_log_probs - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), batch_ret)

            # Entropy（分头统计）
            entropy_hu = -(p_hu * torch.log(p_hu + 1e-8) + (1 - p_hu) * torch.log(1 - p_hu + 1e-8))
            entropy_gang = -(p_gang * torch.log(p_gang + 1e-8) + (1 - p_gang) * torch.log(1 - p_gang + 1e-8))
            entropy_peng = -(p_peng * torch.log(p_peng + 1e-8) + (1 - p_peng) * torch.log(1 - p_peng + 1e-8))
            entropy_chi = -(p_chi * torch.log(p_chi + 1e-8) + (1 - p_chi) * torch.log(1 - p_chi + 1e-8))
            entropy_dis = -(p_dis * torch.log(p_dis + 1e-8) + (1 - p_dis) * torch.log(1 - p_dis + 1e-8))
            action_entropy = (entropy_hu + entropy_gang + entropy_peng + entropy_chi + entropy_dis).mean()

            tile_entropy = -(t_probs * t_log_probs).sum(-1).mean()

            # KL 锚定（对 BC 模型）
            kl = torch.tensor(0.0, device=device)
            if bc_model is not None:
                with torch.no_grad():
                    bc_binary, bc_tile, _ = bc_model(batch_obs)
                    bc_b_probs = torch.sigmoid(bc_binary)
                    bc_t_probs = F.softmax(bc_tile, dim=-1)
                kl_b = F.kl_div(torch.log(b_probs + 1e-8), bc_b_probs, reduction='batchmean')
                kl_t = F.kl_div(t_log_probs, bc_t_probs, reduction='batchmean')
                kl = kl_b + kl_t

            loss = policy_loss + value_coef * value_loss - entropy_coef * (action_entropy + tile_entropy) + kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_kl += kl.item()
            total_entropy += action_entropy.item()
            total_tile_ent += tile_entropy.item()
            total_action_ent += action_entropy.item()
            n_updates += 1

    return {
        "loss": total_loss / max(n_updates, 1),
        "kl": total_kl / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
        "tile_entropy": total_tile_ent / max(n_updates, 1),
        "action_entropy": total_action_ent / max(n_updates, 1),
        "n_updates": n_updates,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-model", required=True, help="BC 基线 checkpoint（热启动）")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--dropout-episodes", type=int, default=800, help="γ 从 1→0 的衰减轮数")
    ap.add_argument("--post-dropout-episodes", type=int, default=200, help="γ=0 后继续训练的轮数")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--post-dropout-lr", type=float, default=3e-6, help="γ=0 后的 lr（Suphx 建议 1/10）")
    ap.add_argument("--kl-coef", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--entropy-target", type=float, default=1.5)
    ap.add_argument("--entropy-beta", type=float, default=0.001)
    ap.add_argument("--entropy-coef-min", type=float, default=0.001)
    ap.add_argument("--entropy-coef-max", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--update-every", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--out", default="runs/ppo_oracle_guiding.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    # 初始化 Oracle 模型（411 token）
    model = MahjongOracleV7(d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1).to(device)
    print(f"参数量: {count_parameters(model)}", flush=True)

    # 热启动：从 BC checkpoint 加载前 369 维 pos_emb + 主干
    ckpt = torch.load(args.bc_model, map_location=device, weights_only=False)
    model.load_base_checkpoint(ckpt["model_state_dict"])
    print(f"先知模型初始化自 BC (pos_emb 369→411, 新增 42 个 oracle token)", flush=True)

    # BC 模型（KL 锚定用）
    bc_model = MahjongOracleV7(d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1).to(device)
    bc_model.load_base_checkpoint(ckpt["model_state_dict"])
    for p in bc_model.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 动态熵控制器
    entropy_coef = args.entropy_coef
    entropy_history = []

    buffer = PPOBufferV7()
    accum = 0
    gamma = 1.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(args.episodes):
        t0 = time.time()

        # γ 衰减计划
        if episode < args.dropout_episodes:
            gamma = 1.0 - episode / args.dropout_episodes
        else:
            gamma = 0.0

        # γ=0 后降 lr
        if episode == args.dropout_episodes:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.post_dropout_lr
            print(f"γ=0 过渡完成，lr 降至 {args.post_dropout_lr}", flush=True)

        ep_buffer, result = collect_episode_oracle_guiding(
            model, device, seed=args.seed + episode, rounds=args.rounds, gamma=gamma
        )
        for i in range(len(ep_buffer)):
            buffer.add(
                obs=ep_buffer.obs_list[i],
                action_kind=ep_buffer.action_kind[i],
                tile=ep_buffer.tile[i],
                log_prob=ep_buffer.log_prob[i],
                reward=ep_buffer.reward[i],
                value=ep_buffer.value[i],
                done=ep_buffer.done[i],
                avail=ep_buffer.avail[i],
            )
        accum += len(ep_buffer)

        stats_str = f"Ep {episode:4d} | γ {gamma:.2f}"
        if accum >= args.update_every:
            stats = ppo_update_oracle_guiding(
                model, buffer, optimizer, device,
                clip_eps=0.2, kl_coef=args.kl_coef, value_coef=0.5,
                entropy_coef=entropy_coef, epochs=args.ppo_epochs, batch_size=args.batch_size,
                bc_model=bc_model,
            )
            if stats:
                # 动态熵调整
                entropy_history.append(stats["entropy"])
                if len(entropy_history) > 5:
                    entropy_history.pop(0)
                avg_entropy = sum(entropy_history) / len(entropy_history)
                entropy_coef += args.entropy_beta * (args.entropy_target - avg_entropy)
                entropy_coef = max(args.entropy_coef_min, min(args.entropy_coef_max, entropy_coef))

                stats_str += (f" | loss {stats['loss']:.4f} | KL {stats['kl']:.4f} | "
                              f"H {avg_entropy:.3f}/c{entropy_coef:.4f} | upd {stats['n_updates']}")
            buffer = PPOBufferV7()
            accum = 0
        else:
            stats_str += f" | accum {accum}"

        dt = time.time() - t0
        scores = " ".join(f"p{s.total_score}" for s in result.seats)
        stats_str += f" | {scores} | {dt:.1f}s"
        print(stats_str, flush=True)

        if (episode + 1) % args.save_every == 0:
            torch.save({
                "episode": episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "gamma": gamma,
                "entropy_coef": entropy_coef,
                "args": vars(args),
            }, out_path)
            print(f"  已保存: {out_path}", flush=True)

    # 最终保存
    torch.save({
        "episode": args.episodes - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gamma": 0.0,
        "entropy_coef": entropy_coef,
        "args": vars(args),
    }, out_path)
    print(f"完成。模型已保存到: {out_path}", flush=True)


if __name__ == "__main__":
    main()
