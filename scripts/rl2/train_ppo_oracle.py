#!/usr/bin/env python
"""Oracle（先知）PPO 训练：全知视角模型，Suphx Oracle Guiding 简化版。

合规红线：先知模型只进训练管线，交付的比赛模型绝不调用本脚本产物。
蒸馏路径（后续）：先知在自对弈状态分布上提供动作分布 → 正常 v7 模型 KL 蒸馏模仿。

复用 train_ppo_v7 的骨架，差异：
- 观测：situation_to_oracle_obs（公开 obs + 三家真实手牌）
- 模型：MahjongOracleV7（pos_emb 411，oracle token 42 个）
- Decider：实现 observe_state(state) 接收 RoundState（仿真器钩子）
- BC 锚定：kl_coef=0（先知的公开 obs 子分布 ≈ BC 分布，锚定意义不大且拖慢收敛；
  先知的价值在于利用全知信息快速收敛到高胜率，再用蒸馏回正常模型）
- 对手：自对弈（4 个先知互打，信息优势下收敛快，无需 v5 教师）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # 让 from train_ppo_v7 import 可用

from nnrl2.model_oracle import MahjongOracleV7
from nnrl2.oracle_obs import situation_to_oracle_obs
from nnrl2.model_v7 import MahjongTransformerV7

# 复用 PPO 核心组件
from train_ppo_v7 import (
    ACTION_NAMES,
    PPOBufferV7,
    compute_gae,
    collate_obs,
)


class OraclePolicyDecider:
    """先知策略决策器：observe_state 接收 RoundState，choose 用全知观测。"""

    def __init__(self, model, device, deterministic=False):
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.name = "oracle_policy_v7"
        self._state = None  # RoundState（observe_state 注入）
        # 轨迹记录
        self.traj_obs = []
        self.traj_action_kind = []
        self.traj_tile = []
        self.traj_log_prob = []
        self.traj_value = []
        self.traj_avail = []

    def observe_state(self, state):
        self._state = state

    def configure(self, tournament):
        pass

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

        # 先知观测
        state = self._state
        if state is None:
            # 降级：无全知状态时用公开 obs（不应发生）
            from nnrl2.obs import situation_to_obs
            obs = situation_to_obs(situation, situation.seat)
            obs["oracle_hands"] = np.zeros((3, 34), dtype=np.int8)
        else:
            obs = situation_to_oracle_obs(situation, situation.seat, state)

        obs_t = {k: torch.from_numpy(np.expand_dims(v, 0)).to(self.device) for k, v in obs.items()}

        with torch.no_grad():
            binary_logits, tile_logits, value = self.model(obs_t)
            import torch.nn.functional as F
            sig = torch.sigmoid(binary_logits).clamp(0.01, 0.99).squeeze(0).cpu().numpy()
            tile_probs = F.softmax(tile_logits, dim=-1).squeeze(0).cpu().numpy()

        # 合成 6 类动作概率（链式优先级 hu > gang > peng > chi > discard > pass）
        p_hu, p_gang, p_peng, p_chi, p_dis = sig
        rem = 1.0 - p_hu
        action_probs = np.zeros(6, dtype=np.float64)
        action_probs[4] = p_hu
        action_probs[3] = rem * p_gang; rem *= (1 - p_gang)
        action_probs[2] = rem * p_peng; rem *= (1 - p_peng)
        action_probs[1] = rem * p_chi; rem *= (1 - p_chi)
        action_probs[0] = rem * p_dis; rem *= (1 - p_dis)
        action_probs[5] = rem

        masked_probs = action_probs * avail
        total = masked_probs.sum()
        if total < 1e-12:
            masked_probs = avail / avail.sum()
        else:
            masked_probs /= total

        if self.deterministic:
            chosen_kind = int(np.argmax(masked_probs))
        else:
            chosen_kind = int(np.random.choice(6, p=masked_probs))

        chosen_action = None
        chosen_tile = -1
        kind_name = ACTION_NAMES[chosen_kind]
        if kind_name == "discard":
            legal = [(a, a.tile, tile_probs[a.tile]) for a in actions
                     if a.kind == "discard" and a.tile is not None and 0 <= a.tile < 34]
            if not legal:
                return actions[0]
            tp = np.array([p for _, _, p in legal], dtype=np.float64)
            tp_sum = tp.sum()
            if tp_sum < 1e-8:
                chosen_action, chosen_tile, _ = legal[0]
            else:
                tp /= tp_sum
                idx = int(np.argmax(tp)) if self.deterministic else int(np.random.choice(len(legal), p=tp))
                chosen_action, chosen_tile, _ = legal[idx]
            log_prob = np.log(max(action_probs[0], 1e-8)) + np.log(max(tile_probs[chosen_tile], 1e-8))
        else:
            for a in actions:
                if a.kind == kind_name:
                    chosen_action = a
                    break
            if chosen_action is None:
                return actions[0]
            log_prob = np.log(max(action_probs[chosen_kind], 1e-8))

        self.traj_obs.append(obs)
        self.traj_action_kind.append(chosen_kind)
        self.traj_tile.append(chosen_tile)
        self.traj_log_prob.append(log_prob)
        self.traj_value.append(value.item())
        self.traj_avail.append(avail)

        return chosen_action


def collect_episode_oracle(model, device, seed, rounds=8):
    """收集一个 episode（4 个先知自对弈）。"""
    from majiang.sim.batch import run_match
    buffer = PPOBufferV7()
    learner = OraclePolicyDecider(model, device, deterministic=False)
    deciders = [learner] + [OraclePolicyDecider(model, device, deterministic=False) for _ in range(3)]
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


def ppo_update_oracle(model, buffer, optimizer, device, *,
                      clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                      epochs=4, batch_size=16):
    """PPO 更新（先知版：无 KL 锚定，因为先知策略天然偏离 BC）。"""
    import torch.nn.functional as F

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

            # 动作类型 log_prob（sigmoid 头链式）
            sig = torch.sigmoid(binary_logits)
            log_sig = torch.log(sig.clamp(min=1e-8))
            log_neg = torch.log((1 - sig).clamp(min=1e-8))
            lp_hu = log_sig[:, 0]
            lp_gang = log_neg[:, 0] + log_sig[:, 1]
            lp_peng = log_neg[:, 0] + log_neg[:, 1] + log_sig[:, 2]
            lp_chi = log_neg[:, 0] + log_neg[:, 1] + log_neg[:, 2] + log_sig[:, 3]
            lp_dis = log_neg[:, 0] + log_neg[:, 1] + log_neg[:, 2] + log_neg[:, 3] + log_sig[:, 4]
            lp_pass = log_neg.sum(dim=1)
            action_log_probs = torch.stack([lp_dis, lp_chi, lp_peng, lp_gang, lp_hu, lp_pass], dim=1)

            # tile log_prob
            tile_log_softmax = F.log_softmax(tile_logits, dim=-1)

            # 联合 log_prob
            new_log_probs = torch.zeros(len(idx), device=device)
            for i in range(len(idx)):
                ak = batch_ak[i].item()
                lp_action = action_log_probs[i, ak]
                if ak == 0:
                    t = batch_tile[i].item()
                    lp_action = lp_action + tile_log_softmax[i, t]
                new_log_probs[i] = lp_action

            # entropy
            tile_entropy = -(torch.exp(tile_log_softmax) * tile_log_softmax).sum(-1).mean()
            action_probs = torch.exp(action_log_probs)
            action_probs = action_probs * batch_avail
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            action_entropy = -(action_probs * torch.log(action_probs.clamp(min=1e-8))).sum(-1).mean()
            entropy = tile_entropy + 0.5 * action_entropy

            # PPO clip
            ratio = torch.exp(new_log_probs - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = F.mse_loss(values.squeeze(-1), batch_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            n_updates += 1

    buffer.clear()
    return {"loss": total_loss / max(n_updates, 1), "n_updates": n_updates, "buffer_size": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Oracle PPO：全知先知模型训练")
    ap.add_argument("--bc-model", default="runs/bc_v7_base.pt")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default="runs/oracle_v7.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--update-every", type=int, default=256)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    # 加载 BC checkpoint（用于热启动先知模型）
    ckpt = torch.load(args.bc_model, map_location=device, weights_only=False)
    bc_args = ckpt.get("args", {})
    model_kwargs = dict(
        d_model=bc_args.get("d_model", 128),
        nhead=bc_args.get("nhead", 8),
        num_layers=bc_args.get("num_layers", 4),
        dim_feedforward=bc_args.get("dim_feedforward", 512),
    )

    # 先知模型：从 BC 热启动
    model = MahjongOracleV7(**model_kwargs).to(device)
    model.load_base_checkpoint(ckpt["model_state_dict"])
    print(f"先知模型初始化自 BC (pos_emb 369→411, 新增 42 个 oracle token)", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable:,}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accum_buffer = PPOBufferV7()
    total_updates = 0

    for episode in range(args.episodes):
        t0 = time.perf_counter()
        buffer, result = collect_episode_oracle(model, device, seed=args.seed + episode, rounds=args.rounds)
        for i in range(len(buffer)):
            accum_buffer.add(
                obs=buffer.obs_list[i], action_kind=buffer.action_kind[i], tile=buffer.tile[i],
                log_prob=buffer.log_prob[i], reward=buffer.reward[i], value=buffer.value[i],
                done=buffer.done[i], avail=buffer.avail[i],
            )

        stats = {}
        if len(accum_buffer) >= args.update_every:
            stats = ppo_update_oracle(
                model, accum_buffer, optimizer, device,
                clip_eps=args.clip_eps, entropy_coef=args.entropy_coef,
                batch_size=args.batch_size, epochs=args.ppo_epochs,
            )
            total_updates += 1
        dt = time.perf_counter() - t0

        if episode % 10 == 0:
            seat_info = " ".join(f"p{s.place_points}" for s in result.seats)
            print(
                f"Ep {episode:4d} | loss {stats.get('loss', 0):.4f} | "
                f"upd {stats.get('n_updates', 0)} | buf {stats.get('buffer_size', 0)} | "
                f"accum {len(accum_buffer)} | {seat_info} | {dt:.1f}s",
                flush=True,
            )

        if (episode + 1) % args.save_every == 0:
            torch.save({
                "episode": episode,
                "total_updates": total_updates,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
            }, out_path)
            print(f"  已保存: {out_path}", flush=True)

    torch.save({
        "episode": args.episodes,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, out_path)
    print(f"\n完成。先知模型已保存到: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
