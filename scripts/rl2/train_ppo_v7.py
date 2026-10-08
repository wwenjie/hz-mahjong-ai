#!/usr/bin/env python
"""PPO + KL 锚定训练（v7 Suphx 式多头模型专用）。

架构适配：
- 策略 = MahjongTransformerV7（5 个 sigmoid 二分类头 + tile 34 类 + value）
- 动作空间 = (动作类型 6 类, tile 0-33)，log_prob = log p(动作类型) + log p(tile|discard)
- Rollout：4 座位自对弈，规则兜底能胡必胡（不记入轨迹）
- PPO 更新：tile 头 Categorical clip + 动作头 sigmoid clip + value MSE + entropy
- KL 锚定：对 BC 模型的 tile 分布做 KL 约束

用法::

    PYTHONPATH=src python scripts/train_ppo_v7.py --bc-model runs/bc_v7_base.pt --episodes 500
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path


def _raise_nofile_soft_limit(target: int = 65536) -> None:
    """多进程 worker 的 fd 需求随 workers 数线性增长；
    默认软限 1024 在 workers>=8 时会触发 OSError(24) Too many open files。
    进程内提升软限（不超过硬限），比依赖 shell ulimit 可靠。"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = min(target, hard)
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except (ValueError, OSError):
        pass  # 容器不给提就保持现状，由 workers 数兜底

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# worker 返回的 traj/result 可能含 majiang 引擎对象；
# 主进程反序列化 pool 结果时需要 majiang 可 import，否则
# _handle_results 线程抛 ModuleNotFoundError 并静默卡死。
for _p in [
    "/home/wuwenjie01/majiang_ai/src",
    "/root/autodl-tmp/ab/majiang_ai/src",
    "/root/autodl-tmp/majiang_ai/src",
    "/root/majiang_ai/src",
]:
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

from nnrl2.model_v7 import MahjongTransformerV7
from nnrl2.obs import situation_to_obs
from nnrl2.policy_v7 import collect_episode_v7

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]
HEAD_TO_ACTION = [4, 3, 2, 1, 0]  # hu/gang/peng/chi/discard


class PolicyV7Decider:
    """v7 策略模型决策器（自对弈收集经验用）。"""

    def __init__(self, model, device, deterministic=False):
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.name = "policy_v7"
        self.last_reason = ""
        self.last_detail = {}
        # 轨迹记录
        self.traj_obs = []
        self.traj_action_kind = []  # 6 类动作 id
        self.traj_tile = []         # 出哪张牌（非 discard 记 -1）
        self.traj_log_prob = []     # 联合 log_prob
        self.traj_value = []
        self.traj_avail = []        # (6,) 可行性 mask

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

        obs = situation_to_obs(situation, situation.seat)
        obs_t = {}
        for k, v in obs.items():
            obs_t[k] = torch.from_numpy(np.expand_dims(v, 0)).to(self.device)

        with torch.no_grad():
            binary_logits, tile_logits, value = self.model(obs_t)
            # binary_logits: (1, 5) hu/gang/peng/chi/discard
            # clamp 防坍缩：sigmoid 饱和到 0/1 会导致全拒绝（masked_probs→0）
            sig = torch.sigmoid(binary_logits).clamp(0.01, 0.99).squeeze(0).cpu().numpy()
            tile_probs = F.softmax(tile_logits, dim=-1).squeeze(0).cpu().numpy()
            tile_log_probs = F.log_softmax(tile_logits, dim=-1).squeeze(0).cpu().numpy()

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

        # 只保留可行动作
        masked_probs = action_probs * avail
        total = masked_probs.sum()
        if total < 1e-12:
            # 极端坍缩兜底：按可行动作均匀分布，仍记录轨迹（否则 buffer 永远为空）
            masked_probs = avail / avail.sum()
        else:
            masked_probs /= total

        # 选动作类型
        if self.deterministic:
            chosen_kind = int(np.argmax(masked_probs))
        else:
            chosen_kind = int(np.random.choice(6, p=masked_probs))

        # 找对应的 Action 对象
        chosen_action = None
        chosen_tile = -1
        kind_name = ACTION_NAMES[chosen_kind]
        if kind_name == "discard":
            # 从可打出牌中按 tile_probs 采样
            legal = []
            for a in actions:
                if a.kind == "discard" and a.tile is not None and 0 <= a.tile < 34:
                    legal.append((a, a.tile, tile_probs[a.tile]))
            if not legal:
                return actions[0]
            tp = np.array([p for _, _, p in legal], dtype=np.float64)
            tp_sum = tp.sum()
            if tp_sum < 1e-8:
                chosen_action, chosen_tile, _ = legal[0]
            else:
                tp /= tp_sum
                if self.deterministic:
                    idx = int(np.argmax(tp))
                else:
                    idx = int(np.random.choice(len(legal), p=tp))
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

        # 记录轨迹
        self.traj_obs.append(obs)
        self.traj_action_kind.append(chosen_kind)
        self.traj_tile.append(chosen_tile)
        self.traj_log_prob.append(log_prob)
        self.traj_value.append(value.item())
        self.traj_avail.append(avail)

        return chosen_action


class PPOBufferV7:
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

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.reward)


def compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


def collate_obs(obs_list, device):
    keys = obs_list[0].keys()
    result = {}
    for key in keys:
        values = [obs[key] for obs in obs_list]
        if isinstance(values[0], np.ndarray):
            result[key] = torch.from_numpy(np.stack(values)).to(device)
        else:
            result[key] = torch.tensor(values, device=device)
    return result


def collect_episode(model, device, seed, rounds=8, opponent_factories=None):
    """收集一个 episode（串行路径，单进程）。

    与并行路径（nnrl2.parallel_rollout）语义一致：返回 (PPOBufferV7, result)。
    """
    from majiang.sim.batch import run_match
    buffer = PPOBufferV7()
    learner = PolicyV7Decider(model, device, deterministic=False)
    if opponent_factories is not None:
        deciders = [learner] + [f() for f in opponent_factories]
    else:
        deciders = [learner] + [PolicyV7Decider(model, device, deterministic=False) for _ in range(3)]
    result = run_match(deciders, rounds=rounds, base_score=1, seed=seed)

    # 只收 learner（seat 0）的轨迹
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


def collect_episode_parallel(pool, seed, rounds=8):
    """用 ParallelRolloutPool 并行收集一个 episode（多进程）。

    返回与串行版相同格式：(PPOBufferV7, result)。
    """
    traj, result = pool.collect(seed_start=seed, num_episodes=1, rounds=rounds)[0]
    buffer = PPOBufferV7()
    n = len(traj["obs"])
    for t in range(n):
        buffer.add(
            obs=traj["obs"][t],
            action_kind=traj["action_kind"][t],
            tile=traj["tile"][t],
            log_prob=traj["log_prob"][t],
            reward=traj["reward"][t],
            value=traj["value"][t],
            done=traj["done"][t],
            avail=traj["avail"][t],
        )
    return buffer, result


def ppo_update(model, bc_model, buffer, optimizer, device, *,
               clip_eps=0.2, kl_coef=0.1, value_coef=0.5, entropy_coef=0.01,
               epochs=4, batch_size=16):
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

            # --- 动作类型 log_prob（sigmoid 头链式） ---
            sig = torch.sigmoid(binary_logits)  # (B, 5) hu/gang/peng/chi/discard
            log_sig = torch.log(sig.clamp(min=1e-8))
            log_neg = torch.log((1 - sig).clamp(min=1e-8))
            # 展开成 6 类 log_prob
            lp_hu = log_sig[:, 0]
            lp_gang = log_neg[:, 0] + log_sig[:, 1]
            lp_peng = log_neg[:, 0] + log_neg[:, 1] + log_sig[:, 2]
            lp_chi = log_neg[:, 0] + log_neg[:, 1] + log_neg[:, 2] + log_sig[:, 3]
            lp_dis = log_neg[:, 0] + log_neg[:, 1] + log_neg[:, 2] + log_neg[:, 3] + log_sig[:, 4]
            lp_pass = log_neg.sum(dim=1)
            action_log_probs = torch.stack([lp_dis, lp_chi, lp_peng, lp_gang, lp_hu, lp_pass], dim=1)  # (B, 6)

            # --- tile log_prob ---
            tile_log_softmax = F.log_softmax(tile_logits, dim=-1)  # (B, 34)

            # 联合 log_prob
            new_log_probs = torch.zeros(len(idx), device=device)
            for i in range(len(idx)):
                ak = batch_ak[i].item()
                lp_action = action_log_probs[i, ak]
                if ak == 0:  # discard
                    t = batch_tile[i].item()
                    lp_action = lp_action + tile_log_softmax[i, t]
                new_log_probs[i] = lp_action

            # entropy（tile 分布 + 动作分布）
            tile_entropy = -(torch.exp(tile_log_softmax) * tile_log_softmax).sum(-1).mean()
            action_probs = torch.exp(action_log_probs)
            action_probs = action_probs * batch_avail
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            action_entropy = -(action_probs * torch.log(action_probs.clamp(min=1e-8))).sum(-1).mean()
            entropy = tile_entropy + 0.5 * action_entropy
            total_entropy += entropy.item()
            total_tile_ent += tile_entropy.item()
            total_action_ent += action_entropy.item()

            # KL 锚定（tile 分布 + 6类动作分布，对 BC 模型）
            with torch.no_grad():
                bc_b_logits, bc_tile_logits, _ = bc_model(batch_obs)
                bc_sig = torch.sigmoid(bc_b_logits)
                bc_lp = torch.log(bc_sig.clamp(min=1e-8))
                bc_ln = torch.log((1 - bc_sig).clamp(min=1e-8))
                bc_action_lp = torch.stack([
                    bc_ln[:, 0] + bc_ln[:, 1] + bc_ln[:, 2] + bc_ln[:, 3] + bc_lp[:, 4],
                    bc_ln[:, 0] + bc_ln[:, 1] + bc_ln[:, 2] + bc_lp[:, 3],
                    bc_ln[:, 0] + bc_ln[:, 1] + bc_lp[:, 2],
                    bc_ln[:, 0] + bc_lp[:, 1],
                    bc_lp[:, 0],
                    bc_ln.sum(dim=1),
                ], dim=1)  # (B, 6) 同 action_log_probs 的排列 dis/chi/peng/gang/hu/pass
            kl_tile = F.kl_div(
                tile_log_softmax,
                F.softmax(bc_tile_logits, dim=-1),
                reduction="batchmean",
            )
            # 动作分布 KL：两边都在 avail mask 上归一化后，计算 KL(π_new || π_BC)
            bc_action_probs = torch.exp(bc_action_lp) * batch_avail
            bc_action_probs = bc_action_probs / bc_action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            log_bc = torch.log(bc_action_probs.clamp(min=1e-8))
            log_new = torch.log(action_probs.clamp(min=1e-8))
            kl_action = (action_probs * (log_new - log_bc)).sum(-1).mean().clamp(min=0)
            kl_div = kl_tile + kl_action

            # PPO clip
            ratio = torch.exp(new_log_probs - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = F.mse_loss(values.squeeze(-1), batch_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + kl_coef * kl_div

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_kl += kl_div.item()
            n_updates += 1

        # KL 早停：策略漂移过大时终止本轮 PPO epochs
        avg_kl = total_kl / max(n_updates, 1)
        if avg_kl > 0.1:
            break

    buffer.clear()
    n_upd = max(n_updates, 1)
    return {"loss": total_loss / n_upd, "kl": total_kl / n_upd,
            "n_updates": n_updates, "buffer_size": n,
            "entropy": total_entropy / n_upd,
            "tile_entropy": total_tile_ent / n_upd,
            "action_entropy": total_action_ent / n_upd}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO v7：Suphx 式多头模型")
    ap.add_argument("--bc-model", default="runs/bc_v7_base.pt")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--entropy-target", type=float, default=1.5, help="目标熵（softmax 6 类最大 ln6≈1.79），<=0 关闭动态调整")
    ap.add_argument("--entropy-beta", type=float, default=0.001, help="动态熵调整步长：alpha += beta * (H_target - H_empirical)")
    ap.add_argument("--entropy-coef-min", type=float, default=0.001)
    ap.add_argument("--entropy-coef-max", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default="runs/ppo_v7.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--update-every", type=int, default=256, help="攒够多少条经验才 PPO 更新")
    ap.add_argument("--vs-teacher", action="store_true", help="对手用 3 个 v5 教师（否则自对弈）")
    ap.add_argument("--freeze-trunk", action="store_true", default=True, help="冻结主干+动作头（默认，Meowjong 方案）")
    ap.add_argument("--no-freeze-trunk", dest="freeze_trunk", action="store_false", help="解冻主干（只冻 4 个动作头）")
    ap.add_argument("--workers", type=int, default=0, help="并行 rollout worker 数（0=串行，>0=多进程）")
    ap.add_argument("--grp-model", type=str, default="", help="GRP HistGBM joblib 路径（reward shaping，空=不启用）")
    ap.add_argument("--grp-lambda", type=float, default=0.1, help="GRP 整形强度 λ：reward = 终局分 + λ×GRP差分")
    args = ap.parse_args(argv)
    _raise_nofile_soft_limit()  # workers>=8 时防 OSError(24) Too many open files

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    # 加载 BC 模型
    ckpt = torch.load(args.bc_model, map_location=device, weights_only=False)
    bc_args = ckpt.get("args", {})
    model_kwargs = dict(
        d_model=bc_args.get("d_model", 128),
        nhead=bc_args.get("nhead", 8),
        num_layers=bc_args.get("num_layers", 4),
        dim_feedforward=bc_args.get("dim_feedforward", 512),
    )

    bc_model = MahjongTransformerV7(**model_kwargs).to(device)
    bc_model.load_state_dict(ckpt["model_state_dict"])
    bc_model.eval()
    for p in bc_model.parameters():
        p.requires_grad = False
    print(f"BC 锚定模型: {args.bc_model} (loss={ckpt.get('train_loss', 0):.4f})", flush=True)

    model = MahjongTransformerV7(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"策略模型初始化自 BC", flush=True)

    # 冻结动作头（v5 动作决策已可靠，不训练）
    frozen_heads = [model.hu_head, model.gang_head, model.peng_head, model.chi_head]
    for head in frozen_heads:
        for p in head.parameters():
            p.requires_grad = False
    if args.freeze_trunk:
        # Meowjong 方案：主干也冻结，只训 tile_head + value_head + discard_head
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.transformer.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trunk_status = "冻结主干+动作头" if args.freeze_trunk else "解冻主干（只冻动作头）"
    print(f"{trunk_status}，可训练参数: {trainable:,}", flush=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 对手设置
    opponent_factories = None
    if args.vs_teacher:
        from majiang.strategy.versions import build
        from majiang.strategy.policy import Mode
        opponent_factories = [lambda: build("v5", Mode.QUALIFIER) for _ in range(3)]
        print("对手: 3 个 v5 教师（QUALIFIER）", flush=True)
    else:
        print("对手: 自对弈", flush=True)

    # 多进程 rollout 池（可选）：worker 在 CPU 上推理，主进程只负责 PPO 更新
    pool = None
    eps_per_batch = 1
    if args.workers > 0:
        from nnrl2.parallel_rollout import ParallelRolloutPool
        pool = ParallelRolloutPool(
            MahjongTransformerV7, model_kwargs,
            num_workers=args.workers,
            opponent_kind="v5" if args.vs_teacher else None,
            state_dict=model.state_dict(),
        )
        # 每批收集 episodes 数：打满 worker，同时不超过 update_every 的量级
        eps_per_batch = max(args.workers, min(args.update_every // 30, 64))
        print(f"并行 rollout: {args.workers} workers, 每批 {eps_per_batch} episodes", flush=True)

    # GRP reward shaping（可选）
    grp_shaper = None
    if args.grp_model:
        from nnrl2.grp_reward import load_shaper
        grp_shaper = load_shaper(args.grp_model, args.grp_lambda)
        print(f"GRP shaping: {args.grp_model} (λ={args.grp_lambda})", flush=True)

    # 攒批 buffer：跨 episode 累积，凑够 update_every 条才做一次 PPO 更新
    accum_buffer = PPOBufferV7()
    total_updates = 0

    # 动态熵正则（Suphx）：控制器状态
    entropy_coef = args.entropy_coef
    entropy_window: list[float] = []  # 每次 PPO 更新的实测熵

    for episode in range(0, args.episodes, eps_per_batch):
        t0 = time.perf_counter()
        if pool is not None:
            # 并行路径：同步权重 → 批量收集
            pool.sync_model(model.state_dict())
            batch_results = pool.collect(seed_start=args.seed + episode, num_episodes=eps_per_batch, rounds=args.rounds)
            result = None
            for traj, result in batch_results:
                rewards = grp_shaper.shape(traj) if grp_shaper is not None else traj["reward"]
                n = len(traj["obs"])
                for t in range(n):
                    accum_buffer.add(
                        obs=traj["obs"][t], action_kind=traj["action_kind"][t], tile=traj["tile"][t],
                        log_prob=traj["log_prob"][t], reward=rewards[t], value=traj["value"][t],
                        done=traj["done"][t], avail=traj["avail"][t],
                    )
        else:
            buffer, result = collect_episode(model, device, seed=args.seed + episode, rounds=args.rounds, opponent_factories=opponent_factories)
            if grp_shaper is not None:
                traj = {"obs": buffer.obs_list, "reward": buffer.reward, "done": buffer.done}
                buffer.reward = grp_shaper.shape(traj)
            for i in range(len(buffer)):
                accum_buffer.add(
                    obs=buffer.obs_list[i], action_kind=buffer.action_kind[i], tile=buffer.tile[i],
                    log_prob=buffer.log_prob[i], reward=buffer.reward[i], value=buffer.value[i],
                    done=buffer.done[i], avail=buffer.avail[i],
                )

        stats = {}
        if len(accum_buffer) >= args.update_every:
            stats = ppo_update(
                model, bc_model, accum_buffer, optimizer, device,
                clip_eps=args.clip_eps, kl_coef=args.kl_coef,
                entropy_coef=entropy_coef,
                batch_size=args.batch_size, epochs=args.ppo_epochs,
            )
            total_updates += 1
            # --- 动态熵控制器（Suphx）：alpha += beta * (H_target - H_empirical) ---
            if args.entropy_target > 0 and "entropy" in stats:
                entropy_window.append(stats["entropy"])
                h_emp = float(np.mean(entropy_window[-5:]))
                entropy_coef = float(np.clip(
                    entropy_coef + args.entropy_beta * (args.entropy_target - h_emp),
                    args.entropy_coef_min, args.entropy_coef_max,
                ))
                stats["h_emp"] = h_emp
                stats["ent_coef"] = entropy_coef
        dt = time.perf_counter() - t0

        if episode % 10 == 0:
            seat_info = " ".join(f"p{s.place_points}" for s in result.seats)
            print(
                f"Ep {episode:4d} | loss {stats.get('loss', 0):.4f} | KL {stats.get('kl', 0):.4f} | "
                f"H {stats.get('h_emp', 0):.3f}/c{stats.get('ent_coef', entropy_coef):.4f} | "
                f"upd {stats.get('n_updates', 0)} | buf {stats.get('buffer_size', 0)} | "
                f"accum {len(accum_buffer)} | {seat_info} | {dt:.1f}s",
                flush=True,
            )

        # 按 update 步数周期保存（16 workers 下 episode 按批跳，ep%save_every 不可靠）
        if total_updates > 0 and total_updates % args.save_every == 0:
            torch.save({
                "episode": episode,
                "total_updates": total_updates,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
            }, out_path)
            print(f"  已保存: {out_path} (ep={episode}, upd={total_updates})", flush=True)

    torch.save({
        "episode": args.episodes,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, out_path)
    if pool is not None:
        pool.close()
    print(f"\n完成。模型已保存到: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
