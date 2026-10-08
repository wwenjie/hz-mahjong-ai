"""v7 策略决策器 + episode 收集。

共享实现：`train_ppo_v7.py` 的串行路径与 `parallel_rollout.py` 的 worker 进程
必须使用同一份决策/轨迹记录逻辑，避免两处拷贝漂移。

注意：与 `train_ppo_v7.py` 内联的 PolicyV7Decider 保持语义一致（同一来源迁出）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from nnrl2.obs import situation_to_obs

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


def collect_episode_v7(model, device, seed, rounds=8, opponent_factories=None, record_seat=0):
    """收集一个 episode。

    返回 ``(traj, result)``：traj 是可直接喂 PPOBuffer 的列表字典，
    稀疏奖励（名次分）记在最后一步；result 是 run_match 的原始结果（日志用）。
    """
    from majiang.sim.batch import run_match

    np.random.seed(seed)
    learner = PolicyV7Decider(model, device, deterministic=False)
    if opponent_factories is not None:
        deciders = [learner] + [f() for f in opponent_factories]
    else:
        deciders = [learner] + [PolicyV7Decider(model, device, deterministic=False) for _ in range(3)]
    result = run_match(deciders, rounds=rounds, base_score=1, seed=seed)

    reward = result.seats[record_seat].place_points
    n = len(learner.traj_obs)
    traj = {
        "obs": learner.traj_obs,
        "action_kind": learner.traj_action_kind,
        "tile": learner.traj_tile,
        "log_prob": learner.traj_log_prob,
        "reward": [0.0] * n,
        "value": learner.traj_value,
        "done": [False] * n,
        "avail": learner.traj_avail,
    }
    if n > 0:
        traj["reward"][-1] = reward
        traj["done"][-1] = True
    return traj, result
