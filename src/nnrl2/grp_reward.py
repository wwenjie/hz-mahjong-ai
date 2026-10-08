"""GRP reward shaping：用训练好的 HistGBM 对 PPO 轨迹做动量信号奖励整形。

设计（与 probe_grp_v2_dynamic.py 特征口径严格一致）：
- 模型输入 = 20 维动态轨迹特征（probe 探针 R²=0.474 验证可学）
- 边界检测 = obs["wall_remaining"] 上升沿（新局发牌墙重置）
- 每局结束 k（k≥hist_len）计算 GRP 预测 pred_k；
  shaped_reward = terminal + λ × Σ_k (pred_k − pred_{k−1})，
  delta_k 记在第 k 局最后一个决策步上
- multiplier 特征探针重要性≈0，推理时填 1.0（obs 不含 multiplier）
- 冷启动：前 hist_len 局不整形（无预测）

worker 池兼容：输入输出都是轨迹 dict 的 reward 列表，不改 decider。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

HIST_LEN = 3
FEAT_DIM = 20


class GRPShaper:
    """加载 joblib HistGBM，对单条 episode 轨迹做 reward shaping。"""

    def __init__(self, model_path: str, lam: float = 0.1, hist_len: int = HIST_LEN):
        import joblib

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.lam = float(lam)
        self.hist_len = int(bundle.get("hist_len", hist_len))

    def shape(self, traj: dict) -> list[float]:
        """返回整形后的 reward 列表（与 traj["reward"] 等长）。

        traj 需含 obs（每步含 scores/dealer/wall_remaining）、reward、done。
        任何异常（观测缺字段等）→ 返回原 reward，不中断训练。
        """
        rewards = list(traj["reward"])
        try:
            return self._shape_impl(traj, rewards)
        except Exception:
            return rewards

    # --- 内部 ---

    def _shape_impl(self, traj: dict, rewards: list[float]) -> list[float]:
        obs_seq = traj["obs"]
        n = len(obs_seq)
        if n == 0:
            return rewards

        walls = np.array([int(o["wall_remaining"]) for o in obs_seq])
        scores = np.array([np.asarray(o["scores"], dtype=np.float64) for o in obs_seq])  # (n, 4) 相对座位
        dealers = np.array([int(o["dealer"]) for o in obs_seq])

        # 局起点：第 0 步 + wall_remaining 上升沿
        starts = [0] + [t for t in range(1, n) if walls[t] > walls[t - 1]]
        # 局终点：起点错位；最后一局终点 = 轨迹末尾
        ends = [s - 1 for s in starts[1:]] + [n - 1]
        num_rounds = len(starts)
        if num_rounds < self.hist_len + 1:
            return rewards

        # 每局快照：结束步的 scores（=该局后累计分）、起点步的 dealer、self 单局得分
        self_delta = []
        won = []
        dealer_abs = []
        for k in range(num_rounds):
            cum_end = scores[ends[k]]
            cum_prev = scores[starts[k] - 1] if starts[k] > 0 else np.zeros(4)
            delta = cum_end - cum_prev
            self_delta.append(float(delta[0]))          # rel 0 = 学习者自己
            won.append(1.0 if delta[0] > 0 else 0.0)    # 近似：自己得分>0 视为胡
            dealer_abs.append(int(dealers[starts[k]]))  # 绝对座位；学习者恒为 seat 0

        cum = np.cumsum(self_delta)  # cum[k] = 前 k+1 局后自己的累计分

        def feats_after(k: int) -> np.ndarray:
            """第 k 局（0-indexed）结束后、预测用的特征（需 k≥hist_len）。"""
            h = self.hist_len
            # 与探针对齐：特征提取于"第 k+1 局开局"视角，hist 用最近 h 局
            cum_self = cum[k]
            # 其他三家的累计分：从 scores 快照拿（相对座位 1..3）
            others_cum = scores[ends[k]][1:4]
            gap = cum_self - max(cum_self, float(others_cum.max(initial=-1e9)))
            cur_round = k + 1  # 即将开始的局号（探针里 k=当前局 index，特征视角一致）
            dealer_now = 1.0 if dealer_abs[min(cur_round, num_rounds - 1)] == 0 else 0.0
            streak = 0
            for j in range(cur_round - 1, -1, -1):
                if dealer_abs[j] == 0:
                    streak += 1
                else:
                    break
            feats = [cum_self, gap, dealer_now, float(cur_round)]
            for i in range(k - h + 1, k + 1):  # 最近 h 局
                feats.extend([self_delta[i], won[i], 1.0 if dealer_abs[i] == 0 else 0.0, 1.0])
            recent = self_delta[k - h + 1: k + 1]
            slope = float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) >= 2 else 0.0
            feats.extend([
                float(np.mean(recent)),
                float(np.mean(won[k - h + 1: k + 1])),
                slope,
                float(streak),
            ])
            return np.asarray(feats, dtype=np.float64)

        preds = {}
        for k in range(self.hist_len - 1, num_rounds):
            f = feats_after(k)
            if f.shape[0] != FEAT_DIM or not np.all(np.isfinite(f)):
                return rewards
            preds[k] = float(self.model.predict(f.reshape(1, -1))[0])

        out = list(rewards)
        ks = sorted(preds)
        for a, b in zip(ks[:-1], ks[1:]):
            out[ends[b]] += self.lam * (preds[b] - preds[a])
        return out


def load_shaper(model_path: Optional[str], lam: float) -> Optional[GRPShaper]:
    """model_path 为 None/空 → 返回 None（不整形）。"""
    if not model_path:
        return None
    return GRPShaper(model_path, lam=lam)


__all__ = ["GRPShaper", "load_shaper", "HIST_LEN", "FEAT_DIM"]
