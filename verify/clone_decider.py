#!/usr/bin/env python3
"""P3 M3：把克隆模型包成 Decider，供 `ab_test --field` 当「真机对手」用。

**红线（A 的约束 1）**：只许离线自对弈，绝不进真机路径（模型用真机对手手牌训练，
进真机 = 信息泄漏）。

决策方式：
- 出牌：对合法打牌集合算克隆模型概率，按概率**采样**（保留对手行为的真实熵，
  而非 argmax 查表）
- 胡/杠/碰/吃：近似口径——胡必接；杠碰必接；吃在向听不降时接
  （对手实测副露 1.09/局 ≈ 几乎全接不恶化局面的机会）
- 特征映射与 `verify/clone_dataset.py` 严格一致（爆头态按摸牌前 13 张判定）

自测：nice -n 15 uv run python verify/clone_decider.py
"""
import pickle
import random

import numpy as np

from majiang.rules import tiles, win
from majiang.rules.action import CHI, DISCARD, GANG, HU, PASS, PENG

N_FEAT = 116
MODEL_PATH = "verify/out/clone_model.pkl"


class CloneDecider:
    name = "clone-live-opp"

    def __init__(self, model_path=MODEL_PATH, seed=0):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self._model = bundle["model"]
        self._rng = random.Random(seed)

    def configure(self, tournament) -> None:  # 兼容运行时段子；离线自对弈用不到
        return None

    def choose(self, situation, actions, *, budget_ms):
        kinds = {a.kind for a in actions}
        if HU in kinds:
            return next(a for a in actions if a.kind == HU)
        discards = [a for a in actions if a.kind == DISCARD]
        if discards:
            return self._choose_discard(situation, discards)
        for want in (GANG, PENG):
            if want in kinds:
                return next(a for a in actions if a.kind == want)
        if CHI in kinds:
            return self._choose_chi(situation, [a for a in actions if a.kind == CHI])
        return next((a for a in actions if a.kind == PASS), None)

    # ---- 出牌：克隆分布采样 ----

    def _features(self, situation):
        counts = list(situation.hand.counts)
        feat = np.zeros(N_FEAT, dtype=np.float32)
        feat[0:34] = counts
        drawn = situation.drawn_tile
        if drawn is not None:
            feat[34 + drawn] = 1.0
        feat[68:72] = [len(m) for m in situation.melds]
        table_discards = np.zeros(34, dtype=np.float32)
        for seat_discards in situation.discards:
            for t in seat_discards:
                table_discards[t] += 1
        feat[72:106] = table_discards
        feat[106] = situation.table.wall_remaining
        feat[107] = (situation.table.draws_made + 3) // 4  # 近似：均匀分配
        feat[108] = situation.table.round_no
        feat[109] = 1.0 if situation.seat == situation.table.dealer_seat else 0.0
        feat[110] = counts[tiles.GOD]
        pre = list(counts)
        if drawn is not None and pre[drawn] > 0:
            pre[drawn] -= 1
        meld_self = len(situation.melds[situation.seat]) if situation.melds else len(
            situation.hand.melds)
        try:
            feat[111] = 1.0 if win.is_baotou(pre, meld_self) else 0.0
        except Exception:
            feat[111] = 0.0
        # [112:116] 前序累计分：Situation 不含，置 0（训练后期望也很小，影响有限）
        return feat

    def _choose_discard(self, situation, discards):
        feat = self._features(situation).reshape(1, -1)
        proba = self._model.predict_proba(feat)[0]
        classes = self._model.classes_
        weights = []
        for a in discards:
            idx = np.flatnonzero(classes == a.tile)
            weights.append(float(proba[idx[0]]) if len(idx) else 1e-9)
        total = sum(weights)
        if total <= 0:
            return self._rng.choice(discards)
        weights = [w / total for w in weights]
        r = self._rng.random()
        acc = 0.0
        for a, w in zip(discards, weights):
            acc += w
            if r <= acc:
                return a
        return discards[-1]

    # ---- 吃：近似口径，向听不降才接 ----

    def _choose_chi(self, situation, chi_actions):
        from majiang.rules.shanten import shanten_any
        counts = list(situation.hand.counts)
        meld_self = len(situation.melds[situation.seat]) if situation.melds else len(
            situation.hand.melds)
        try:
            now = shanten_any(counts, meld_self)
        except Exception:
            now = 99
        best = None
        for a in chi_actions:
            tiles_used = [t for t in (a.tiles or ()) if t != situation.offered_tile]
            after = list(counts)
            for t in tiles_used:
                if after[t] <= 0:
                    break
                after[t] -= 1
            else:
                try:
                    if shanten_any(after, meld_self + 1) <= now:
                        best = best or a
                except Exception:
                    pass
        return best  # None = 放弃吃（向听会变差）


def _selftest():
    """离线自测：四个克隆打一场 8 局，验证可运行 + 分布 sanity。"""
    from majiang.sim.batch import run_match

    deciders = [CloneDecider(seed=i) for i in range(4)]
    result = run_match(deciders, rounds=8, seed=7)
    total = sum(s.total_score for s in result.seats)
    assert result.rounds == 8 and total == 0, (result.rounds, total)
    print(f"自测通过：8 局完成，流局 {result.flows}，零和 OK")
    for s in result.seats:
        print(f"  座{s.seat}: 分 {s.total_score:+d} 胡 {s.wins} 均番 "
              f"{s.fan_total / s.wins if s.wins else 0:.2f}")


if __name__ == "__main__":
    _selftest()
