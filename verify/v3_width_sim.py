#!/usr/bin/env python3
"""预登记：v3（wait-aware tenpai）在自对弈里应产生的听口宽度分布位移。

用途：v3 已上线真机。本脚本先在自对弈里量出「v2 vs v3 的宽度桶分布」，
作为**真机验证的预登记预期**——等 v3 时代真机数据到位，用 wait_width_table 同口径
对拍：若真机位移方向/量级与自对弈不符，机制承诺未兑现（这就是仪器的用法）。

口径：与 verify/wait_width_table.py 相同的有效听口定义（winning_draws 扣公开可见张）。
只对「出牌后仍 0 向听」的决策点统计。
用法：nice -n 19 uv run python verify/v3_width_sim.py [--matches 30]
"""
import argparse
import collections

import numpy as np

from majiang.rules import tiles, win
from majiang.rules.action import DISCARD
from majiang.rules.shanten import shanten_any
from majiang.rules.situation import Situation
from majiang.strategy.versions import build
from majiang.sim.batch import run_match


class WidthRecorder:
    """包一层：正常委托决策，顺手记录听牌出牌点的有效听口。"""

    def __init__(self, inner, sink, memo):
        self._inner = inner
        self._sink = sink
        self._memo = memo
        self.name = inner.name

    def configure(self, tournament):
        if hasattr(self._inner, "configure"):
            self._inner.configure(tournament)

    def choose(self, situation, actions, *, budget_ms):
        action = self._inner.choose(situation, actions, budget_ms=budget_ms)
        if action is not None and action.kind == DISCARD:
            counts = list(situation.hand.counts)
            counts[action.tile] -= 1
            meld_n = len(situation.melds[situation.seat]) if situation.melds else 0
            try:
                if shanten_any(counts, meld_n, memo=self._memo) == 0:
                    waits = win.winning_draws(counts, meld_n)
                    if waits:
                        visible = [0] * tiles.TILE_KINDS
                        for i, c in enumerate(counts):
                            visible[i] += c
                        for per_seat in situation.discards:
                            for t in per_seat:
                                visible[t] += 1
                        for per_seat in situation.melds:
                            for m in per_seat:
                                for t in getattr(m, "tiles", ()):  # Meld 内容
                                    visible[t] += 1
                        eff = sum(max(0, 4 - visible[w]) for w in waits)
                        self._sink.append(eff)
            except Exception:
                pass
        return action


def bucket(w):
    for hi, name in ((4, "1-4"), (8, "5-8"), (12, "9-12"), (20, "13-20")):
        if w <= hi:
            return name
    return "21+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = ap.parse_args()

    for vid in ("v2", "v3"):
        sink = []
        memo: dict = {}
        for seed in args.seeds:
            deciders = [WidthRecorder(build(vid, "qualifier"), sink, memo)
                        for _ in range(4)]
            run_match(deciders, rounds=8, seed=seed)
        arr = np.array(sink)
        dist = collections.Counter(bucket(w) for w in arr)
        n = len(arr)
        print(f"[{vid}] 听牌出牌点 n={n} 均宽={arr.mean():.2f} "
              f"分布: " + " ".join(f"{b}={dist[b]/n:.1%}" for b in ("1-4", "5-8", "9-12", "13-20", "21+")))


if __name__ == "__main__":
    main()
