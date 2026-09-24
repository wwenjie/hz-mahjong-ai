"""用于前瞻搜索的极速策略与滚出（tasks.md 5.15）。

搜索的成本是 ``样本数 × 候选数 × 单次滚出``，而单次滚出 = 剩余回合数 × 每回合决策成本。
精确 ``shanten`` 约 0.25–1.4 ms，放不进滚出内层；因此这里只用微秒级的
``quick_shanten``（骨架近似，实测与精确值平均误差 0.56）排序出牌，胡牌仍用精确判定
（因为胡牌是收益事件，不能近似）。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules.action import DISCARD, GANG, HU, PASS, PENG, Action
from majiang.rules.situation import PHASE_DRAW, Situation
from majiang.rules.tiles import GOD

from . import risk


def quick_shanten_after(counts: Sequence[int], tile: int, meld_count: int) -> int:
    work = list(counts)
    work[tile] -= 1
    return shanten_module.quick_shanten(work, meld_count)


class FastDecider:
    """滚出专用：能胡就胡，否则用骨架向听秒选一张牌，响应窗口一律过。"""

    name = "fast-rollout"

    def choose(
        self,
        situation: Situation,
        actions: Sequence[Action],
        *,
        budget_ms: int = 0,
    ) -> Action | None:
        for action in actions:
            if action.kind == HU:
                return action
        best: Action | None = None
        best_value = 1 << 30
        counts = situation.hand.counts
        meld_count = situation.hand.meld_count
        for action in actions:
            if action.kind != DISCARD or action.tile is None:
                continue
            value = quick_shanten_after(counts, action.tile, meld_count)
            if value < best_value:
                best_value = value
                best = action
        if best is not None:
            return best
        return next((action for action in actions if action.kind == PASS), actions[0])


@dataclass
class RolloutTiming:
    rounds: int = 0
    seconds: float = 0.0

    @property
    def per_round_ms(self) -> float:
        return self.seconds / self.rounds * 1000 if self.rounds else 0.0


def measure_rollout_speed(rounds: int = 200, seed: int = 1) -> RolloutTiming:
    import time

    from majiang.sim.round import run_round

    rng = random.Random(seed)
    deciders = [FastDecider() for _ in range(4)]
    started = time.perf_counter()
    for index in range(rounds):
        run_round(deciders, dealer=index % 4, rng=rng)
    return RolloutTiming(rounds=rounds, seconds=time.perf_counter() - started)


__all__ = [
    "FastDecider",
    "RolloutTiming",
    "measure_rollout_speed",
    "quick_shanten_after",
]
