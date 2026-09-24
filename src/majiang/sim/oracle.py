"""完全信息上界测量（仅用于研究，**禁止**用于比赛）。

目的：在投入任何训练之前，先回答一个决定性问题——**如果给策略开天眼（看到四家真实手牌），
胜率能到多少？**

- 若上界只比 25% 的公平基线高一点 ⇒ 这个游戏几乎由运气决定，任何策略（含深度学习/强化
  学习）的空间都窄，训练路线不值得投入；
- 若上界明显更高 ⇒ 说明存在被启发式漏掉的策略空间。

这是**上界**测量：读对手手牌在比赛中属违规（合规红线），因此本模块只进研究脚本，
不进交付路径。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules.action import DISCARD, Action
from majiang.rules.situation import Situation
from majiang.rules.tiles import GOD
from majiang.strategy.policy import HeuristicDecider
from majiang.strategy.search import SearchConfig, SearchDecider

# 让对手更难推进的权重：越可能被对手要的牌越该先打掉
FEED_WEIGHT = 2.0


@dataclass
class OracleStats:
    calls: int = 0


class OracleDecider:
    """用四家真实手牌评估出牌：兼顾「自己推进」与「拖慢对手」。"""

    name = "oracle"

    def __init__(self, inner: object | None = None) -> None:
        self.inner = inner
        self.stats = OracleStats()
        self._state = None

    def observe_state(self, state: object) -> None:
        self._state = state

    def choose(self, situation: Situation, actions, *, budget_ms: int = 0) -> Action | None:
        # 只覆盖「出牌」这一步：胡/杠/弃胡飘仍交给启发式——否则会跳过胡牌判定，
        # 表现为 oracle 永不自摸（实测 0% 胜率）。
        first = self._fallback(situation, actions)
        if first is None or first.kind != DISCARD:
            return first
        state = self._state
        if state is None or situation.drawn_tile is None:
            return first
        discards = [action for action in actions if action.kind == DISCARD]
        if not discards:
            return first

        seat = situation.seat
        meld_count = situation.hand.meld_count
        best: tuple[float, Action] | None = None
        for action in discards:
            tile = action.tile
            assert tile is not None
            counts = list(situation.hand.counts)
            counts[tile] -= 1
            mine = shanten_module.quick_shanten(counts, meld_count)

            score = -4.0 * mine
            if tile == GOD:
                score -= 20.0
            score -= FEED_WEIGHT * self._feed_value(state, seat, tile)
            if best is None or score > best[0]:
                best = (score, action)
        self.stats.calls += 1
        return best[1] if best is not None else first

    @staticmethod
    def _feed_value(state: object, seat: int, tile: int) -> float:
        """该牌对对手的「价值」：持有对子/搭子的对手越多，打出去越亏。"""
        value = 0.0
        for other in range(4):
            if other == seat:
                continue
            hand = state.seats[other].hand  # type: ignore[attr-defined]
            if hand[tile] >= 2:
                value += 1.0
            elif tiles.is_number(tile):
                for offset in (1, 2):
                    neighbour = tile + offset
                    if neighbour < tiles.TILE_KINDS and hand[neighbour]:
                        value += 0.3
        return value

    def _fallback(self, situation: Situation, actions) -> Action | None:
        chooser = getattr(self.inner, "choose", None)
        if callable(chooser):
            return chooser(situation, actions, budget_ms=0)
        discards = [action for action in actions if action.kind == DISCARD]
        return discards[0] if discards else None


class OracleSearchDecider(SearchDecider):
    """完全信息前瞻搜索：不做确定化采样，**直接用真实局面**滚出。

    这是真正的「上界」形态——它去掉了 PIMC 的采样误差与信息缺失，因此其表现给出了
    「任何策略在完全信息下的可达水平」。仅用于测量，读对手手牌在比赛中属违规。

    实现在于只覆盖 ``determinize``：其余（候选筛选、滚出、比较）与常规搜索完全一致，
    这样测出的差异纯粹来自信息量，而不是评分函数变了。
    """

    name = "oracle-search"

    def __init__(self, heuristic: HeuristicDecider | None = None, config: SearchConfig | None = None):
        super().__init__(heuristic, config)
        self._true_state = None

    def observe_state(self, state: object) -> None:
        self._true_state = state

    def determinize(self, situation: Situation, rng) -> object:  # type: ignore[override]
        if self._true_state is None:
            return super().determinize(situation, rng)
        return copy.deepcopy(self._true_state)


__all__ = ["OracleDecider", "OracleSearchDecider", "OracleStats"]
