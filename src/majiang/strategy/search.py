"""确定化前瞻搜索（tasks.md 5.15）。

思路（PIMC）：先用启发式评分筛出前 K 个出牌候选，再对每个候选做若干次
「确定化 + 滚出」——即按公开信息采样对手手牌与牌墙，用极速策略把本局滚到底，
以本人最终得分的**平均**作为该候选的期望值，取最大者。

两点设计约束：

1. **合规**：采样只使用公开信息（自己的手牌、四家副露、四家弃牌、各家暗手张数、
   牌墙剩余）。策略侧从不读取真实对手手牌——确定化出来的手牌是**采样**，不是观测。
2. **成本**：``样本数 × 候选数 × 单次滚出``。实测极速策略滚出约 6.9 ms/局（启发式是
   208 ms/局），因此 12 样本 × 3 候选 × 约 5 ms ≈ 180 ms，落在决策预算内。
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass

from majiang.rules import tiles
from majiang.rules.action import DISCARD, Action
from majiang.rules.situation import PHASE_DRAW, Situation
from majiang.rules.tiles import GOD
from majiang.sim.round import (
    SEATS,
    RoundResult,
    RoundState,
    Seat,
    apply_discard,
    play_round,
    resolve_responses,
)

from .policy import DiscardScore, HeuristicDecider

DEFAULT_SAMPLES = 12
DEFAULT_TOP_K = 3
ROLLOUT_MAX_TURNS = 150


@dataclass(frozen=True, slots=True)
class SearchConfig:
    samples: int = DEFAULT_SAMPLES
    top_k: int = DEFAULT_TOP_K
    max_turns: int = ROLLOUT_MAX_TURNS
    seed: int = 20260923


class SearchDecider:
    """启发式筛候选 + 确定化滚出比较期望得分。"""

    name = "search"

    def __init__(
        self,
        heuristic: HeuristicDecider | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.heuristic = heuristic or HeuristicDecider()
        self.config = config or SearchConfig()
        self.last_reason = ""
        self.last_detail: dict[str, object] = {}

    def configure(self, tournament) -> None:
        self.heuristic.configure(tournament)

    def choose(
        self,
        situation: Situation,
        actions: Sequence[Action],
        *,
        budget_ms: int,
    ) -> Action | None:
        self.last_reason = ""
        self.last_detail = {}
        if situation.phase != PHASE_DRAW:
            chosen = self.heuristic.choose(situation, actions, budget_ms=budget_ms)
            self._copy_detail()
            return chosen

        # 先让启发式给出本回合动作：胡与杠直接采纳——**搜索只替换「出牌」这一步**。
        # 若在此处直接跳到候选排序，就会绕过胡/杠判定，导致永远不胡（实测表现为
        # 4 个搜索座全部流局、搜索座 0% 胡率）。
        first = self.heuristic.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = self.heuristic.last_reason
        self.last_detail = dict(self.heuristic.last_detail)
        if first is None or first.kind != DISCARD:
            return first

        ranked = self._rank_discards(situation, actions)
        if not ranked:
            return first
        if len(ranked) == 1:
            self.last_reason = f"搜索：只有一个候选 {ranked[0].describe()}"
            return self._action_for(situation, actions, ranked[0])
        if budget_ms and budget_ms < 200:
            # 预算太紧就不搜，退回启发式（宁可按期出牌，也不要超时兜底）
            self.last_reason = "搜索：预算不足，退回启发式"
            return first

        candidates = ranked[: self.config.top_k]
        rng = random.Random(self._seed_for(situation))
        totals = {score.tile: 0.0 for score in candidates}
        for _ in range(self.config.samples):
            world = self.determinize(situation, rng)
            for score in candidates:
                totals[score.tile] += self._rollout(world, situation, score.tile)
        best = max(candidates, key=lambda score: totals[score.tile])
        self.last_detail = {
            "search": {
                score.tile: round(totals[score.tile] / self.config.samples, 2)
                for score in candidates
            },
            "samples": self.config.samples,
        }
        self.last_reason = (
            f"搜索：{self.config.samples} 个确定化世界里 "
            + "、".join(
                f"{tiles.to_code(score.tile)}={totals[score.tile] / self.config.samples:.1f}"
                for score in candidates
            )
        )
        return self._action_for(situation, actions, best)

    # ---- 内部 -------------------------------------------------------------

    def _rank_discards(self, situation: Situation, actions: Sequence[Action]) -> list[DiscardScore]:
        scores = [
            self.heuristic._score_discard(situation, action)
            for action in actions
            if action.kind == "discard"
        ]
        scores.sort(key=lambda item: item.total, reverse=True)
        return scores

    @staticmethod
    def _action_for(
        situation: Situation, actions: Sequence[Action], score: DiscardScore
    ) -> Action | None:
        for action in actions:
            if action.kind == "discard" and action.tile == score.tile:
                return action
        return None

    def _copy_detail(self) -> None:
        self.last_detail = dict(self.heuristic.last_detail)
        if not self.last_reason:
            self.last_reason = self.heuristic.last_reason

    @staticmethod
    def _seed_for(situation: Situation) -> int:
        base = sum(amount * (index + 1) for index, amount in enumerate(situation.hand.counts))
        return (base * 1000003 + situation.table.round_no * 7919) & 0x7FFFFFFF

    def determinize(self, situation: Situation, rng: random.Random) -> RoundState:
        """按公开信息采样一个完整局面（对手手牌与牌墙内容由采样决定）。"""
        pool = [tile for tile in range(tiles.TILE_KINDS) for _ in range(tiles.COPIES_PER_KIND)]
        for tile, amount in enumerate(situation.hand.counts):
            for _ in range(amount):
                pool.remove(tile)
        for meld in situation.all_melds:
            for tile in meld.tiles:
                pool.remove(tile)
        for seat_discards in situation.discards:
            for tile in seat_discards:
                pool.remove(tile)
        rng.shuffle(pool)

        seats: list[Seat] = []
        for seat in range(SEATS):
            if seat == situation.seat:
                hand = list(situation.hand.counts)
                melds = list(situation.hand.melds)
            else:
                melds = list(situation.melds_for(seat))
                concealed = situation.hand_counts_for(seat)
                if concealed < 0:
                    concealed = len(pool)
                hand = [0] * tiles.TILE_KINDS
                for _ in range(concealed):
                    if not pool:
                        break
                    hand[pool.pop()] += 1
            seats.append(
                Seat(
                    hand=hand,
                    melds=melds,
                    discards=list(situation.discards[seat]) if seat < len(situation.discards) else [],
                    god_count=hand[GOD],
                )
            )
        return RoundState(
            wall=pool,
            seats=seats,
            dealer=situation.table.dealer_seat,
            round_no=situation.table.round_no,
            turn=situation.seat,
            catch_play=situation.god.catch_play,
            god_discarder=situation.god.god_discarder_seat,
        )

    def _rollout(
        self,
        world: RoundState,
        situation: Situation,
        tile: int,
    ) -> float:
        state = copy.deepcopy(world)
        seat = situation.seat
        deciders = self._rollout_deciders()
        apply_discard(state, seat, tile, situation.drawn_tile)
        claim = resolve_responses(state, seat, tile, deciders)
        current, need_draw = (seat + 1) % SEATS, True
        if claim is not None:
            current, need_draw = claim[0], False
        result: RoundResult = play_round(
            state,
            deciders,
            current=current,
            drawn=None,
            need_draw=need_draw,
            max_turns=self.config.max_turns,
        )
        return float(result.scores[seat])

    @staticmethod
    def _rollout_deciders() -> list[object]:
        from .rollout import FastDecider

        return [FastDecider() for _ in range(SEATS)]


__all__ = ["DEFAULT_SAMPLES", "DEFAULT_TOP_K", "SearchConfig", "SearchDecider"]
