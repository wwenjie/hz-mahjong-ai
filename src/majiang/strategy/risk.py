"""对手自摸风险估计（tasks.md 5.12）。

按 design.md D14 的分解式建模：

    每对手自摸风险      pᵢ = sᵢ × rᵢ
      听牌概率          sᵢ = P(对手 i 当前处于听牌，含听任意)
      条件自摸率        rᵢ = E[自摸张数] / 剩余可摸牌数
    本圈无人自摸生存率  q  ≈ ∏(1 − pᵢ)

**只用公开信息**：对手副露、对手弃牌、全局资源、局面状态；不含任何对手手牌。

这里是**启发式实现**（供决策热路径使用，微秒级）。它同时是接口样板：`6B` 会训练一个
数据驱动模型替换 `ReadyModel`，其余调用方无需改动。已知偏差方向——听任意状态在公开
信息下不可观测，会低估 `pᵢ` 从而高估 `q`，因此末尾统一施加保守修正。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from majiang.rules import tiles
from majiang.rules.situation import Situation

SEATS = 4
OBSERVER = -1

# 保守修正：听任意不可观测导致 p_i 被低估，按固定比例上抬，避免 q 偏高而过度贪链
CONSERVATIVE_UPLIFT = 1.25

# 启发式先验（待 6B 用数据校准）
BASE_READY = 0.06
DISCARD_PROGRESS_WEIGHT = 0.30
MELD_READY_WEIGHT = 0.16

READY_MAX = 0.85
READY_MIN = 0.02
CONDITIONAL_DRAW_MAX = 0.40
EXPECTED_WAITS = 4.0


@dataclass(frozen=True, slots=True)
class OpponentRisk:
    seat: int
    ready_probability: float
    self_draw_probability: float


@runtime_checkable
class ReadyModel(Protocol):
    """对手听牌概率模型。返回每位对手（不含本人）的估计。"""

    def estimate(
        self,
        situation: Situation,
        *,
        opponents: Sequence[int],
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class HeuristicReadyModel:
    """基于公开信息的启发式听牌概率。"""

    base: float = BASE_READY
    discard_weight: float = DISCARD_PROGRESS_WEIGHT
    meld_weight: float = MELD_READY_WEIGHT

    def estimate(self, situation: Situation, *, opponents: Sequence[int]) -> tuple[float, ...]:
        game_progress = situation.table.progress
        out: list[float] = []
        for seat in opponents:
            melds = situation.melds_for(seat)
            discards = situation.discards[seat] if seat < len(situation.discards) else ()
            # 副露越多、弃牌越多说明手牌推进越深；副露还直接减少了手牌张数
            discard_term = 1.0 - math.exp(-len(discards) / 6.0)
            ready = (
                self.base
                + self.discard_weight * discard_term
                + self.meld_weight * len(melds)
                + 0.18 * game_progress
            )
            if situation.god.restricts(seat):
                ready -= 0.05  # 抓打圈内受限，推进被拖慢
            out.append(min(READY_MAX, max(READY_MIN, ready)))
        return tuple(out)


def conditional_self_draw(draws_left: int) -> float:
    """条件自摸率：听牌后每次摸牌命中胡牌张的概率。"""
    drawable = max(1, draws_left)
    return min(CONDITIONAL_DRAW_MAX, EXPECTED_WAITS / drawable)


def opponents_of(seat: int, seats: int = SEATS) -> tuple[int, ...]:
    return tuple(other for other in range(seats) if other != seat)


def assess(
    situation: Situation,
    *,
    model: ReadyModel | None = None,
    uplift: float = CONSERVATIVE_UPLIFT,
) -> tuple[OpponentRisk, ...]:
    """估计每位对手在其下次摸牌时自摸的风险。"""
    if situation.is_observer:
        return ()
    opponents = opponents_of(situation.seat)
    ready = (model or HeuristicReadyModel()).estimate(situation, opponents=opponents)
    conditional = conditional_self_draw(situation.table.draws_left)
    risks: list[OpponentRisk] = []
    for seat, ready_probability in zip(opponents, ready):
        risk = min(1.0, ready_probability * conditional * uplift)
        risks.append(
            OpponentRisk(
                seat=seat,
                ready_probability=ready_probability,
                self_draw_probability=risk,
            )
        )
    return tuple(risks)


def lap_survival(risks: Sequence[OpponentRisk]) -> float:
    """本圈（三家各摸一次）无人自摸的生存概率 q。"""
    survival = 1.0
    for risk in risks:
        survival *= 1.0 - risk.self_draw_probability
    return max(0.0, min(1.0, survival))


def lap_survival_for(situation: Situation, **kwargs: object) -> float:
    return lap_survival(assess(situation, **kwargs))  # type: ignore[arg-type]


def threat_level(risks: Sequence[OpponentRisk], seat: int) -> float:
    """某个座位带来的威胁（用于「压制对象应该是庄家」这类判断）。"""
    for risk in risks:
        if risk.seat == seat:
            return risk.self_draw_probability
    return 0.0


def visible_need(tile: int) -> float:
    """该牌种对对手的吸引力：中张比幺九与字牌更容易被要。"""
    if not tiles.is_number(tile):
        return 0.4
    rank = tiles.rank(tile)
    if 3 <= rank <= 7:
        return 1.0
    return 0.6


__all__ = [
    "CONSERVATIVE_UPLIFT",
    "HeuristicReadyModel",
    "OpponentRisk",
    "ReadyModel",
    "assess",
    "conditional_self_draw",
    "lap_survival",
    "lap_survival_for",
    "opponents_of",
    "threat_level",
    "visible_need",
]
