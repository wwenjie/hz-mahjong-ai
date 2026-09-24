"""两条互斥路线的期望值评估（tasks.md 5.6）。

平台上有两条通往高番的路，且互相排斥：

- **七对 + 财飘**：七对禁止任何吃碰杠。``6 对 + 1 财神`` 就是听任意（爆头），
  是本平台最廉价的爆头入口，而爆头又是财飘链的唯一前提。上限极高。
- **副露 + 杠链**：吃碰加速成牌、杠提供「无暴露的 ×2」并自带补牌。成牌率高、
  番数上限低。

按 design.md D12 的要求，两条路线**各自独立估值后再比较**，不允许用一个打分函数隐式
混合——因为两者在关键分叉点上的最优方向常常相反（例如「要不要吃这张牌」）。

估值口径（``期望得分 ≈ 胜率 × 该路线可及的番 × 庄闲赔付 − 失败时承担的赔付``）：
公式里的常数都是**先验**，待任务 6.11 用自对弈数据校准；结构本身才是重点。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from majiang.rules import score as score_module
from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules.melds import Meld
from majiang.rules.situation import Situation
from majiang.rules.tiles import GOD

from . import risk

PAIR = "pair"
MELD = "meld"

# 向听 → 最终自摸概率：**按路线各自实测**（2026-09-23，本地模拟器各 1000 局，
# 四座同时承诺同一条路线，因此对「该路线」无混淆；复跑见 `tools/calibrate.py`）。
#
# 路线无关的共用表是 5.6 首版失败的原因：七对不能吃碰，同向听的实际成牌率显著更低。
# 实测差距在中段最大——2 向听 10.5% vs 17.7%，七对只有副露的六成左右。共用表会把
# 七对估高一倍，于是策略一路去做七对而放弃速度（A/B：胡率 9.7% vs 29%）。
WIN_RATE_PAIR: tuple[float, ...] = (0.327, 0.140, 0.105, 0.088, 0.067, 0.041, 0.029)
WIN_RATE_MELD: tuple[float, ...] = (0.399, 0.214, 0.177, 0.159, 0.146, 0.097, 0.097)
WIN_RATE_REFERENCE_DRAWS = 45.0
DRAWS_SCALE_MIN = 0.40
DRAWS_SCALE_MAX = 1.35

# 胡牌时的**实际**期望番：同样是按路线实测（承诺七对 700 局 / 承诺副露 400 局）。
#
# 这里曾按「七对分支 ×2」给七对路线记功，是 5.6 的第二处模型误差。实测：
#   承诺副露：均番 1.07（平胡 93%、七对 6%）
#   承诺七对：均番 1.68（七对 60%、平胡 38%——**不吃碰并不等于一定胡七对**）
# 于是七对路线换来 1.57× 的番，但同向听胜率只有 0.59×，净 0.93 < 1：**即便同向听，
# 七对也略差**。按理论分支记 2.0 会得出 1.87× 的番优势，恰好抹平该劣势（1.10 > 1），
# 这正是首版一路偏向七对的机制。
PAIR_MEAN_FAN = 1.68
MELD_MEAN_FAN = 1.07
# 财神同时是爆头与链的入口，对两条路线都成立，故给同等上抬——比较不受影响
GOD_FAN_UPLIFT = 1.25
GOD_UPLIFT_MIN = 2
OPPONENT_FAN_PRIOR = 2.0
VALUE_MAX = 1e9


class Route(StrEnum):
    PAIR = PAIR
    MELD = MELD


@dataclass(frozen=True, slots=True)
class RouteValuation:
    route: Route
    feasible: bool
    route_shanten: int
    reach: float
    expected_fan: float
    gain: float
    loss: float
    value: float
    note: str = ""

    def describe(self) -> str:
        flag = "" if self.feasible else "（不可行）"
        return (
            f"{self.route.value}: 向听={self.route_shanten}{flag} 到听={self.reach:.2f} "
            f"番≈{self.expected_fan:.1f} 期望={self.value:.1f} {self.note}"
        )


def pair_progress(counts: Sequence[int]) -> tuple[int, int]:
    """返回（已成对子数, 单张数）；财神按可补一张单张计入对子。"""
    real = list(counts)
    wildcards = real[GOD]
    real[GOD] = 0
    pairs = sum(amount // 2 for amount in real)
    singles = sum(amount % 2 for amount in real)
    completed = min(singles, wildcards)
    return pairs + completed + (wildcards - completed) // 2, singles - completed


def natural_quads(counts: Sequence[int]) -> int:
    return sum(1 for tile, amount in enumerate(counts) if tile != GOD and amount >= tiles.COPIES_PER_KIND)


def concealed_gang_potential(counts: Sequence[int]) -> int:
    return natural_quads(counts)


def win_probability(route_shanten: int, draws_left: int, *, route: Route = Route.MELD) -> float:
    """该路线、该向听下最终自摸的概率（查该路线的实测表，再按剩余摸牌数缩放）。"""
    table = WIN_RATE_PAIR if route is Route.PAIR else WIN_RATE_MELD
    index = max(0, min(route_shanten, len(table) - 1))
    scale = draws_left / WIN_RATE_REFERENCE_DRAWS
    return min(0.95, table[index] * min(DRAWS_SCALE_MAX, max(DRAWS_SCALE_MIN, scale)))


def expected_pay(fan: float, situation: Situation, base_score: int) -> float:
    settlement = score_module.settle(max(1, int(round(fan))), base_score)
    is_dealer = situation.table.dealer_seat == situation.seat
    return float(settlement.line(dealer_hu=is_dealer).win)


def expected_loss(situation: Situation, base_score: int) -> float:
    deltas = score_module.seat_deltas(
        int(OPPONENT_FAN_PRIOR),
        base_score,
        winner_seat=(situation.seat + 1) % 4,
        dealer_seat=situation.table.dealer_seat,
    )
    return abs(min(deltas[situation.seat], 0))


def pair_route_fan(counts: Sequence[int]) -> float:
    """七对路线的实际期望番（实测均值 + 财神上抬）。"""
    return PAIR_MEAN_FAN * (GOD_FAN_UPLIFT if counts[GOD] >= GOD_UPLIFT_MIN else 1.0)


def meld_route_fan(counts: Sequence[int]) -> float:
    """副露路线的实际期望番（实测均值 + 财神上抬）。"""
    return MELD_MEAN_FAN * (GOD_FAN_UPLIFT if counts[GOD] >= GOD_UPLIFT_MIN else 1.0)


def evaluate(
    situation: Situation,
    *,
    base_score: int = 1,
    counts: Sequence[int] | None = None,
    melds: Sequence[Meld] | None = None,
) -> tuple[RouteValuation, RouteValuation]:
    """评估两条路线的期望得分，返回 ``(七对路线, 副露路线)``。

    ``counts`` / ``melds`` 可覆盖当前手牌与副露，用于回答反事实问题——例如
    「打了这张牌之后两条路线各值多少」「吃了这张牌之后值多少」。
    """
    hand = situation.hand
    resolved = tuple(melds) if melds is not None else hand.melds
    meld_count = len(resolved)
    counts = list(counts) if counts is not None else list(hand.counts)
    draws_left = situation.table.draws_left

    risks = risk.assess(situation)
    survival = risk.lap_survival(risks)
    opponent_win = 1.0 - survival
    loss = expected_loss(situation, base_score)

    pair_feasible = meld_count == 0
    pairs, _singles = pair_progress(counts)
    pair_shanten = (
        shanten_module.seven_pairs_shanten(counts)
        if pair_feasible
        else shanten_module.SHANTEN_MAX
    )
    pair_win = (
        win_probability(pair_shanten, draws_left, route=Route.PAIR) if pair_feasible else 0.0
    )
    pair_fan = pair_route_fan(counts)
    pair_gain = pair_win * expected_pay(pair_fan, situation, base_score)
    pair_note = ""
    if pair_feasible and pairs >= shanten_module.SEVEN_PAIRS_MAX_PAIRS and counts[GOD] >= 1:
        pair_note = "已近廉价爆头入口（6 对 + 1 财神）"
    pair_value = pair_gain - (1.0 - pair_win) * opponent_win * loss if pair_feasible else -VALUE_MAX

    meld_shanten = shanten_module.shanten_any(counts, meld_count)
    meld_win = win_probability(meld_shanten, draws_left, route=Route.MELD)
    meld_fan = meld_route_fan(counts)
    meld_gain = meld_win * expected_pay(meld_fan, situation, base_score)
    meld_value = meld_gain - (1.0 - meld_win) * opponent_win * loss

    return (
        RouteValuation(
            route=Route.PAIR,
            feasible=pair_feasible,
            route_shanten=pair_shanten,
            reach=pair_win,
            expected_fan=pair_fan,
            gain=pair_gain,
            loss=loss,
            value=pair_value,
            note=pair_note,
        ),
        RouteValuation(
            route=Route.MELD,
            feasible=True,
            route_shanten=meld_shanten,
            reach=meld_win,
            expected_fan=meld_fan,
            gain=meld_gain,
            loss=loss,
            value=meld_value,
        ),
    )


def preferred(situation: Situation, *, base_score: int = 1) -> RouteValuation:
    """当前更优的路线。"""
    pair, meld = evaluate(situation, base_score=base_score)
    return pair if pair.value >= meld.value else meld


def worth_melding(situation: Situation, *, base_score: int = 1, melds: Sequence[Meld]) -> bool:
    """吃/碰（副露变为 ``melds``）之后，副露路线的期望是否超过保留七对路线。"""
    pair, meld = evaluate(situation, base_score=base_score)
    if not pair.feasible:
        return True
    _, after = evaluate(situation, base_score=base_score, melds=melds)
    return after.value > pair.value


__all__ = [
    "MELD",
    "PAIR",
    "Route",
    "RouteValuation",
    "WIN_RATE_MELD",
    "WIN_RATE_PAIR",
    "concealed_gang_potential",
    "evaluate",
    "expected_loss",
    "expected_pay",
    "meld_route_fan",
    "natural_quads",
    "pair_progress",
    "pair_route_fan",
    "preferred",
    "win_probability",
    "worth_melding",
]
