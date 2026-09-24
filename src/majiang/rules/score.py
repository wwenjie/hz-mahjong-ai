"""得分结算。

得分 = 底分 × 番型倍率 × (庄家 ×8 / 闲家 ×1)，三家分别结算：

- 庄家自摸：三家闲家各付 ``底分 × 倍率 × 8``
- 闲家自摸：庄家付 ``底分 × 倍率 × 8``，另外两个闲家各付 ``底分 × 倍率 × 1``

平台变更记录里给出的实测值可直接作为回归用例：
``fan=4`` 且庄家自摸时四家净分为 ``[-32, 96, -32, -32]``；``fan=2`` 且庄家自摸时
每家闲家付 16。
"""

from __future__ import annotations

from dataclasses import dataclass

DEALER_MULTIPLIER = 8
NON_DEALER_MULTIPLIER = 1
SEATS = 4
BASE_SCORE_MIN = 1
BASE_SCORE_MAX = 10000


class ScoreError(ValueError):
    """结算入参非法。"""


@dataclass(frozen=True, slots=True)
class ScoreLine:
    win: int
    lose: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Settlement:
    dealer_hu: ScoreLine
    nondealer_hu: ScoreLine

    def line(self, *, dealer_hu: bool) -> ScoreLine:
        return self.dealer_hu if dealer_hu else self.nondealer_hu

    def to_api_shape(self) -> dict[str, dict[str, object]]:
        """按平台番型计算端点的响应形状输出。

        ``lose`` 的顺序按庄家优先排列（与平台文档示例 ``[8, 1, 1]`` 一致）；
        由于示例未标明胡牌者座位，精确顺序待在 2.13 对拍中确认。
        """
        return {
            "dealer_hu": {"win": self.dealer_hu.win, "lose": list(self.dealer_hu.lose)},
            "nondealer_hu": {"win": self.nondealer_hu.win, "lose": list(self.nondealer_hu.lose)},
        }


def _check_base(base: int) -> None:
    if not BASE_SCORE_MIN <= base <= BASE_SCORE_MAX:
        raise ScoreError(f"底分应在 {BASE_SCORE_MIN}–{BASE_SCORE_MAX}，实际 {base}")


def settle(fan: int, base: int = 1) -> Settlement:
    if fan < 0:
        raise ScoreError(f"番数不能为负: {fan}")
    _check_base(base)
    dealer_pay = base * fan * DEALER_MULTIPLIER
    non_dealer_pay = base * fan * NON_DEALER_MULTIPLIER
    dealer_hu = ScoreLine(
        win=dealer_pay * (SEATS - 1),
        lose=(dealer_pay,) * (SEATS - 1),
    )
    nondealer_lose = (dealer_pay, *((non_dealer_pay,) * (SEATS - 2)))
    nondealer_hu = ScoreLine(
        win=sum(nondealer_lose),
        lose=nondealer_lose,
    )
    return Settlement(dealer_hu=dealer_hu, nondealer_hu=nondealer_hu)


def seat_deltas(
    fan: int,
    base: int,
    winner_seat: int,
    dealer_seat: int,
    seats: int = SEATS,
) -> tuple[int, ...]:
    """按座位顺序返回四家净分变化。

    付款额取决于胡牌方是否为庄家：庄家自摸时三家闲家各付 ×8；闲家自摸时庄家付 ×8、
    另外两家闲家各付 ×1。
    """
    if not 0 <= winner_seat < seats or not 0 <= dealer_seat < seats:
        raise ScoreError(f"座位越界: winner={winner_seat} dealer={dealer_seat} seats={seats}")
    if fan < 0:
        raise ScoreError(f"番数不能为负: {fan}")
    _check_base(base)
    winner_is_dealer = winner_seat == dealer_seat
    payments: list[int] = []
    for seat in range(seats):
        if seat == winner_seat:
            payments.append(0)
        elif winner_is_dealer or seat == dealer_seat:
            payments.append(base * fan * DEALER_MULTIPLIER)
        else:
            payments.append(base * fan * NON_DEALER_MULTIPLIER)
    deltas = [-payment for payment in payments]
    deltas[winner_seat] = -sum(deltas)
    return tuple(deltas)
