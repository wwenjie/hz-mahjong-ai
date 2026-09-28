"""动作模型。

平台的动作枚举：``discard / chi / peng / gang / hu / pass``。吃可附带 ``tiles``
指定使用哪两张手牌。``pass`` 按官方最小 Bot 的写法带一个空牌码。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import tiles
from . import win
from .melds import CHI_MAX_PER_HAND
from .situation import (
    ACTIONABLE_PHASES,
    PHASE_DRAW,
    PHASE_RESPONSE_CHI,
    PHASE_RESPONSE_PENG,
    Situation,
    SituationError,
)

DISCARD = "discard"
CHI = "chi"
PENG = "peng"
GANG = "gang"
HU = "hu"
PASS = "pass"

ACTION_KINDS = (DISCARD, CHI, PENG, GANG, HU, PASS)

ANGANG = "angang"
BUGANG = "bugang"
MINGGANG = "minggang"


class ActionError(ValueError):
    """动作非法。"""


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    tile: int | None = None
    tiles: tuple[int, ...] = ()
    gang_kind: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ActionError(f"未知动作: {self.kind!r}")
        if self.kind == DISCARD and self.tile is None:
            raise ActionError("出牌必须指定牌")
        if self.kind == GANG and self.gang_kind not in (ANGANG, BUGANG, MINGGANG):
            raise ActionError(f"杠必须指定类型: {self.gang_kind!r}")
        if self.kind != GANG and self.gang_kind is not None:
            raise ActionError(f"{self.kind} 不应带 gang_kind")

    def to_payload(self) -> dict[str, object]:
        """转成提交给 ``POST /api/games/{id}/action`` 的请求体。"""
        if self.kind == PASS:
            return {"action": PASS, "tile": ""}
        if self.kind == HU:
            return {"action": HU}
        payload: dict[str, object] = {"action": self.kind, "tile": tiles.to_code(self.tile)}  # type: ignore[arg-type]
        if self.kind == CHI and self.tiles:
            payload["tiles"] = tiles.to_codes(self.tiles)
        return payload

    def describe(self) -> str:
        if self.kind == PASS:
            return "pass"
        if self.kind == HU:
            return "hu"
        label = tiles.to_code(self.tile)  # type: ignore[arg-type]
        if self.kind == GANG:
            return f"gang({self.gang_kind}):{label}"
        if self.kind == CHI:
            return f"chi:{label}+{'+'.join(tiles.to_codes(self.tiles))}"
        return f"{self.kind}:{label}"


# 抓打圈封锁的范围。文档写作「不能吃、碰、明杠（仅暗杠与自摸胡）」，
# 括号内是对可用动作的穷举，因此补杠也按禁用处理（tasks.md 2.14 待实测）。
GANGS_ALLOWED_WHEN_RESTRICTED = frozenset({ANGANG})

CHI_WINDOW_OFFSETS: tuple[tuple[int, int], ...] = ((-2, -1), (-1, 1), (1, 2))


def chi_combinations(hand_counts: Sequence[int], offered_tile: int) -> tuple[tuple[int, int], ...]:
    """列出可用哪两张手牌吃这张牌。"""
    combos: list[tuple[int, int]] = []
    for first_offset, second_offset in CHI_WINDOW_OFFSETS:
        first = offered_tile + first_offset
        second = offered_tile + second_offset
        if first < 0 or second >= tiles.TILE_KINDS:
            continue
        if tiles.is_god(first) or tiles.is_god(second):
            continue
        run_start = min(first, second, offered_tile)
        if not tiles.run_is_valid(run_start):
            continue
        if hand_counts[first] >= 1 and hand_counts[second] >= 1:
            combos.append((first, second))
    return tuple(combos)


def concealed_gang_options(situation: Situation) -> tuple[Action, ...]:
    """暗杠与补杠选项。杠需要从牌墙补牌，故受可杠判据约束。

    **抓打圈内只放开暗杠、仍禁补杠。** 平台规则 §1.1 原文是「不能吃、碰、明杠
    （**仅暗杠与自摸胡**）」——括号内是**允许**的动作。原实现两道闸门
    （本函数 + `_turn_actions`）都按「受限则无杠」处理，把暗杠一起禁掉了，
    等于**丢了一类合法动作**。

    暗杠在本平台价值极高：链 +1、自带补牌、不经过对手回合，且**不暴露任何信息**。
    这是规则合规 bug、不是策略调参，修复不需要 A/B 或样本量（agent C 复核指出）。
    """
    if not situation.table.can_gang:
        return ()
    hand = situation.hand
    peng_tiles = {meld.tiles[0] for meld in hand.melds if meld.is_peng}
    options: list[Action] = []
    for tile, amount in enumerate(hand.counts):
        if tiles.is_god(tile):
            continue
        if amount >= tiles.COPIES_PER_KIND:
            options.append(Action(GANG, tile=tile, gang_kind=ANGANG))
        elif not situation.is_restricted and amount >= 1 and tile in peng_tiles:
            # 补杠在抓打圈内仍禁：规则只放行「暗杠与自摸胡」，补杠会经对手回合、
            # 且要打出手上的牌（圈内只能打刚摸到的那张）。
            options.append(Action(GANG, tile=tile, gang_kind=BUGANG))
    return tuple(options)


def _turn_actions(situation: Situation) -> tuple[Action, ...]:
    hand = situation.hand
    actions: list[Action] = []
    drawn = situation.drawn_tile
    # 只能自摸：胡必须发生在本人摸牌之后。碰/吃之后轮到本人时手上会多一张（副露腾出的
    # 空位），但并未摸牌，此时即使牌型已成胡也不可胡——那是点炮/抢杠，平台禁止。
    if drawn is not None and win.is_winning_shape(hand.counts, hand.meld_count):
        actions.append(Action(HU))
    if situation.is_restricted and drawn is None:
        raise SituationError("抓打圈内出牌只能打刚摸到的牌，但快照未给出 drawn_tile")
    for tile, amount in enumerate(hand.counts):
        if amount == 0:
            continue
        if situation.is_restricted and tile != drawn:
            continue
        actions.append(Action(DISCARD, tile=tile))
    # **受限时也要给出杠选项**：`concealed_gang_options` 自己区分暗杠/补杠
    # （圈内只放行暗杠）。原先这里再挡一道，把暗杠一并禁掉。
    actions.extend(concealed_gang_options(situation))
    return tuple(actions)


def _peng_window_actions(situation: Situation) -> tuple[Action, ...]:
    if situation.seat not in situation.responding_seats:
        return ()
    actions = [Action(PASS)]
    offered = situation.offered_tile
    if offered is None or tiles.is_god(offered) or situation.is_restricted:
        return tuple(actions)
    held = situation.hand.counts[offered]
    if held >= tiles.SET_LENGTH - 1:
        actions.append(Action(PENG, tile=offered))
    if held >= tiles.SET_LENGTH and situation.table.can_gang:
        actions.append(Action(GANG, tile=offered, gang_kind=MINGGANG))
    return tuple(actions)


def _chi_window_actions(situation: Situation) -> tuple[Action, ...]:
    if situation.seat not in situation.responding_seats:
        return ()
    actions = [Action(PASS)]
    offered = situation.offered_tile
    if offered is None or tiles.is_god(offered) or situation.is_restricted:
        return tuple(actions)
    if situation.hand.chi_count >= CHI_MAX_PER_HAND:
        return tuple(actions)
    for first, second in chi_combinations(situation.hand.counts, offered):
        actions.append(Action(CHI, tile=offered, tiles=(first, second)))
    return tuple(actions)


def legal_actions(situation: Situation) -> tuple[Action, ...]:
    """当前局面下本人可执行的合法动作。"""
    if situation.is_observer or situation.phase not in ACTIONABLE_PHASES:
        return ()
    if situation.phase == PHASE_DRAW:
        if situation.turn != situation.seat:
            return ()
        return _turn_actions(situation)
    if situation.phase == PHASE_RESPONSE_PENG:
        return _peng_window_actions(situation)
    if situation.phase == PHASE_RESPONSE_CHI:
        return _chi_window_actions(situation)
    return ()
