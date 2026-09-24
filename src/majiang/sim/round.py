"""单局模拟（tasks.md 6.1）。

与真实协议共用同一套东西：规则引擎的判定、`Situation` 局面结构、`legal_actions`
动作枚举，以及运行时依赖的 `Decider` 接口。因此在模拟器里验证过的策略路径，就是线上
跑的那条路径。

**信息边界由此天然成立**：模拟器持有四家真实手牌，但交给每个座位的 `Situation` 只含
该座位自己的手牌 + 公开信息（四家副露、弃牌、牌墙剩余、抓打圈状态）。

抓打圈的存续条件容易写错：它由**打出财神者本人**再打出一张非财神牌才解除；其余受限
玩家在其间打牌不影响它，且他们只能打出刚摸到的牌、不能吃碰明杠。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from majiang.rules import melds as melds_module
from majiang.rules import score as score_module
from majiang.rules import tiles, win
from majiang.rules.action import (
    CHI,
    DISCARD,
    GANG,
    HU,
    PASS,
    PENG,
    Action,
    chi_combinations,
    legal_actions,
)
from majiang.rules.fan import compute_fan
from majiang.rules.god import NO_DISCARDER, GodState
from majiang.rules.hand import Hand
from majiang.rules.melds import Meld
from majiang.rules.situation import (
    PHASE_DRAW,
    PHASE_RESPONSE_CHI,
    PHASE_RESPONSE_PENG,
    Situation,
)
from majiang.rules.table import INITIAL_WALL, RESERVED_TILES, TableState
from majiang.rules.tiles import GOD

SEATS = 4
MAX_TURNS = 400
DECISION_BUDGET_MS = 1000
PIAO = "piao"
GOD_BREAK = "god_break"
BREAK = "break"


@dataclass
class Seat:
    hand: list[int] = field(default_factory=lambda: [0] * tiles.TILE_KINDS)
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)
    chain_count: int = 0
    piao_count: int = 0
    god_count: int = 0


@dataclass
class RoundResult:
    round_no: int
    dealer: int
    winner: int | None
    fan: int
    detail: tuple[str, ...]
    scores: tuple[int, ...]
    draws: int
    turns: int
    god_counts: tuple[int, ...] = (0, 0, 0, 0)

    @property
    def is_flow(self) -> bool:
        return self.winner is None


@dataclass
class RoundState:
    wall: list[int]
    seats: list[Seat]
    dealer: int
    round_no: int
    turn: int
    catch_play: bool = False
    god_discarder: int = NO_DISCARDER

    @property
    def table(self) -> TableState:
        return TableState(
            wall_remaining=len(self.wall), dealer_seat=self.dealer, round_no=self.round_no
        )

    def hand_of(self, seat: int) -> Hand:
        return Hand.from_counts(self.seats[seat].hand, self.seats[seat].melds)

    def god_state(self, seat: int) -> GodState:
        seat_state = self.seats[seat]
        return GodState(
            hand_gods=seat_state.hand[GOD],
            chain_count=seat_state.chain_count,
            piao_count=seat_state.piao_count,
            baotou=win.is_baotou(seat_state.hand, len(seat_state.melds)),
            catch_play=self.catch_play,
            god_discarder_seat=self.god_discarder,
        )


def build_wall(rng: random.Random) -> list[int]:
    wall = [tile for tile in range(tiles.TILE_KINDS) for _ in range(tiles.COPIES_PER_KIND)]
    rng.shuffle(wall)
    return wall


def deal(rng: random.Random, dealer: int, round_no: int = 1) -> RoundState:
    wall = build_wall(rng)
    seats: list[Seat] = []
    for _ in range(SEATS):
        hand = tiles.counts_from(wall[:13])
        del wall[:13]
        seat_state = Seat(hand=hand)
        seat_state.god_count = hand[GOD]
        seats.append(seat_state)
    return RoundState(wall=wall, seats=seats, dealer=dealer, round_no=round_no, turn=dealer)


def situation_for(
    state: RoundState,
    seat: int,
    phase: str,
    *,
    offered: int | None = None,
    responding: Sequence[int] = (),
    drawn: int | None = None,
) -> Situation:
    """为某座位构造局面：只含该座位手牌与公开信息。"""
    return Situation.from_parts(
        seat=seat,
        phase=phase,
        turn=state.turn,
        hand=state.hand_of(seat),
        god=state.god_state(seat),
        table=state.table,
        responding_seats=tuple(responding),
        offered_tile=offered,
        drawn_tile=drawn,
        discards=tuple(tuple(seat_state.discards) for seat_state in state.seats),
        melds=tuple(tuple(seat_state.melds) for seat_state in state.seats),
        hand_counts=tuple(tiles.total_tiles(seat_state.hand) for seat_state in state.seats),
    )


def restricted(state: RoundState, seat: int) -> bool:
    return state.catch_play and state.god_discarder != seat


def _draw(state: RoundState, seat: int) -> int | None:
    """从牌墙取一张；最后 20 张保留不摸，耗尽返回 None。"""
    if len(state.wall) <= RESERVED_TILES:
        return None
    tile = state.wall.pop()
    state.seats[seat].hand[tile] += 1
    if tile == GOD:
        state.seats[seat].god_count += 1
    return tile


def classify_discard(state: RoundState, seat: int, tile: int, drawn: int | None) -> str:
    """判定这次出牌对动作链与抓打圈的影响。"""
    if tile != GOD:
        return BREAK
    seat_state = state.seats[seat]
    meld_count = len(seat_state.melds)
    pre = list(seat_state.hand)
    if drawn is not None and pre[drawn] > 0:
        pre[drawn] -= 1
    post = list(seat_state.hand)
    post[tile] -= 1
    if win.is_baotou(pre, meld_count) and win.is_baotou(post, meld_count):
        return PIAO
    return GOD_BREAK


def apply_discard(state: RoundState, seat: int, tile: int, drawn: int | None) -> str:
    seat_state = state.seats[seat]
    kind = classify_discard(state, seat, tile, drawn)
    seat_state.hand[tile] -= 1
    seat_state.discards.append(tile)
    if tile == GOD:
        if kind == PIAO:
            seat_state.chain_count += 1
            seat_state.piao_count += 1
        else:
            seat_state.chain_count = 0
            seat_state.piao_count = 0
        state.catch_play = True
        state.god_discarder = seat
    else:
        seat_state.chain_count = 0
        seat_state.piao_count = 0
        if state.catch_play and state.god_discarder == seat:
            # 只有打出财神者本人再打非财神牌才解除抓打圈
            state.catch_play = False
            state.god_discarder = NO_DISCARDER
    return kind


def apply_gang(state: RoundState, seat: int, action: Action) -> None:
    seat_state = state.seats[seat]
    tile = action.tile
    assert tile is not None
    if action.gang_kind == "bugang":
        seat_state.hand[tile] -= 1
        for index, meld in enumerate(seat_state.melds):
            if meld.kind == melds_module.PENG and meld.tiles[0] == tile:
                seat_state.melds[index] = Meld(kind=melds_module.GANG, tiles=(tile,) * 4)
                break
    else:
        # 暗杠用掉手上全部四张，随后由调用方补摸一张
        seat_state.hand[tile] -= tiles.COPIES_PER_KIND
        seat_state.melds.append(
            Meld(
                kind=melds_module.GANG,
                tiles=(tile,) * 4,
                concealed=action.gang_kind == "angang",
            )
        )
    seat_state.chain_count += 1


def take_from_discards(state: RoundState, discarder: int, tile: int) -> bool:
    """把被吃碰杠取走的那张牌从弃牌堆移除，避免与副露重复计数。"""
    discards = state.seats[discarder].discards
    for index in range(len(discards) - 1, -1, -1):
        if discards[index] == tile:
            del discards[index]
            return True
    return False


def apply_peng(state: RoundState, seat: int, discarder: int, tile: int) -> None:
    seat_state = state.seats[seat]
    seat_state.hand[tile] -= 2
    seat_state.melds.append(Meld(kind=melds_module.PENG, tiles=(tile,) * 3))
    seat_state.chain_count = 0
    seat_state.piao_count = 0
    take_from_discards(state, discarder, tile)


def apply_minggang(state: RoundState, seat: int, discarder: int, tile: int) -> None:
    seat_state = state.seats[seat]
    seat_state.hand[tile] -= 3
    seat_state.melds.append(Meld(kind=melds_module.GANG, tiles=(tile,) * 4))
    seat_state.chain_count += 1
    take_from_discards(state, discarder, tile)


def apply_chi(state: RoundState, seat: int, discarder: int, tile: int, action: Action) -> None:
    seat_state = state.seats[seat]
    for used in action.tiles:
        seat_state.hand[used] -= 1
    seat_state.melds.append(Meld(kind=melds_module.CHI, tiles=tuple(sorted((*action.tiles, tile)))))
    seat_state.chain_count = 0
    seat_state.piao_count = 0
    take_from_discards(state, discarder, tile)


def chi_options(state: RoundState, seat: int, offered: int) -> tuple[tuple[int, int], ...]:
    if restricted(state, seat) or offered == GOD:
        return ()
    seat_state = state.seats[seat]
    if melds_module.chi_count(seat_state.melds) >= melds_module.CHI_MAX_PER_HAND:
        return ()
    return chi_combinations(seat_state.hand, offered)


def response_order(discarder: int) -> tuple[int, ...]:
    return tuple((discarder + step) % SEATS for step in range(1, SEATS))


def _ask(decider: object, situation: Situation, actions: Sequence[Action]) -> Action | None:
    choose = getattr(decider, "choose", None)
    if not callable(choose):
        return None
    return choose(situation, tuple(actions), budget_ms=DECISION_BUDGET_MS)


def resolve_responses(
    state: RoundState,
    discarder: int,
    tile: int,
    deciders: Sequence[object],
) -> tuple[int, bool] | None:
    """先碰（含明杠）后吃。返回 ``(吃碰方座位, 是否有补牌)``，无人响应返回 None。"""
    if tile == GOD:
        return None  # 财神不能被吃碰杠
    peng_seats = [
        seat
        for seat in response_order(discarder)
        if state.seats[seat].hand[tile] >= 2 and not restricted(state, seat)
    ]
    if peng_seats:
        responding = tuple(peng_seats)
        for seat in peng_seats:
            situation = situation_for(
                state, seat, PHASE_RESPONSE_PENG, offered=tile, responding=responding
            )
            actions = legal_actions(situation)
            chosen = _ask(deciders[seat], situation, actions)
            if chosen is None or chosen.kind == PASS:
                continue
            if chosen.kind == PENG:
                apply_peng(state, seat, discarder, tile)
                return seat, False
            if chosen.kind == GANG:
                apply_minggang(state, seat, discarder, tile)
                return seat, True
    chi_seat = (discarder + 1) % SEATS
    if chi_options(state, chi_seat, tile):
        situation = situation_for(
            state, chi_seat, PHASE_RESPONSE_CHI, offered=tile, responding=(chi_seat,)
        )
        actions = legal_actions(situation)
        chosen = _ask(deciders[chi_seat], situation, actions)
        if chosen is not None and chosen.kind == CHI:
            apply_chi(state, chi_seat, discarder, tile, chosen)
            return chi_seat, False
    return None


def play_round(
    state: RoundState,
    deciders: Sequence[object],
    *,
    current: int | None = None,
    drawn: int | None = None,
    need_draw: bool = True,
    base_score: int = 1,
    observer: object | None = None,
    max_turns: int = MAX_TURNS,
) -> RoundResult:
    """从一个已有局面续跑到底——供前瞻搜索从中途滚出使用。

    ``run_round`` 是本函数 + ``deal`` 的组合；分开是为了让搜索能从任意局面继续，
    而不必重新发牌。
    """
    seat_to_move = state.turn if current is None else current
    turns = 0

    def notify() -> None:
        if callable(observer):
            observer(state)

    notify()

    while turns < max_turns:
        turns += 1
        if need_draw:
            drawn = _draw(state, seat_to_move)
            if drawn is None:
                return flow_result(state, turns)
            need_draw = False
            notify()

        state.turn = seat_to_move
        situation = situation_for(state, seat_to_move, PHASE_DRAW, drawn=drawn)
        actions = legal_actions(situation)
        if not actions:
            return flow_result(state, turns)
        # 供「完全信息上界」这类**仅用于测量**的决策器读取真实局面；
        # 比赛用的决策器不得实现该方法（读对手手牌属违规）。
        observe = getattr(deciders[seat_to_move], "observe_state", None)
        if callable(observe):
            observe(state)
        chosen = _ask(deciders[seat_to_move], situation, actions)
        if chosen is None:
            chosen = next((a for a in actions if a.kind == DISCARD), None)
        if chosen is None:
            return flow_result(state, turns)

        if chosen.kind == HU:
            return settle_win(state, seat_to_move, drawn, base_score, turns)
        if chosen.kind == GANG:
            apply_gang(state, seat_to_move, chosen)
            need_draw = True
            notify()
            continue
        if chosen.kind != DISCARD:
            # 策略返回了非本阶段动作：不掩盖问题，直接结束本局
            return flow_result(state, turns)

        apply_discard(state, seat_to_move, chosen.tile, drawn)  # type: ignore[arg-type]
        notify()
        claim = resolve_responses(state, seat_to_move, chosen.tile, deciders)  # type: ignore[arg-type]
        if claim is None:
            seat_to_move = (seat_to_move + 1) % SEATS
            need_draw = True
            drawn = None
            continue
        seat_to_move, replaced = claim
        drawn = None
        notify()
        if replaced:
            drawn = _draw(state, seat_to_move)
            if drawn is None:
                return flow_result(state, turns)
            notify()
        need_draw = False

    return flow_result(state, turns)


def run_round(
    deciders: Sequence[object],
    *,
    dealer: int = 0,
    round_no: int = 1,
    base_score: int = 1,
    rng: random.Random | None = None,
    observer: object | None = None,
) -> RoundResult:
    """发牌并跑完一局，返回番型与四家净分。"""
    state = deal(rng or random.Random(), dealer, round_no)
    return play_round(
        state, deciders, base_score=base_score, observer=observer
    )


def settle_win(
    state: RoundState, seat: int, drawn: int | None, base_score: int, turns: int
) -> RoundResult:
    seat_state = state.seats[seat]
    waiting = list(seat_state.hand)
    if drawn is not None and waiting[drawn] > 0:
        waiting[drawn] -= 1
    result = compute_fan(
        waiting,
        drawn if drawn is not None else 0,
        len(seat_state.melds),
        chain_count=seat_state.chain_count,
        piao_count=seat_state.piao_count,
    )
    if not result.hu:
        return flow_result(state, turns)
    deltas = score_module.seat_deltas(
        result.fan, base_score, winner_seat=seat, dealer_seat=state.dealer
    )
    return RoundResult(
        round_no=state.round_no,
        dealer=state.dealer,
        winner=seat,
        fan=result.fan,
        detail=result.detail,
        scores=deltas,
        draws=INITIAL_WALL - len(state.wall),
        turns=turns,
        god_counts=tuple(seat_state.god_count for seat_state in state.seats),
    )


def flow_result(state: RoundState, turns: int) -> RoundResult:
    return RoundResult(
        round_no=state.round_no,
        dealer=state.dealer,
        winner=None,
        fan=0,
        detail=(),
        scores=(0, 0, 0, 0),
        draws=INITIAL_WALL - len(state.wall),
        turns=turns,
        god_counts=tuple(seat_state.god_count for seat_state in state.seats),
    )


__all__ = [
    "RoundResult",
    "RoundState",
    "Seat",
    "apply_chi",
    "apply_discard",
    "apply_gang",
    "apply_minggang",
    "apply_peng",
    "build_wall",
    "chi_options",
    "classify_discard",
    "deal",
    "flow_result",
    "play_round",
    "restricted",
    "run_round",
    "settle_win",
    "situation_for",
    "take_from_discards",
    "total_tiles_accounted",
]


def total_tiles_accounted(state: RoundState) -> int:
    """守恒校验：四家手牌 + 副露 + 弃牌 + 牌墙 应恒等于 136。"""
    total = len(state.wall)
    for seat_state in state.seats:
        total += tiles.total_tiles(seat_state.hand)
        total += sum(meld.size for meld in seat_state.melds)
        total += len(seat_state.discards)
    return total
