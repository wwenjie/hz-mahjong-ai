"""从平台事件流重建逐时刻局面（tasks.md 6A.4）。

数据来源是 test/auto 房的免认证事件流：``blocks[].start_hands`` 给出四家起手手牌，
``blocks[].events`` 给出完整事件序列。实测事件类型与字段：

- ``tile_drawn``    ``{seat, tile}``
- ``tile_discarded````{seat, tile, data:{catch_play}}``
- ``peng``          ``{seat, tile}``
- ``chi``           ``{seat, tile, data:{tiles:[三张]}}``
- ``gang``          ``{seat, tile, data:{kind: ming|bu|an}}``——实测 1828 次 / 1100 局
- ``pass``          ``{seat}``——响应窗口放弃，**不改变局面**（实测占全部事件的 13%）
- ``timeout``       ``{seat, data:{kind}}``——服务端超时兜底，**不改变局面**
- ``round_ended`` / ``game_ended``  ``data`` 带官方 ``fan``/``detail``/``scores``

### 一份事件流是**一场多局**（实测 Rounds=8）

8 局各有自己的 ``start_hands`` / ``dealer``，边界只能由 ``blocks[].round_no`` 变化判定
（事件流里没有 ``round_started``）。**必须逐局重建**：把 8 局事件铺在同一局面上跑，
第 2 局起的起手牌全错（实测 34% 的出牌「手上没有这张牌」）。用 :func:`iter_rounds`。

### 重建目标不是还原真相，而是还原「当时可见的信息」

对观察者座位只暴露它自己的暗手，其余三家只暴露公开量（副露、弃牌、暗手张数），与线上
``sim.round.situation_for`` 的口径一致。这样从真实对局生成的训练样本与线上服务走**同一条
特征提取路径**，不会出现训练-服务偏差。

### 两个由实测决定的实现细节

1. **庄家的第 14 张牌已在 ``start_hands`` 里**（实测 dealer 那手 14 张、其余 13 张，且该局
   第一个事件是庄家的 ``tile_discarded``，没有 ``tile_drawn``）。这与我们模拟器 ``deal()``
   只发 13 张、首回合再摸的做法不同，**重建时不要补摸**，否则牌数会多一张。
2. **抓打圈只有打出财神者本人再打非财神牌才解除**（与 ``round.apply_discard`` 一致），
   因此必须跟踪 ``god_discarder``，否则 ``god.restricts()`` 会算错。

已知近似：``chain_count``/``piao_count`` 不做「财飘」与「非财飘」的区分（都需要爆头判定），
但这两个量不参与任何对手特征，只影响 ``GodState`` 里未被使用的字段。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

from majiang.rules import melds as melds_module
from majiang.rules import table as table_rules
from majiang.rules import tiles
from majiang.rules.god import NO_DISCARDER, GodState
from majiang.rules.hand import Hand
from majiang.rules.melds import Meld
from majiang.rules.situation import PHASE_DRAW, Situation
from majiang.rules.table import TableState

SEATS = 4
DISCARDED = "tile_discarded"
DRAWN = "tile_drawn"
PENG = "peng"
CHI = "chi"
GANG = "gang"
PASS = "pass"
TIMEOUT = "timeout"
ROUND_ENDED = "round_ended"
GAME_ENDED = "game_ended"


@dataclass
class ReplaySeat:
    hand: list[int] = field(default_factory=lambda: [0] * tiles.TILE_KINDS)
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)
    chain_count: int = 0
    piao_count: int = 0


@dataclass
class ReplayState:
    seats: list[ReplaySeat]
    dealer: int = 0
    round_no: int = 1
    turn: int = 0
    catch_play: bool = False
    god_discarder: int = NO_DISCARDER
    draws: int = 0
    last_discard: int | None = None
    last_discarder: int = NO_DISCARDER
    result: Mapping[str, object] | None = None
    final_scores: tuple[int, ...] = ()
    anomalies: Counter = field(default_factory=Counter)

    @property
    def wall_remaining(self) -> int:
        return max(0, table_rules.INITIAL_WALL - self.draws)

    @property
    def opened(self) -> bool:
        """是否已经摸过第一张牌。未摸牌前的牌墙是 84（含庄家尚未摸走的那张），
        而 ``TableState`` 只接受 ≤ ``INITIAL_WALL``，因此此时不能构造局面。"""
        return self.draws > 0

    def situation_for(
        self,
        observer: int,
        *,
        phase: str = PHASE_DRAW,
        offered: int | None = None,
        responding: Sequence[int] = (),
        drawn: int | None = None,
    ) -> Situation:
        """构造观察者视角的局面——只含它自己的暗手与全部公开信息。"""
        if not self.opened:
            raise ValueError(
                "尚未摸第一张牌：此时牌墙为 84，TableState 不接受（它按发牌 53 张定义）"
            )
        seat_state = self.seats[observer]
        hand = Hand.from_counts(seat_state.hand, seat_state.melds)
        god = GodState(
            hand_gods=hand.god_count,
            chain_count=seat_state.chain_count,
            piao_count=seat_state.piao_count,
            catch_play=self.catch_play,
            god_discarder_seat=self.god_discarder,
        )
        return Situation.from_parts(
            seat=observer,
            phase=phase,
            turn=self.turn,
            hand=hand,
            god=god,
            table=TableState(
                wall_remaining=self.wall_remaining,
                dealer_seat=self.dealer,
                round_no=self.round_no,
            ),
            responding_seats=tuple(responding),
            offered_tile=offered,
            drawn_tile=drawn,
            discards=tuple(tuple(state.discards) for state in self.seats),
            melds=tuple(tuple(state.melds) for state in self.seats),
            hand_counts=tuple(sum(state.hand) for state in self.seats),
        )


def _tile_of(raw: object) -> int | None:
    if raw in (None, ""):
        return None
    try:
        return tiles.parse(str(raw))
    except Exception:  # noqa: BLE001 —— 未知牌码视为缺失，不猜测
        return None


def _sorted_blocks(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    blocks = sorted(
        payload.get("blocks") or [], key=lambda block: block.get("seq_start", 0)  # type: ignore[union-attr]
    )
    if not blocks:
        raise ValueError("事件流没有 blocks")
    return blocks


@dataclass(frozen=True)
class RoundSpan:
    """一局（``round_no``）在整个事件流里占的片段。

    **这条边界必须显式处理**：``blocks`` 只是按 seq 区间分页，而**每块都自带一份
    ``start_hands`` / ``dealer`` / ``round_no``**（实测确认）。一场 8 局的自由匹配对局
    是一份事件流，若把 8 局的事件铺在同一个局面上跑，第 2 局起手牌就全错——实测表现为
    34% 的出牌「手上没有这张牌」。事件流里没有 ``round_started``，边界只能由 ``round_no``
    变化判定。
    """

    round_no: int
    dealer: int
    start_hands: tuple[object, ...]
    events: tuple[Mapping[str, object], ...]


def round_spans(payload: Mapping[str, object]) -> list[RoundSpan]:
    """把事件流按局切分，顺序与 seq 一致。"""
    spans: list[RoundSpan] = []
    for block in _sorted_blocks(payload):
        round_no = int(block.get("round_no", 1) or 1)
        if not spans or spans[-1].round_no != round_no:
            spans.append(
                RoundSpan(
                    round_no=round_no,
                    dealer=int(block.get("dealer", 0) or 0),
                    start_hands=tuple(block.get("start_hands") or ()),
                    events=(),
                )
            )
        spans[-1] = replace(spans[-1], events=(*spans[-1].events, *(block.get("events") or ())))
    return spans


def _round_result(payload: Mapping[str, object], round_no: int) -> Mapping[str, object] | None:
    for entry in payload.get("rounds") or ():  # type: ignore[union-attr]
        if int(entry.get("round_no", 0) or 0) == round_no:
            return entry  # type: ignore[return-value]
    return None


def state_for_span(
    payload: Mapping[str, object], span: RoundSpan
) -> ReplayState:
    """为某一局构造初始局面（该局起手手牌 + 庄家/局号）。"""
    seats: list[ReplaySeat] = []
    for hand_codes in span.start_hands:
        counts = tiles.counts_from(tiles.parse_all(list(hand_codes)))  # type: ignore[arg-type]
        seats.append(ReplaySeat(hand=list(counts)))
    while len(seats) < SEATS:
        seats.append(ReplaySeat())
    return ReplayState(
        seats=seats,
        dealer=span.dealer,
        round_no=span.round_no,
        turn=span.dealer,
        result=_round_result(payload, span.round_no),
    )


def iter_rounds(
    payload: Mapping[str, object],
) -> Iterator[tuple[ReplayState, list[Mapping[str, object]]]]:
    """按局产出 ``(该局的初始状态, 该局事件列表)``。

    状态对象**每局都是新的**（调用方按顺序自行 ``apply_event``），因此不存在跨局污染。
    """
    for span in round_spans(payload):
        yield state_for_span(payload, span), list(span.events)


def from_payload(payload: Mapping[str, object]) -> ReplayState:
    """由事件流构造**第一局**的初始局面。

    多局对局请改用 :func:`iter_rounds`——它按 ``round_no`` 逐局重建。
    """
    span = round_spans(payload)[0]
    return state_for_span(payload, span)


def _take_from_discards(state: ReplayState, discarder: int, tile: int) -> bool:
    """与 ``round.take_from_discards`` 一致：从末尾找第一张并移除。"""
    discards = state.seats[discarder].discards
    for index in range(len(discards) - 1, -1, -1):
        if discards[index] == tile:
            del discards[index]
            return True
    return False


def _apply_discard(state: ReplayState, seat: int, tile: int) -> None:
    seat_state = state.seats[seat]
    if seat_state.hand[tile] <= 0:
        state.anomalies["discard-without-tile"] += 1
    seat_state.hand[tile] = max(0, seat_state.hand[tile] - 1)
    seat_state.discards.append(tile)
    state.last_discard = tile
    state.last_discarder = seat
    if tile == tiles.GOD:
        seat_state.chain_count += 1
        seat_state.piao_count += 1
        state.catch_play = True
        state.god_discarder = seat
    else:
        seat_state.chain_count = 0
        seat_state.piao_count = 0
        # 只有打出财神者本人再打非财神牌才解除抓打圈
        if state.catch_play and state.god_discarder == seat:
            state.catch_play = False
            state.god_discarder = NO_DISCARDER


def _seat_of(raw: object) -> int:
    """解析座位号。

    **不能用 ``raw or -1``**：座位 0 是合法座位但 falsy，写成 ``event.get("seat", -1) or -1``
    会把 0 变成 -1，于是**座位 0 的每个事件都被当成非法而丢弃**，重建出来的手牌全错
    （表现为牌种超过 4 张、吃碰的牌在牌河里找不到）。这个坑踩过一次。
    """
    if raw is None:
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def apply_event(state: ReplayState, event: Mapping[str, object]) -> None:
    """把一个事件作用到状态上。未知事件只计数，不猜测其语义。"""
    kind = str(event.get("type", ""))
    seat = _seat_of(event.get("seat"))
    tile = _tile_of(event.get("tile"))
    data = event.get("data") or {}

    if kind == DRAWN and tile is not None and 0 <= seat < SEATS:
        state.seats[seat].hand[tile] += 1
        state.draws += 1
        state.turn = seat
    elif kind == DISCARDED and tile is not None and 0 <= seat < SEATS:
        _apply_discard(state, seat, tile)
    elif kind == PENG and tile is not None and 0 <= seat < SEATS:
        seat_state = state.seats[seat]
        if seat_state.hand[tile] < 2:
            state.anomalies["peng-without-pair"] += 1
        seat_state.hand[tile] = max(0, seat_state.hand[tile] - 2)
        seat_state.melds.append(Meld(kind=melds_module.PENG, tiles=(tile,) * 3))
        seat_state.chain_count = 0
        seat_state.piao_count = 0
        if not _take_from_discards(state, state.last_discarder, tile):
            state.anomalies["peng-discard-missing"] += 1
    elif kind == CHI and tile is not None and 0 <= seat < SEATS:
        seat_state = state.seats[seat]
        run = tuple(
            value
            for value in (_tile_of(code) for code in (data.get("tiles") or ()))  # type: ignore[union-attr]
            if value is not None
        )
        if not run:
            # 没有 data.tiles 就无法确定用哪两张手牌；宁可留异常也不编一个错副露
            state.anomalies["chi-without-tiles"] += 1
        else:
            # **``data.tiles`` 已包含被吃的那张**（实测 ["7b","8b","9b"] 配 tile="9b"），
            # 所以不能再把 tile 拼进去——否则副露里会多一张，牌种计数会超过 4 张。
            for used in run:
                if used == tile:
                    continue
                if seat_state.hand[used] <= 0:
                    state.anomalies["chi-without-tile"] += 1
                seat_state.hand[used] = max(0, seat_state.hand[used] - 1)
            seat_state.melds.append(
                Meld(kind=melds_module.CHI, tiles=tuple(sorted(run)))
            )
            seat_state.chain_count = 0
            seat_state.piao_count = 0
            if not _take_from_discards(state, state.last_discarder, tile):
                state.anomalies["chi-discard-missing"] += 1
    elif kind == GANG and tile is not None and 0 <= seat < SEATS:
        # 实测事件流的 gang 共 1828 次 / 1100 局（ming 947 / bu 532 / an 349），
        # 不是稀有事件——原先整支丢弃会让杠家的暗手多出 3~4 张、副露少一个。
        # 杠后补牌由随后的 ``tile_drawn`` 承担，这里只处理手牌与副露。
        gang_kind = str(data.get("kind", ""))  # type: ignore[union-attr]
        seat_state = state.seats[seat]
        if gang_kind == "an":
            seat_state.hand[tile] = max(0, seat_state.hand[tile] - tiles.COPIES_PER_KIND)
            seat_state.melds.append(Meld(kind=melds_module.GANG, tiles=(tile,) * 4, concealed=True))
        elif gang_kind == "bu":
            seat_state.hand[tile] = max(0, seat_state.hand[tile] - 1)
            for index, meld in enumerate(seat_state.melds):
                if meld.kind == melds_module.PENG and meld.tiles[0] == tile:
                    seat_state.melds[index] = Meld(kind=melds_module.GANG, tiles=(tile,) * 4)
                    break
            else:
                state.anomalies["bugang-without-peng"] += 1
        elif gang_kind == "ming":
            seat_state.hand[tile] = max(0, seat_state.hand[tile] - 3)
            seat_state.melds.append(Meld(kind=melds_module.GANG, tiles=(tile,) * 4))
            if not _take_from_discards(state, state.last_discarder, tile):
                state.anomalies["minggang-discard-missing"] += 1
        else:
            state.anomalies[f"gang-unknown-kind:{gang_kind}"] += 1
        seat_state.chain_count += 1
    elif kind == ROUND_ENDED:
        # 只有 round_ended 的 data 带官方 fan/detail/scores；**不能让 game_ended 覆盖它**
        # （game_ended 的 data 只有 final_scores，覆盖后 fan 就丢了）
        if data:
            state.result = data  # type: ignore[assignment]
    elif kind == GAME_ENDED:
        if data:
            state.final_scores = tuple(data.get("final_scores") or ())  # type: ignore[union-attr]
    elif kind == TIMEOUT or kind == PASS:
        # 服务端超时兜底与显式放弃：都不改变局面。pass 实测占全部事件 13%，
        # 若计入 unknown 会把真正的未知事件淹没。
        pass
    else:
        state.anomalies[f"unknown:{kind}"] += 1


def all_events(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    """按 ``seq`` 顺序拼出全部事件（``blocks`` 只是按 seq 区间分页）。"""
    blocks = sorted(
        payload.get("blocks") or [], key=lambda block: block.get("seq_start", 0)  # type: ignore[union-attr]
    )
    events: list[Mapping[str, object]] = []
    for block in blocks:
        events.extend(block.get("events") or [])  # type: ignore[union-attr]
    return events


def iter_before_each_event(
    payload: Mapping[str, object],
) -> Iterator[tuple[Mapping[str, object], ReplayState]]:
    """依次产出 ``(事件, 该事件发生**之前**的状态)``。

    状态是**复用的同一对象**，调用方必须在下一次迭代前把需要的东西取走（或自行拷贝）。
    这样设计是为了避免每个事件都深拷贝四家手牌。

    跨局时会切换到该局**全新**的状态对象——否则第 2 局的事件会铺在第 1 局的终局局面上。
    """
    for state, events in iter_rounds(payload):
        for event in events:
            yield event, state
            apply_event(state, event)


def conservation(state: ReplayState) -> dict[str, int]:
    """守恒校验：每种牌全场合计不得超过 4 张，总量不得超过 136。"""
    total = [0] * tiles.TILE_KINDS
    for seat_state in state.seats:
        for tile, amount in enumerate(seat_state.hand):
            total[tile] += amount
        for meld in seat_state.melds:
            for tile, amount in enumerate(meld.counts()):
                total[tile] += amount
        for tile in seat_state.discards:
            total[tile] += 1
    return {
        "over_copies": sum(1 for value in total if value > tiles.COPIES_PER_KIND),
        "held_and_discarded": sum(total),
        "wall_remaining": state.wall_remaining,
    }


__all__ = [
    "ReplaySeat",
    "ReplayState",
    "RoundSpan",
    "all_events",
    "apply_event",
    "conservation",
    "from_payload",
    "iter_before_each_event",
    "iter_rounds",
    "round_spans",
    "state_for_span",
]
