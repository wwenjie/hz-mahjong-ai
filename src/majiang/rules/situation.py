"""局面：一次决策所需的全部事实。

平台快照不含 ``allowed_actions``，因此动作合法性必须由客户端自行判定；本模块只负责
表达一份自洽、可校验的局面，具体规则在 :mod:`majiang.rules.action`。

本模块不感知任何平台字段名——快照到局面的映射属于协议层，见
:func:`majiang.client.snapshot.Snapshot.to_situation`。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .god import GodState
from .hand import Hand
from .melds import Meld
from .table import TableState

PHASE_DEAL = "deal"
PHASE_DRAW = "draw"
PHASE_RESPONSE_PENG = "response_peng"
PHASE_RESPONSE_CHI = "response_chi"
PHASE_SETTLED = "settled"
PHASE_FINISHED = "finished"

PHASES = (
    PHASE_DEAL,
    PHASE_DRAW,
    PHASE_RESPONSE_PENG,
    PHASE_RESPONSE_CHI,
    PHASE_SETTLED,
    PHASE_FINISHED,
)
RESPONSE_PHASES = (PHASE_RESPONSE_PENG, PHASE_RESPONSE_CHI)
ACTIONABLE_PHASES = (PHASE_DRAW, *RESPONSE_PHASES)

OBSERVER_SEAT = -1


class SituationError(ValueError):
    """局面非法。"""


@dataclass(frozen=True, slots=True)
class Situation:
    seat: int
    phase: str
    turn: int
    hand: Hand
    god: GodState
    table: TableState
    responding_seats: tuple[int, ...] = ()
    offered_tile: int | None = None
    drawn_tile: int | None = None
    discards: tuple[tuple[int, ...], ...] = ()
    melds: tuple[tuple[Meld, ...], ...] = ()
    hand_counts: tuple[int, ...] = ()
    window_deadline_ms: int | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise SituationError(f"未知阶段: {self.phase!r}")
        if self.seat < OBSERVER_SEAT:
            raise SituationError(f"非法座位: {self.seat}")
        if self.phase in RESPONSE_PHASES and self.offered_tile is None:
            raise SituationError(f"响应阶段 {self.phase} 必须提供被响应的牌")

    @property
    def is_observer(self) -> bool:
        return self.seat == OBSERVER_SEAT

    @property
    def is_my_turn(self) -> bool:
        return self.phase == PHASE_DRAW and self.turn == self.seat

    @property
    def in_response_window(self) -> bool:
        return self.phase in RESPONSE_PHASES and self.seat in self.responding_seats

    @property
    def is_restricted(self) -> bool:
        """本人是否受抓打圈限制。"""
        return not self.is_observer and self.god.restricts(self.seat)

    def melds_for(self, seat: int) -> tuple[Meld, ...]:
        """指定座位的副露（公开信息）。"""
        if not 0 <= seat < len(self.melds):
            return ()
        return self.melds[seat]

    @property
    def all_melds(self) -> tuple[Meld, ...]:
        return tuple(meld for group in self.melds for meld in group)

    @classmethod
    def from_parts(
        cls,
        *,
        seat: int,
        phase: str,
        turn: int,
        hand: Hand,
        god: GodState,
        table: TableState,
        responding_seats: Sequence[int] = (),
        offered_tile: int | None = None,
        drawn_tile: int | None = None,
        discards: Sequence[Sequence[int]] = (),
        melds: Sequence[Sequence[Meld]] = (),
        hand_counts: Sequence[int] = (),
        window_deadline_ms: int | None = None,
    ) -> Situation:
        """由已解析好的各部分组装局面。

        协议快照到本结构的映射属于协议层职责，见
        :func:`majiang.client.snapshot.Snapshot.to_situation`——规则引擎不感知平台字段名。
        """
        return cls(
            seat=seat,
            phase=phase,
            turn=turn,
            hand=hand,
            god=god,
            table=table,
            responding_seats=tuple(responding_seats),
            offered_tile=offered_tile,
            drawn_tile=drawn_tile,
            discards=tuple(tuple(seat_discards) for seat_discards in discards),
            melds=tuple(tuple(seat_melds) for seat_melds in melds),
            hand_counts=tuple(int(count) for count in hand_counts),
            window_deadline_ms=window_deadline_ms,
        )

    def hand_counts_for(self, seat: int) -> int:
        """指定座位的暗手张数（公开信息）；未知时返回 -1。"""
        if not 0 <= seat < len(self.hand_counts):
            return -1
        return self.hand_counts[seat]
