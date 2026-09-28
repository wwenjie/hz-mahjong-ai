"""对局快照模型与到规则引擎局面的桥接。

字段与取值以 2026-09-23 实测为准（指南 v34，测试房真实对局），几处与文档表述不同、
必须按实测处理的地方：

- ``my_hand`` 是**完整的**暗手牌（本人回合 14 张、其余 13 张），``drawn_tile`` 只是
  标注哪一张是刚摸的，**两者不可相加**。不适用时 ``drawn_tile`` 为空串。
- ``responding_seats`` 与 ``window_deadline_ms`` **只在** ``response_peng`` /
  ``response_chi`` 阶段出现。
- ``last_discard`` 是当前被响应的那张牌，即规则引擎的 ``offered_tile``。
- ``discards`` 直接给出四家弃牌列表，``wall_remaining`` 直接给出牌墙剩余张数，
  都不需要从事件流重建。
- 响应阶段里 ``turn`` 是**打出牌的那一家**，不是响应者。
- 事件流中 ``tile_drawn`` 仅在本家事件里带真实牌码，他家事件的 ``tile`` 恒为空串。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from majiang.rules import melds as melds_module
from majiang.rules import tiles
from majiang.rules.god import GodState
from majiang.rules.hand import Hand, HandError
from majiang.rules.melds import Meld
from majiang.rules.situation import (
    OBSERVER_SEAT,
    PHASE_DRAW,
    RESPONSE_PHASES,
    Situation,
    SituationError,
)
from majiang.rules.table import TableState

SEATS = 4


def optional_tile(value: Any) -> int | None:
    """把牌码解析为索引；空串与 ``None`` 表示不适用。"""
    if value is None or value == "":
        return None
    return value if isinstance(value, int) else tiles.parse(str(value))


# 平台用**点数为 0 的牌码**表示「本字段不适用」，我们此前只认空串与 None。
# 实测（`logs/a_*.jsonl`）：307 场里 125 场出现过 `TileCodeError: 非法牌码: '0w'`，
# 共 1795 次，**全部集中在开局 2.6 秒内**（每场对局的前几个快照，
# 那时还没有人打牌），中局出现 **0 次**。后果是那几个快照被整条丢弃、
# 每场开局多花约 0.6 秒重拉——**目前没丢动作，但若这个占位哪天出现在中局就会丢**。
PLACEHOLDER_CODES = frozenset({"0", "0w", "0b", "0t"})


def nullable_tile(value: Any) -> int | None:
    """同 :func:`optional_tile`，但把平台的「不适用」占位码也当作 ``None``。

    **只用于可空的单张字段**（`drawn_tile` / `last_discard`）。**不要用于暗手与弃牌**：
    若哪天 `my_hand` 里真的出现 `0w`（例如平台启用赤 5），把它当占位就会**静默丢掉一张真牌**，
    那比报错糟得多。暗手必须继续走严格的 :func:`tiles_of`。
    """
    if isinstance(value, str) and value.strip() in PLACEHOLDER_CODES:
        return None
    return optional_tile(value)


def tiles_of(raw: Sequence[Any] | None) -> tuple[int, ...]:
    return tuple(optional_tile(code) for code in (raw or ()))  # type: ignore[misc]


@dataclass(frozen=True, slots=True)
class Snapshot:
    game_id: str
    phase: str
    seat: int
    round_no: int = 1
    dealer: int = 0
    turn: int = OBSERVER_SEAT
    waited_seat: int = OBSERVER_SEAT
    wall_remaining: int = 0
    my_hand: tuple[int, ...] = ()
    hand_counts: tuple[int, ...] = ()
    melds: tuple[tuple[Meld, ...], ...] = ()
    discards: tuple[tuple[int, ...], ...] = ()
    scores: tuple[int, ...] = ()
    drawn_tile: int | None = None
    last_discard: int | None = None
    god: Mapping[str, Any] = field(default_factory=dict)
    responding_seats: tuple[int, ...] = ()
    window_deadline_ms: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_observer(self) -> bool:
        return self.seat < 0

    @property
    def my_melds(self) -> tuple[Meld, ...]:
        if not 0 <= self.seat < len(self.melds):
            return ()
        return self.melds[self.seat]

    @property
    def is_my_turn(self) -> bool:
        return self.phase == PHASE_DRAW and self.turn == self.seat

    @property
    def in_response_window(self) -> bool:
        return self.phase in RESPONSE_PHASES and self.seat in self.responding_seats

    @property
    def expects_action(self) -> bool:
        """本快照是否在等本人做决策。"""
        return not self.is_observer and (self.is_my_turn or self.in_response_window)

    def hand(self) -> Hand:
        """本人暗手牌 + 副露。``my_hand`` 已完整，不需要再叠加 ``drawn_tile``。"""
        hand = Hand.from_counts(tiles.counts_from(self.my_hand), self.my_melds)
        expected = {hand.waiting_size, hand.drawn_size}
        if expected and hand.tile_count not in expected:
            raise HandError(
                f"my_hand 张数 {hand.tile_count} 与 {hand.meld_count} 组副露不符"
                f"（应为 {hand.waiting_size} 或 {hand.drawn_size}）"
            )
        if self.hand_counts and not 0 <= self.seat < len(self.hand_counts):
            raise SituationError(f"hand_counts 缺少 seat={self.seat}")
        return hand

    def table_state(self) -> TableState:
        return TableState(
            wall_remaining=self.wall_remaining,
            dealer_seat=self.dealer,
            round_no=self.round_no,
        )

    def god_state(self, *, piao_count: int = 0, hand: Hand | None = None) -> GodState:
        resolved = hand if hand is not None else self.hand()
        return GodState.from_god_field(
            self.god, hand_gods=resolved.god_count, piao_count=piao_count
        )

    def to_situation(
        self,
        *,
        piao_count: int = 0,
        hand: Hand | None = None,
        table: TableState | None = None,
    ) -> Situation:
        resolved_hand = hand if hand is not None else self.hand()
        return Situation.from_parts(
            seat=self.seat,
            phase=self.phase,
            turn=self.turn,
            hand=resolved_hand,
            god=self.god_state(piao_count=piao_count, hand=resolved_hand),
            table=table if table is not None else self.table_state(),
            responding_seats=self.responding_seats,
            offered_tile=self.last_discard,
            drawn_tile=self.drawn_tile,
            discards=self.discards,
            melds=self.melds,
            hand_counts=self.hand_counts,
            window_deadline_ms=self.window_deadline_ms,
        )

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> Snapshot:
        raw_melds = raw.get("melds") or ()
        parsed_melds = tuple(
            melds_module.parse_melds(seat_melds or ()) for seat_melds in raw_melds
        )
        god_raw = raw.get("god")
        window = raw.get("window_deadline_ms")
        return cls(
            game_id=str(raw.get("game_id", "")),
            phase=str(raw.get("phase", "")),
            seat=int(raw.get("seat", OBSERVER_SEAT)),
            round_no=int(raw.get("round_no", 1) or 1),
            dealer=int(raw.get("dealer", 0) or 0),
            turn=int(raw.get("turn", OBSERVER_SEAT)),
            waited_seat=int(raw.get("waited_seat", OBSERVER_SEAT)),
            wall_remaining=int(raw.get("wall_remaining", 0) or 0),
            my_hand=tiles_of(raw.get("my_hand")),
            hand_counts=tuple(int(n) for n in raw.get("hand_counts") or ()),
            melds=parsed_melds,
            discards=tuple(tiles_of(seat_discards) for seat_discards in raw.get("discards") or ()),
            scores=tuple(int(v) for v in raw.get("scores") or ()),
            drawn_tile=nullable_tile(raw.get("drawn_tile")),
            last_discard=nullable_tile(raw.get("last_discard")),
            god=god_raw if isinstance(god_raw, Mapping) else {},
            responding_seats=tuple(int(s) for s in raw.get("responding_seats") or ()),
            window_deadline_ms=None if window in (None, "") else int(window),
            extra={
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "game_id",
                    "phase",
                    "seat",
                    "round_no",
                    "dealer",
                    "turn",
                    "waited_seat",
                    "wall_remaining",
                    "my_hand",
                    "hand_counts",
                    "melds",
                    "discards",
                    "scores",
                    "drawn_tile",
                    "last_discard",
                    "god",
                    "responding_seats",
                    "window_deadline_ms",
                }
            },
        )


__all__ = ["Snapshot", "optional_tile", "nullable_tile", "tiles_of"]
