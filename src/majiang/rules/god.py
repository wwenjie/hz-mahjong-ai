"""财神（白板）状态与抓打圈语义。

抓打圈：某家打出财神后，其余玩家在该圈内不能吃、碰、明杠，且出牌只能打刚摸到的
牌；打出财神者本人不受此限。圈以新的打财神者重启，打其他牌则链断且圈解除。

状态转换写成纯函数，由调用方在确定的事件时机触发（打牌/杠/飘各自触发哪个转换是
平台语义，本模块只负责状态本身）。

两处文档未明说、按当前理解处理并在 tasks.md 2.14 待实测确认：

1. **「不能明杠」是否包含补杠**：补杠不消耗他人打出的牌，机制上更接近暗杠，
   因此本模块把动作层面的判断留给调用方，这里只暴露 `restricts` / `is_exempt`。
2. **圈解除的时机**：文档只说「打其他牌 = 链断且圈解除」，未说明是打财神者的下一张
   即解除，还是整圈后才解除。`after_break` 是显式的，何时调用由调用方决定。
3. **飘次数无法从快照推导**：快照只给 `chain_count`（飘与杠累计），因此 `piao_count`
   必须由运行时自行累计后传入。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import tiles
from .fan import CHAIN_MAX, FOUR_GODS_TOTAL

SEATS = 4
NO_DISCARDER = -1


class GodStateError(ValueError):
    """财神状态非法。"""


@dataclass(frozen=True, slots=True)
class GodState:
    hand_gods: int = 0
    chain_count: int = 0
    piao_count: int = 0
    baotou: bool = False
    catch_play: bool = False
    god_discarder_seat: int = NO_DISCARDER

    def __post_init__(self) -> None:
        if not 0 <= self.hand_gods <= tiles.COPIES_PER_KIND:
            raise GodStateError(f"手留白越界: {self.hand_gods}")
        if not 0 <= self.chain_count <= CHAIN_MAX:
            raise GodStateError(f"动作链次数应在 0–{CHAIN_MAX}: {self.chain_count}")
        if not 0 <= self.piao_count <= self.chain_count:
            raise GodStateError(f"飘次数不能超过动作链次数: piao={self.piao_count} chain={self.chain_count}")
        if self.hand_gods + self.piao_count > FOUR_GODS_TOTAL:
            raise GodStateError(
                f"手留白 {self.hand_gods} 加飘出 {self.piao_count} 超过全场财神总数 {FOUR_GODS_TOTAL}"
            )
        if not self.catch_play and self.god_discarder_seat != NO_DISCARDER:
            raise GodStateError("无抓打圈时不应存在打财神者")
        if self.catch_play and not 0 <= self.god_discarder_seat < SEATS:
            raise GodStateError(f"抓打圈缺少有效的打财神者座位: {self.god_discarder_seat}")

    @property
    def gang_count(self) -> int:
        return self.chain_count - self.piao_count

    @classmethod
    def from_god_field(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        hand_gods: int = 0,
        piao_count: int = 0,
    ) -> GodState:
        """解析快照的 ``god`` 字段。

        快照只提供 ``baotou`` / ``chain_count`` / ``catch_play`` / ``god_discarder_seat``；
        ``piao_count`` 需由运行时自行累计。
        """
        source = raw or {}
        return cls(
            hand_gods=hand_gods,
            chain_count=int(source.get("chain_count") or 0),
            piao_count=piao_count,
            baotou=bool(source.get("baotou")),
            catch_play=bool(source.get("catch_play")),
            god_discarder_seat=int(source.get("god_discarder_seat", NO_DISCARDER)),
        )

    def restricts(self, seat: int) -> bool:
        """该座位是否受抓打圈限制。"""
        return self.catch_play and self.god_discarder_seat != seat

    def is_exempt(self, seat: int) -> bool:
        """该座位是否为抓打圈豁免方（即打出财神者本人）。"""
        return self.catch_play and self.god_discarder_seat == seat

    def with_hand_gods(self, hand_gods: int) -> GodState:
        return GodState(
            hand_gods=hand_gods,
            chain_count=self.chain_count,
            piao_count=self.piao_count,
            baotou=self.baotou,
            catch_play=self.catch_play,
            god_discarder_seat=self.god_discarder_seat,
        )

    def with_baotou(self, baotou: bool) -> GodState:
        return GodState(
            hand_gods=self.hand_gods,
            chain_count=self.chain_count,
            piao_count=self.piao_count,
            baotou=baotou,
            catch_play=self.catch_play,
            god_discarder_seat=self.god_discarder_seat,
        )

    def after_gang(self) -> GodState:
        """杠：动作链 +1，不影响抓打圈。"""
        return GodState(
            hand_gods=self.hand_gods,
            chain_count=self.chain_count + 1,
            piao_count=self.piao_count,
            baotou=self.baotou,
            catch_play=self.catch_play,
            god_discarder_seat=self.god_discarder_seat,
        )

    def after_piao(self, discarder_seat: int) -> GodState:
        """爆头态弃胡打财神（财飘）：动作链 +1、飘 +1，并以本人重启抓打圈。"""
        return GodState(
            hand_gods=max(0, self.hand_gods - 1),
            chain_count=self.chain_count + 1,
            piao_count=self.piao_count + 1,
            baotou=self.baotou,
            catch_play=True,
            god_discarder_seat=discarder_seat,
        )

    def after_god_discard_without_piao(self, discarder_seat: int) -> GodState:
        """非爆头态打出财神：链断，但抓打圈照常触发并以本人重启。"""
        return GodState(
            hand_gods=max(0, self.hand_gods - 1),
            chain_count=0,
            piao_count=0,
            baotou=self.baotou,
            catch_play=True,
            god_discarder_seat=discarder_seat,
        )

    def after_break(self) -> GodState:
        """打出非财神且非飘非杠的牌：链断且抓打圈解除。"""
        return GodState(
            hand_gods=self.hand_gods,
            chain_count=0,
            piao_count=0,
            baotou=self.baotou,
            catch_play=False,
            god_discarder_seat=NO_DISCARDER,
        )
