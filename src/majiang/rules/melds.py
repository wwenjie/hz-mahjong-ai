"""副露（吃 / 碰 / 杠）。

平台只明确了 ``kind == "chi"`` 这一取值（本人吃摊数按它统计）。其余取值按常见
命名做兼容映射；未识别的取值一律抛错，避免静默误判。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import tiles
from .tiles import GOD, TILE_KINDS

CHI = "chi"
PENG = "peng"
GANG = "gang"

_KIND_ALIASES: dict[str, str] = {
    "chi": CHI,
    "chow": CHI,
    "peng": PENG,
    "pong": PENG,
    "pon": PENG,
    "gang": GANG,
    "kong": GANG,
    "kan": GANG,
    # 实测（2026-09-23）：暗杠下发为 "gang_an"，因此按 "gang_<中文拼音缩写>"
    # 的命名约定补齐明杠与补杠。
    "gang_an": GANG,
    "gang_ming": GANG,
    "gang_bu": GANG,
    "angang": GANG,
    "minggang": GANG,
    "bugang": GANG,
    "jiagang": GANG,
    "an_gang": GANG,
    "ming_gang": GANG,
    "bu_gang": GANG,
}

_CONCEALED_RAW_KINDS = frozenset({"angang", "gang_an", "an_gang"})

CHI_MAX_PER_HAND = 2


class MeldError(ValueError):
    """副露结构非法。"""


@dataclass(frozen=True, slots=True)
class Meld:
    kind: str
    tiles: tuple[int, ...]
    concealed: bool = False

    @property
    def is_chi(self) -> bool:
        return self.kind == CHI

    @property
    def is_peng(self) -> bool:
        return self.kind == PENG

    @property
    def is_gang(self) -> bool:
        return self.kind == GANG

    @property
    def size(self) -> int:
        return len(self.tiles)

    def counts(self) -> list[int]:
        out = [0] * TILE_KINDS
        for tile in self.tiles:
            out[tile] += 1
        return out

    def validate(self) -> None:
        if self.kind not in (CHI, PENG, GANG):
            raise MeldError(f"未知副露类型: {self.kind!r}")
        if any(not 0 <= tile < TILE_KINDS for tile in self.tiles):
            raise MeldError(f"副露含非法牌索引: {self.tiles!r}")
        if any(tile == GOD for tile in self.tiles):
            raise MeldError("财神不能被吃、碰、杠")
        if self.kind == CHI:
            self._validate_chi()
        elif self.kind == PENG:
            self._validate_peng()
        else:
            self._validate_gang()

    def _validate_chi(self) -> None:
        if len(self.tiles) != tiles.RUN_LENGTH:
            raise MeldError(f"吃副露应为 {tiles.RUN_LENGTH} 张，实际 {len(self.tiles)}")
        ordered = sorted(self.tiles)
        if not tiles.run_is_valid(ordered[0]) or ordered[1] != ordered[0] + 1 or ordered[2] != ordered[0] + 2:
            raise MeldError(f"吃副露不是同一花色的顺子: {self.tiles!r}")

    def _validate_peng(self) -> None:
        if len(self.tiles) != tiles.SET_LENGTH or len(set(self.tiles)) != 1:
            raise MeldError(f"碰副露应为同一牌种的 {tiles.SET_LENGTH} 张: {self.tiles!r}")

    def _validate_gang(self) -> None:
        # 杠的编码在平台侧未确认：部分实现下发 4 张，部分只在补杠场景下发 3 张 + 已有碰。
        if len(self.tiles) not in (tiles.SET_LENGTH, tiles.COPIES_PER_KIND) or len(set(self.tiles)) != 1:
            raise MeldError(f"杠副露应为同一牌种的 3 或 4 张: {self.tiles!r}")


def parse_meld(raw: Mapping[str, Any]) -> Meld:
    raw_kind = str(raw.get("kind", "")).strip().lower()
    kind = _KIND_ALIASES.get(raw_kind)
    if kind is None:
        raise MeldError(f"未识别的副露类型: {raw_kind!r}")
    raw_tiles = raw.get("tiles") or ()
    if not isinstance(raw_tiles, Sequence) or isinstance(raw_tiles, (str, bytes)):
        raise MeldError(f"副露 tiles 应为牌码序列: {raw_tiles!r}")
    meld = Meld(
        kind=kind,
        tiles=tuple(tiles.parse(str(code)) for code in raw_tiles),
        concealed=raw_kind in _CONCEALED_RAW_KINDS,
    )
    meld.validate()
    return meld


def parse_melds(raw_melds: Sequence[Mapping[str, Any]]) -> tuple[Meld, ...]:
    return tuple(parse_meld(raw) for raw in raw_melds or ())


def meld_count(melds: Sequence[Meld]) -> int:
    return len(melds)


def chi_count(melds: Sequence[Meld]) -> int:
    return sum(1 for meld in melds if meld.is_chi)


def gang_count(melds: Sequence[Meld]) -> int:
    return sum(1 for meld in melds if meld.is_gang)


def validate_melds(melds: Sequence[Meld]) -> None:
    for meld in melds:
        meld.validate()
    if len(melds) > tiles.MAX_MELDS:
        raise MeldError(f"副露数超过 {tiles.MAX_MELDS} 组: {len(melds)}")
    if chi_count(melds) > CHI_MAX_PER_HAND:
        raise MeldError(f"吃副露最多 {CHI_MAX_PER_HAND} 摊，实际 {chi_count(melds)}")
