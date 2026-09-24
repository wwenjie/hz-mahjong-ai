"""平台牌码与内部整数表示之间的转换。

平台牌码：``1w``–``9w``（万）、``1b``–``9b``（筒）、``1t``–``9t``（条）、
``东南西北中发白``（字牌）。

内部用 0–33 的整数表示牌种：0–8 万、9–17 筒、18–26 条、27–33 字牌。
白板即财神（百搭），在内部始终按百搭处理，不参与自然面子与对子的组合。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

SUITS: tuple[str, ...] = ("w", "b", "t")
HONORS: tuple[str, ...] = ("东", "南", "西", "北", "中", "发", "白")

NUMBER_SUIT_SIZE = 9
NUMBER_KINDS = len(SUITS) * NUMBER_SUIT_SIZE
HONOR_KINDS = len(HONORS)
TILE_KINDS = NUMBER_KINDS + HONOR_KINDS
COPIES_PER_KIND = 4
WALL_SIZE = TILE_KINDS * COPIES_PER_KIND

HONOR_START = NUMBER_KINDS
GOD = HONOR_START + HONORS.index("白")

RUN_LENGTH = 3
SET_LENGTH = 3
HAND_SIZE = 13
MELD_SLOTS = 3
SETS_PER_HAND = 4
MAX_MELDS = SETS_PER_HAND


def _build_codes() -> tuple[str, ...]:
    codes: list[str] = []
    for suit in SUITS:
        for rank_value in range(1, NUMBER_SUIT_SIZE + 1):
            codes.append(f"{rank_value}{suit}")
    codes.extend(HONORS)
    return tuple(codes)


TILE_CODES: tuple[str, ...] = _build_codes()
_CODE_TO_TILE: dict[str, int] = {code: tile for tile, code in enumerate(TILE_CODES)}


class TileCodeError(ValueError):
    """牌码无法解析。"""


def parse(code: str) -> int:
    """把平台牌码解析为内部牌索引。"""
    if not isinstance(code, str) or not code:
        raise TileCodeError(f"非法牌码: {code!r}")
    key = code.strip()
    exact = _CODE_TO_TILE.get(key)
    if exact is not None:
        return exact
    if len(key) == 2:
        rank_text, suit_text = key[0], key[1].lower()
        if rank_text in "123456789" and suit_text in SUITS:
            return _CODE_TO_TILE[f"{rank_text}{suit_text}"]
    raise TileCodeError(f"非法牌码: {code!r}")


def to_code(tile: int) -> str:
    """把内部牌索引转回平台牌码。"""
    if not 0 <= tile < TILE_KINDS:
        raise TileCodeError(f"非法牌索引: {tile}")
    return TILE_CODES[tile]


def parse_all(codes: Iterable[str]) -> list[int]:
    return [parse(code) for code in codes]


def to_codes(tiles: Iterable[int]) -> list[str]:
    return [to_code(tile) for tile in tiles]


def counts_from(tiles: Iterable[int]) -> list[int]:
    counts = [0] * TILE_KINDS
    for tile in tiles:
        counts[tile] += 1
    return counts


def tiles_of(counts: Sequence[int]) -> list[int]:
    """把计数展开为牌索引列表。"""
    out: list[int] = []
    for tile, amount in enumerate(counts):
        out.extend([tile] * amount)
    return out


def remaining_copies(counts: Sequence[int]) -> list[int]:
    """给定已观测计数，返回各牌种在牌墙中的剩余张数。"""
    return [COPIES_PER_KIND - amount for amount in counts]


def is_god(tile: int) -> bool:
    return tile == GOD


def is_number(tile: int) -> bool:
    return 0 <= tile < NUMBER_KINDS


def is_honor(tile: int) -> bool:
    return NUMBER_KINDS <= tile < TILE_KINDS


def suit(tile: int) -> int:
    """数牌返回花色序号 0/1/2，字牌返回 -1。"""
    return tile // NUMBER_SUIT_SIZE if is_number(tile) else -1


def rank(tile: int) -> int:
    """数牌返回 1–9，字牌返回 1–7。"""
    if is_number(tile):
        return tile % NUMBER_SUIT_SIZE + 1
    return tile - HONOR_START + 1


def run_is_valid(start: int) -> bool:
    """``start, start+1, start+2`` 是否构成同一花色的顺子。"""
    return is_number(start) and rank(start) + RUN_LENGTH - 1 <= NUMBER_SUIT_SIZE


def total_tiles(counts: Sequence[int]) -> int:
    return sum(counts)
