"""暗手牌 + 副露构成的完整手牌。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from . import melds as melds_module
from . import tiles
from .melds import Meld
from .tiles import GOD, TILE_KINDS


class HandError(ValueError):
    """手牌结构非法。"""


@dataclass(frozen=True, slots=True)
class Hand:
    counts: tuple[int, ...]
    melds: tuple[Meld, ...] = ()

    def __post_init__(self) -> None:
        if len(self.counts) != TILE_KINDS:
            raise HandError(f"计数长度应为 {TILE_KINDS}，实际 {len(self.counts)}")
        if any(amount < 0 or amount > tiles.COPIES_PER_KIND for amount in self.counts):
            raise HandError(f"单牌种计数越界: {self.counts!r}")

    @classmethod
    def from_codes(cls, codes: Iterable[str], melds: Sequence[Meld] = ()) -> Hand:
        return cls.from_counts(tiles.counts_from(tiles.parse_all(codes)), melds)

    @classmethod
    def from_counts(cls, counts: Sequence[int], melds: Sequence[Meld] = ()) -> Hand:
        melds_module.validate_melds(melds)
        hand = cls(counts=tuple(counts), melds=tuple(melds))
        hand.validate()
        return hand

    @property
    def meld_count(self) -> int:
        return len(self.melds)

    @property
    def chi_count(self) -> int:
        return melds_module.chi_count(self.melds)

    @property
    def gang_count(self) -> int:
        return melds_module.gang_count(self.melds)

    @property
    def sets_needed(self) -> int:
        """暗手牌还需要的面子数。"""
        return tiles.SETS_PER_HAND - self.meld_count

    @property
    def waiting_size(self) -> int:
        """未摸牌时的暗手张数。"""
        return tiles.HAND_SIZE - tiles.MELD_SLOTS * self.meld_count

    @property
    def drawn_size(self) -> int:
        """摸牌后的暗手张数。"""
        return self.waiting_size + 1

    @property
    def tile_count(self) -> int:
        return sum(self.counts)

    @property
    def god_count(self) -> int:
        """暗手中留存的财神数（即平台的「手留白」）。"""
        return self.counts[GOD]

    @property
    def real_counts(self) -> list[int]:
        """剔出财神后的计数。"""
        out = list(self.counts)
        out[GOD] = 0
        return out

    @property
    def real_count(self) -> int:
        return self.tile_count - self.god_count

    @property
    def is_waiting_shape(self) -> bool:
        return self.tile_count == self.waiting_size

    @property
    def is_drawn_shape(self) -> bool:
        return self.tile_count == self.drawn_size

    def with_tile(self, tile: int) -> Hand:
        counts = list(self.counts)
        counts[tile] += 1
        return Hand(tuple(counts), self.melds)

    def without_tile(self, tile: int) -> Hand:
        counts = list(self.counts)
        if counts[tile] <= 0:
            raise HandError(f"手中没有 {tiles.to_code(tile)}")
        counts[tile] -= 1
        return Hand(tuple(counts), self.melds)

    def validate(self) -> None:
        melds_module.validate_melds(self.melds)
        for meld in self.melds:
            for tile, amount in enumerate(meld.counts()):
                if self.counts[tile] + amount > tiles.COPIES_PER_KIND:
                    raise HandError(
                        f"{tiles.to_code(tile)} 的手牌与副露合计超过 {tiles.COPIES_PER_KIND} 张"
                    )

    def to_codes(self) -> list[str]:
        return tiles.to_codes(tiles.tiles_of(self.counts))
