"""用独立实现的暴力算法对胡牌判定做差分验证。

参考实现把财神逐个指派为具体牌种（不检查该牌种剩余张数——百搭牌本身是实体牌，
只是充当别的牌），再用不含百搭的教科书式递归检查是否成胡。它与主实现在搜索结构上
完全独立，因此能有效暴露主实现漏解。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import combinations_with_replacement

from majiang.rules import tiles, win
from majiang.rules.tiles import GOD, TILE_KINDS

from .helpers import counts_of

MELD_SLOTS = 3
BATCH = 150


def _plain_regular(counts: list[int], sets_needed: int) -> bool:
    first = next((tile for tile, amount in enumerate(counts) if amount), None)
    if first is None:
        return sets_needed == 0
    if sets_needed == 0:
        return False
    if counts[first] >= 3:
        counts[first] -= 3
        solved = _plain_regular(counts, sets_needed - 1)
        counts[first] += 3
        if solved:
            return True
    if tiles.run_is_valid(first) and counts[first + 1] and counts[first + 2]:
        counts[first] -= 1
        counts[first + 1] -= 1
        counts[first + 2] -= 1
        solved = _plain_regular(counts, sets_needed - 1)
        counts[first] += 1
        counts[first + 1] += 1
        counts[first + 2] += 1
        if solved:
            return True
    return False


def _plain_winning(counts: list[int], meld_count: int) -> bool:
    sets_needed = 4 - meld_count
    work = list(counts)
    for tile in range(TILE_KINDS):
        if work[tile] >= 2:
            work[tile] -= 2
            solved = _plain_regular(work, sets_needed)
            work[tile] += 2
            if solved:
                return True
    return False


def _plain_seven_pairs(counts: list[int]) -> bool:
    if sum(counts) != 14:
        return False
    return all(amount % 2 == 0 for amount in counts) and sum(
        amount // 2 for amount in counts
    ) == 7


def reference_is_winning(counts: Sequence[int], meld_count: int = 0) -> bool:
    if sum(counts) != 14 - MELD_SLOTS * meld_count:
        return False
    wildcards = counts[GOD]
    base = list(counts)
    base[GOD] = 0
    targets = [tile for tile in range(TILE_KINDS) if tile != GOD]
    for combo in combinations_with_replacement(targets, wildcards):
        trial = list(base)
        for tile in combo:
            trial[tile] += 1
        if _plain_winning(trial, meld_count):
            return True
        if meld_count == 0 and _plain_seven_pairs(trial):
            return True
    return False


def _random_hand(size: int, wildcards: int, rng: random.Random) -> list[int]:
    pool = [tile for tile in range(TILE_KINDS) if tile != GOD for _ in range(tiles.COPIES_PER_KIND)]
    rng.shuffle(pool)
    counts = tiles.counts_from(pool[: size - wildcards])
    counts[GOD] = wildcards
    return counts


def _drawn_size(meld_count: int) -> int:
    return 14 - MELD_SLOTS * meld_count


def test_winning_shape_matches_reference_without_melds() -> None:
    rng = random.Random(20260923)
    for wildcards in (0, 1, 2):
        for _ in range(BATCH):
            counts = _random_hand(14, wildcards, rng)
            assert win.is_winning_shape(counts, 0) is reference_is_winning(counts, 0)


def test_winning_shape_matches_reference_with_three_wildcards() -> None:
    rng = random.Random(7788)
    for _ in range(40):
        counts = _random_hand(14, 3, rng)
        assert win.is_winning_shape(counts, 0) is reference_is_winning(counts, 0)


def test_winning_shape_matches_reference_with_melds() -> None:
    rng = random.Random(31337)
    for meld_count in (1, 2):
        for wildcards in (0, 1, 2):
            for _ in range(BATCH):
                counts = _random_hand(_drawn_size(meld_count), wildcards, rng)
                assert win.is_winning_shape(counts, meld_count) is reference_is_winning(
                    counts, meld_count
                )


def test_winning_draws_matches_reference() -> None:
    rng = random.Random(4242)
    for wildcards in (0, 1, 2):
        for _ in range(BATCH):
            counts = _random_hand(13, wildcards, rng)
            expected = []
            for tile in range(TILE_KINDS):
                if counts[tile] >= tiles.COPIES_PER_KIND:
                    continue
                trial = list(counts)
                trial[tile] += 1
                if reference_is_winning(trial, 0):
                    expected.append(tile)
            assert win.winning_draws(counts, 0) == tuple(expected)


def test_baotou_matches_reference() -> None:
    rng = random.Random(9001)
    for wildcards in (0, 1, 2):
        for _ in range(BATCH):
            counts = _random_hand(13, wildcards, rng)
            expected = all(
                reference_is_winning(
                    [amount + (1 if index == tile else 0) for index, amount in enumerate(counts)],
                    0,
                )
                for tile in range(TILE_KINDS)
                if counts[tile] < tiles.COPIES_PER_KIND
            )
            assert win.is_baotou(counts, 0) is expected


STRUCTURED_SPECS: tuple[str, ...] = (
    "1w1w2w2w3w3w4w4w5w5w6w6w7w7w",
    "1w1w1w1w2w2w2w2w3w3w3w3w4w4w",
    "白白白白1w1w2w2w3w3w4w4w5w5w",
    "1w2w3w4w5w6w7w8w9w1b2b3b5b5b",
    "1w1w1w1w2w2w2w2w3w3w4w4w5w5w",
    "1w2w3w4w5w6w7w8w9w白白白白东",
    "白白白1w1w1w2w2w2w3w3w3w5b5b",
    "1w1w1w2w2w2w3w3w3w4w4w4w白白",
    "1w1w1w1w2w2w2w2w3w3w3w3w白东",
)


def test_structured_hands_match_reference() -> None:
    for spec in STRUCTURED_SPECS:
        counts = counts_of(spec)
        assert win.is_winning_shape(counts, 0) is reference_is_winning(counts, 0)
        if counts[GOD] == 0:
            assert (win.seven_pairs_quads(counts) is not None) is _plain_seven_pairs(counts)
