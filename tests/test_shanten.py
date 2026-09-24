import random

import pytest

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles, win
from majiang.rules.shanten import shanten

from .helpers import counts_of


def ready(counts: list[int]) -> bool:
    return len(win.winning_draws(counts, 0)) > 0


def one_swap_reaches_ready(counts: list[int]) -> bool:
    """是否存在一次「打出一张、摸进一张」就进入听牌态。"""
    hand = list(counts)
    for discard_tile, amount in enumerate(hand):
        if amount == 0:
            continue
        hand[discard_tile] -= 1
        for draw_tile in range(tiles.TILE_KINDS):
            if hand[draw_tile] >= tiles.COPIES_PER_KIND:
                continue
            hand[draw_tile] += 1
            found = ready(hand)
            hand[draw_tile] -= 1
            if found:
                hand[discard_tile] += 1
                return True
        hand[discard_tile] += 1
    return False


def random_hand(size: int, wildcards: int, rng: random.Random) -> list[int]:
    pool = [t for t in range(tiles.TILE_KINDS) if t != tiles.GOD for _ in range(tiles.COPIES_PER_KIND)]
    rng.shuffle(pool)
    counts = tiles.counts_from(pool[: size - wildcards])
    counts[tiles.GOD] = wildcards
    return counts


def test_ready_hands_are_zero_shanten() -> None:
    assert shanten(counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b")) == 0
    assert shanten(counts_of("1w2w3w4w5w6w7w8w9w1b2b3b白")) == 0


def test_isolated_tiles_need_many_swaps() -> None:
    hand = counts_of("1w4w7w1b4b7b1t4t7t东南西北")
    assert win.winning_draws(hand, 0) == ()
    # 一般形为 8，但七对形在此手上只需 6 次换牌，故取较小者
    assert shanten(hand) == shanten_module.SEVEN_PAIRS_MAX_PAIRS


def test_seven_pairs_shapes() -> None:
    assert shanten(counts_of("1w1w2w2w3w3w4w4w5w5w6w6w7w")) == 0
    assert shanten(counts_of("1w1w2w2w3w3w4w4w5w5w6w6w白")) == 0
    assert shanten(counts_of("1w1w1w1w2w2w2w2w3w3w3w3w白")) == 0


def test_one_shanten_hand_is_not_ready() -> None:
    hand = counts_of("1w2w3w4w5w6w7w8w9w5b5b东中")
    assert not ready(hand)
    assert shanten(hand) == 1


def test_melds_reduce_shanten() -> None:
    assert shanten(counts_of("1w2w3w4w5w6w7w8w9w5b"), 1) == 0
    assert shanten(counts_of("1w2w3w4w5w6w5b5b6b6b"), 1) == 0
    assert shanten(counts_of("1w2w3w4w5w6w5b5b东中"), 1) == 1


@pytest.mark.parametrize("meld_count", [0, 1, 2, 3])
def test_shanten_is_bounded(meld_count: int) -> None:
    size = tiles.HAND_SIZE - 3 * meld_count
    rng = random.Random(1000 + meld_count)
    for wildcards in (0, 1, 2):
        for _ in range(40):
            hand = random_hand(size, wildcards, rng)
            value = shanten(hand, meld_count)
            assert 0 <= value <= shanten_module.SHANTEN_MAX


@pytest.mark.parametrize("meld_count", [0, 1, 2])
def test_zero_shanten_is_equivalent_to_ready(meld_count: int) -> None:
    size = tiles.HAND_SIZE - 3 * meld_count
    rng = random.Random(2024 + meld_count)
    checked = 0
    for wildcards in (0, 1, 2):
        for _ in range(150):
            hand = random_hand(size, wildcards, rng)
            is_ready = len(win.winning_draws(hand, meld_count)) > 0
            assert (shanten(hand, meld_count) == 0) is is_ready
            checked += 1
    assert checked == 450


def test_one_shanten_hands_have_a_single_swap_path() -> None:
    rng = random.Random(5150)
    found = 0
    for wildcards in (0, 1, 2):
        for _ in range(200):
            hand = random_hand(13, wildcards, rng)
            if shanten(hand) != 1:
                continue
            found += 1
            assert not ready(hand)
            assert one_swap_reaches_ready(hand)
    assert found > 0


def test_two_or_more_shanten_has_no_single_swap_path() -> None:
    rng = random.Random(8080)
    checked = 0
    for wildcards in (0, 1, 2):
        for _ in range(400):
            hand = random_hand(13, wildcards, rng)
            if shanten(hand) < 2:
                continue
            assert not one_swap_reaches_ready(hand)
            checked += 1
            if checked >= 8:
                break
        if checked >= 8:
            break
    assert checked == 8


def test_helps_from_a_single_god() -> None:
    without_god = counts_of("1w2w3w4w5w6w7w8w9w5b5b东中")
    with_god = counts_of("1w2w3w4w5w6w7w8w9w5b5b东白")
    assert shanten(with_god) < shanten(without_god)


def test_shanten_rejects_wrong_size() -> None:
    with pytest.raises(shanten_module.ShantenError):
        shanten(counts_of("1w2w3w"))
    with pytest.raises(shanten_module.ShantenError):
        shanten(counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b"), 5)


def test_helpful_tiles_reduce_shanten() -> None:
    hand = counts_of("1w2w3w4w5w6w7w8w9w5b5b东中")
    improved = counts_of("1w2w3w4w5w6w7w8w9w5b5b东东")
    assert shanten(improved) < shanten(hand)
    assert shanten(counts_of("1w2w3w4w5w6w7w8w9w5b5b6b7b")) < shanten(hand)
