import pytest

from majiang.rules import tiles
from majiang.rules import win
from majiang.rules.tiles import TILE_KINDS

from .helpers import counts_of


def test_plain_win_without_god() -> None:
    counts = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b5b")
    assert win.is_winning_shape(counts) is True
    branch = win.best_branch(counts)
    assert branch is not None and branch.name == "平胡" and branch.fan == 1


def test_triplets_win() -> None:
    counts = counts_of("1w1w1w9b9b9b东东东中中中白白")
    assert win.is_winning_shape(counts) is True
    assert win.best_branch(counts).name == "平胡"  # type: ignore[union-attr]


def test_missing_pair_is_not_a_win() -> None:
    counts = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b6b")
    assert win.is_winning_shape(counts) is False
    assert win.best_branch(counts) is None


def test_size_mismatch_is_not_a_win() -> None:
    assert win.is_winning_shape(counts_of("1w2w3w4w5w6w7w8w9w1b2b3b"), 0) is False
    assert win.is_winning_shape(counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b5b"), 1) is False


def test_meld_based_win() -> None:
    concealed = counts_of("1w2w3w4w5w6w7w8w9w5b5b")
    assert win.is_winning_shape(concealed, 1) is True
    assert win.is_winning_shape(concealed, 0) is False


def test_seven_pairs() -> None:
    counts = counts_of("1w1w2w2w3w3w4w4w5w5w6w6w7w7w")
    assert win.seven_pairs_quads(counts) == 0
    branch = win.best_branch(counts)
    assert branch is not None
    assert branch.seven_pairs and branch.name == "七对" and branch.fan == 2


@pytest.mark.parametrize(
    ("spec", "quads", "name", "fan"),
    [
        ("1w1w1w1w2w2w3w3w4w4w5w5w6w6w", 1, "豪华七对×1", 4),
        ("1w1w1w1w2w2w2w2w3w3w4w4w5w5w", 2, "豪华七对×2", 8),
        ("1w1w1w1w2w2w2w2w3w3w3w3w4w4w", 3, "豪华七对×3", 16),
    ],
)
def test_luxury_seven_pairs(spec: str, quads: int, name: str, fan: int) -> None:
    counts = counts_of(spec)
    assert win.seven_pairs_quads(counts) == quads
    branch = win.best_branch(counts)
    assert branch is not None and branch.name == name and branch.fan == fan


def test_four_gods_count_as_one_quad_when_rest_are_natural_pairs() -> None:
    counts = counts_of("白白白白1w1w2w2w3w3w4w4w5w5w")
    assert win.seven_pairs_quads(counts) == 1
    branch = win.best_branch(counts)
    assert branch is not None and branch.name == "豪华七对×1" and branch.fan == 4


def test_seven_pairs_not_available_with_melds() -> None:
    counts = counts_of("1w1w2w2w3w3w4w4w5w5w6w6w7w7w")
    assert win.best_branch(counts, 0).seven_pairs is True  # type: ignore[union-attr]
    assert win.best_branch(counts, 1) is None


def test_regular_shape_preferred_over_weaker_seven_pairs() -> None:
    counts = counts_of("1w1w2w2w3w3w4w4w5w5w6w6w7w7w")
    branch = win.best_branch(counts)
    assert branch is not None and branch.seven_pairs is True and branch.name == "七对"


def test_winning_draws_single_wait() -> None:
    counts = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b")
    assert win.winning_draws(counts) == (tiles.parse("5b"), tiles.parse("白"))
    assert win.is_baotou(counts) is False


def test_god_completes_pair_and_yields_baotou() -> None:
    counts = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b白")
    assert win.is_baotou(counts) is True
    assert len(win.winning_draws(counts)) == TILE_KINDS


def test_six_pairs_plus_god_is_baotou() -> None:
    waiting = counts_of("1w1w2w2w3w3w4w4w5w5w6w6w白")
    assert win.is_baotou(waiting) is True
    branch = win.best_branch(counts_of("1w1w2w2w3w3w4w4w5w5w6w6w白9t"), 0)
    assert branch is not None and branch.name == "七对" and branch.fan == 2


def test_god_completes_run() -> None:
    counts = counts_of("1w2w白4w5w6w7w8w9w1b1b东东东")
    assert win.is_winning_shape(counts) is True
    assert win.best_branch(counts).name == "平胡"  # type: ignore[union-attr]


def test_four_gods_hand_is_baotou_and_wins() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w白白白白")
    assert win.is_baotou(waiting) is True
    assert win.is_winning_shape(counts_of("1w2w3w4w5w6w7w8w9w白白白白东"), 0) is True


def test_winning_draws_skips_exhausted_kinds() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w白白白白")
    draws = win.winning_draws(waiting)
    assert tiles.GOD not in draws
    assert len(draws) == TILE_KINDS - 1


def test_winning_draws_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        win.winning_draws(counts_of("1w2w3w"))
