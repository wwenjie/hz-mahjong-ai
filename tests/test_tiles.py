import pytest

from majiang.rules import hand as hand_module
from majiang.rules import melds as melds_module
from majiang.rules import tiles
from majiang.rules.hand import Hand
from majiang.rules.melds import Meld
from majiang.rules.tiles import GOD, TILE_KINDS, TileCodeError


def test_wall_composition() -> None:
    assert TILE_KINDS == 34
    assert tiles.WALL_SIZE == 136
    assert len(tiles.TILE_CODES) == TILE_KINDS
    assert len(set(tiles.TILE_CODES)) == TILE_KINDS


def test_god_is_white_dragon() -> None:
    assert tiles.to_code(GOD) == "白"
    assert tiles.parse("白") == GOD
    assert tiles.is_god(GOD)


@pytest.mark.parametrize("code", tiles.TILE_CODES)
def test_parse_and_to_code_roundtrip(code: str) -> None:
    assert tiles.to_code(tiles.parse(code)) == code


def test_parse_is_case_insensitive_on_suit() -> None:
    assert tiles.parse("5W") == tiles.parse("5w")
    assert tiles.parse("1B") == tiles.parse("1b")


@pytest.mark.parametrize("bad", ["", "0w", "10w", "w1", "东x", "白板", "1x"])
def test_parse_rejects_invalid_codes(bad: str) -> None:
    with pytest.raises(TileCodeError):
        tiles.parse(bad)


def test_suit_and_rank() -> None:
    assert tiles.suit(tiles.parse("1w")) == 0
    assert tiles.suit(tiles.parse("9b")) == 1
    assert tiles.suit(tiles.parse("5t")) == 2
    assert tiles.suit(tiles.parse("东")) == -1

    assert tiles.rank(tiles.parse("1w")) == 1
    assert tiles.rank(tiles.parse("9w")) == 9
    assert tiles.rank(tiles.parse("东")) == 1
    assert tiles.rank(tiles.parse("白")) == 7


def test_run_is_valid_respects_suit_boundaries() -> None:
    assert tiles.run_is_valid(tiles.parse("1w"))
    assert tiles.run_is_valid(tiles.parse("7w"))
    assert not tiles.run_is_valid(tiles.parse("8w"))
    assert not tiles.run_is_valid(tiles.parse("9w"))
    assert not tiles.run_is_valid(tiles.parse("东"))
    assert not tiles.run_is_valid(tiles.parse("中"))


def test_counts_and_tiles_roundtrip() -> None:
    codes = ["1w", "1w", "9t", "东", "白"]
    counts = tiles.counts_from(tiles.parse_all(codes))
    assert tiles.total_tiles(counts) == len(codes)
    assert tiles.to_codes(tiles.tiles_of(counts)) == ["1w", "1w", "9t", "东", "白"]


def test_remaining_copies() -> None:
    counts = tiles.counts_from(tiles.parse_all(["1w", "1w", "白"]))
    remaining = tiles.remaining_copies(counts)
    assert remaining[tiles.parse("1w")] == 2
    assert remaining[GOD] == 3
    assert remaining[tiles.parse("9t")] == 4


def test_chi_meld_validates_run() -> None:
    Meld(kind=melds_module.CHI, tiles=tuple(tiles.parse_all(["3w", "1w", "2w"]))).validate()
    with pytest.raises(melds_module.MeldError):
        Meld(kind=melds_module.CHI, tiles=tuple(tiles.parse_all(["1w", "2w", "4w"]))).validate()
    with pytest.raises(melds_module.MeldError):
        Meld(kind=melds_module.CHI, tiles=tuple(tiles.parse_all(["8w", "9w", "1b"]))).validate()


def test_peng_and_gang_validation() -> None:
    Meld(kind=melds_module.PENG, tiles=tuple(tiles.parse_all(["5t"] * 3))).validate()
    Meld(kind=melds_module.GANG, tiles=tuple(tiles.parse_all(["5t"] * 4))).validate()
    Meld(kind=melds_module.GANG, tiles=tuple(tiles.parse_all(["5t"] * 3))).validate()
    with pytest.raises(melds_module.MeldError):
        Meld(kind=melds_module.PENG, tiles=tuple(tiles.parse_all(["5t", "5t", "6t"]))).validate()


def test_god_cannot_be_melded() -> None:
    with pytest.raises(melds_module.MeldError):
        Meld(kind=melds_module.PENG, tiles=(GOD, GOD, GOD)).validate()


def test_parse_meld_rejects_unknown_kind() -> None:
    with pytest.raises(melds_module.MeldError):
        melds_module.parse_meld({"kind": "weird", "tiles": ["1w", "2w", "3w"]})


def test_parse_meld_maps_known_aliases() -> None:
    meld = melds_module.parse_meld({"kind": "GANG", "tiles": ["1w"] * 4})
    assert meld.kind == melds_module.GANG
    concealed = melds_module.parse_meld({"kind": "angang", "tiles": ["1w"] * 4})
    assert concealed.kind == melds_module.GANG and concealed.concealed


def test_chi_count_limit_enforced() -> None:
    chis = tuple(
        Meld(kind=melds_module.CHI, tiles=tuple(tiles.parse_all(codes)))
        for codes in (["1w", "2w", "3w"], ["4w", "5w", "6w"], ["7w", "8w", "9w"])
    )
    with pytest.raises(melds_module.MeldError):
        melds_module.validate_melds(chis)
    melds_module.validate_melds(chis[:2])


def test_hand_sizes_track_melds() -> None:
    plain = Hand.from_codes(tiles.to_codes(tiles.tiles_of([0] * 13)))
    assert plain.meld_count == 0
    assert plain.waiting_size == 13 and plain.drawn_size == 14

    with_peng = Hand.from_codes(
        ["1w"] * 2 + ["2w"] * 3 + ["3w"] * 3 + ["4w"] * 3,
        [Meld(kind=melds_module.PENG, tiles=tuple(tiles.parse_all(["9t"] * 3)))],
    )
    assert with_peng.meld_count == 1
    assert with_peng.waiting_size == 10 and with_peng.drawn_size == 11
    assert with_peng.sets_needed == 3


def test_hand_with_and_without_tile() -> None:
    base = Hand.from_codes(["1w", "2w", "3w"])
    assert base.with_tile(tiles.parse("1w")).counts[tiles.parse("1w")] == 2
    assert base.without_tile(tiles.parse("3w")).counts[tiles.parse("3w")] == 0
    with pytest.raises(hand_module.HandError):
        base.without_tile(tiles.parse("9t"))


def test_hand_rejects_more_than_four_copies_across_melds() -> None:
    with pytest.raises(hand_module.HandError):
        Hand.from_counts(
            tiles.counts_from(tiles.parse_all(["1w"] * 2)),
            [Meld(kind=melds_module.PENG, tiles=tuple(tiles.parse_all(["1w"] * 3)))],
        )


def test_hand_god_count_and_real_count() -> None:
    hand = Hand.from_codes(["白", "白", "1w", "2w"])
    assert hand.god_count == 2
    assert hand.real_count == 2
    assert hand.real_counts[GOD] == 0
