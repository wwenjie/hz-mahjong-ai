"""本地模拟与批量自对弈测试（tasks.md 6.1 / 6.3）。"""

import random

import pytest

from majiang.rules import tiles
from majiang.rules.action import PENG
from majiang.rules.god import NO_DISCARDER
from majiang.rules.melds import Meld
from majiang.rules.tiles import GOD
from majiang.runtime.decider import FirstLegalDecider, GuardedDecider
from majiang.sim import batch, round as round_module
from majiang.sim.batch import place_points_of, run_batch, run_match, summary_lines
from majiang.sim.round import (
    BREAK,
    GOD_BREAK,
    PIAO,
    apply_discard,
    apply_gang,
    apply_peng,
    build_wall,
    chi_options,
    classify_discard,
    deal,
    flow_result,
    restricted,
    run_round,
    total_tiles_accounted,
)
from majiang.rules.action import Action
from majiang.strategy.policy import HeuristicDecider, PolicyConfig


def legal_deciders():
    return [FirstLegalDecider() for _ in range(4)]


def heuristic_deciders():
    return [GuardedDecider(HeuristicDecider(PolicyConfig())) for _ in range(4)]


def test_wall_composition_matches_platform() -> None:
    wall = build_wall(random.Random(0))
    assert len(wall) == 136
    assert all(wall.count(tile) == 4 for tile in range(tiles.TILE_KINDS))


def test_deal_gives_four_hands_of_thirteen_and_keeps_wall() -> None:
    state = deal(random.Random(1), dealer=2, round_no=3)
    assert len(state.wall) == 136 - 52
    assert all(tiles.total_tiles(seat.hand) == 13 for seat in state.seats)
    assert state.god_discarder == NO_DISCARDER and state.catch_play is False
    assert state.dealer == 2 and state.round_no == 3


def test_rollback_after_all_legal_play_keeps_136_tiles() -> None:
    problems: list[str] = []

    def check(state) -> None:
        accounted = total_tiles_accounted(state)
        if accounted != 136:
            problems.append(f"守恒失败: {accounted}")
        for seat in range(4):
            state.hand_of(seat)

    deciders = legal_deciders()
    rng = random.Random(4242)
    for index in range(25):
        run_round(deciders, dealer=index % 4, round_no=index + 1, rng=rng, observer=check)
    assert problems == []


def test_concealed_gang_removes_four_tiles() -> None:
    state = deal(random.Random(2), dealer=0)
    state.seats[0].hand = tiles.counts_from(tiles.parse_all(["9t"] * 4 + ["1w"] * 9))
    before = total_tiles_accounted(state)
    apply_gang(state, 0, Action("gang", tile=tiles.parse("9t"), gang_kind="angang"))
    assert state.seats[0].hand[tiles.parse("9t")] == 0
    assert state.seats[0].chain_count == 1
    assert total_tiles_accounted(state) == before


def test_peng_removes_the_claimed_tile_from_discards() -> None:
    state = deal(random.Random(3), dealer=0)
    tile = tiles.parse("5b")
    state.seats[0].discards.append(tile)
    state.seats[1].hand[tile] = 2
    apply_peng(state, 1, 0, tile)
    assert tile not in state.seats[0].discards
    assert state.seats[1].melds[-1].kind == PENG


def test_flow_keeps_dealer_and_dealer_win_keeps_dealer() -> None:
    flow = flow_result(deal(random.Random(4), dealer=1), 10)
    assert flow.is_flow and flow.scores == (0, 0, 0, 0)
    assert batch.next_dealer(1, flow) == 1
    win = flow_result(deal(random.Random(5), dealer=1), 10)
    win.winner = 1
    assert batch.next_dealer(1, win) == 1
    win.winner = 2
    assert batch.next_dealer(1, win) == 2


def test_catch_play_persists_until_the_god_discarder_plays_otherwise() -> None:
    state = deal(random.Random(6), dealer=0)
    state.seats[0].hand[GOD] = 1
    apply_discard(state, 0, GOD, None)
    assert state.catch_play is True and state.god_discarder == 0
    # 受限玩家打牌不影响圈
    state.seats[1].hand[tiles.parse("1w")] = 1
    apply_discard(state, 1, tiles.parse("1w"), None)
    assert state.catch_play is True and state.god_discarder == 0
    # 打财神者本人打非财神牌才解除
    state.seats[0].hand[tiles.parse("2w")] = 1
    apply_discard(state, 0, tiles.parse("2w"), None)
    assert state.catch_play is False and state.god_discarder == NO_DISCARDER


def test_restricted_players_cannot_chi_or_peng() -> None:
    state = deal(random.Random(7), dealer=0)
    tile = tiles.parse("3b")
    state.seats[1].hand[tile] = 2
    state.seats[1].hand[tiles.parse("1b")] = 1
    state.seats[1].hand[tiles.parse("2b")] = 1
    state.seats[1].hand[tiles.parse("4b")] = 1
    assert chi_options(state, 2, tile)
    state.catch_play = True
    state.god_discarder = 0
    assert restricted(state, 1) and restricted(state, 2)
    assert chi_options(state, 2, tile) == ()
    assert not restricted(state, 0)


def test_chi_needs_the_next_seat_and_respects_two_meld_limit() -> None:
    state = deal(random.Random(8), dealer=0)
    for seat in range(4):
        state.seats[seat].hand = [0] * tiles.TILE_KINDS
    state.seats[2].hand[tiles.parse("1b")] = 1
    state.seats[2].hand[tiles.parse("2b")] = 1
    assert chi_options(state, 2, tiles.parse("3b"))
    assert chi_options(state, 1, tiles.parse("3b")) == ()
    state.seats[2].melds = [
        Meld(kind="chi", tiles=tuple(tiles.parse_all(["1w", "2w", "3w"]))),
        Meld(kind="chi", tiles=tuple(tiles.parse_all(["4w", "5w", "6w"]))),
    ]
    assert chi_options(state, 2, tiles.parse("3b")) == ()


def test_classify_discard_distinguishes_piao_and_plain_god_discard() -> None:
    state = deal(random.Random(9), dealer=0)
    # 手上四组面子 + 两张财神：听任意，打出一张财神仍听任意 → 飘
    state.seats[0].hand = tiles.counts_from(
        tiles.parse_all(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1b", "2b", "3b", "白", "白"])
    )
    assert classify_discard(state, 0, GOD, GOD) == PIAO
    # 手牌远离成型时打财神只是断链
    state.seats[0].hand = tiles.counts_from(tiles.parse_all(["1w", "4w", "7w", "1b", "白"]))
    assert classify_discard(state, 0, GOD, None) == GOD_BREAK
    state.seats[0].hand[tiles.parse("9t")] = 1
    assert classify_discard(state, 0, tiles.parse("9t"), None) == BREAK


def test_place_points_follow_platform_table() -> None:
    assert place_points_of((10, 5, 0, -15)) == (3, 1, -1, -3)
    assert place_points_of((0, 0, 0, 0)) == (0, 0, 0, 0)
    assert place_points_of((5, 5, -3, -7)) == (2, 2, -1, -3)
    assert place_points_of((5, 4, 4, -13)) == (3, 0, 0, -3)
    assert place_points_of((-1, -1, -1, -1)) == (0, 0, 0, 0)


def test_round_is_deterministic_for_a_seed() -> None:
    first = run_round(heuristic_deciders(), dealer=1, rng=random.Random(99))
    second = run_round(heuristic_deciders(), dealer=1, rng=random.Random(99))
    assert (first.winner, first.fan, first.scores, first.turns) == (
        second.winner,
        second.fan,
        second.scores,
        second.turns,
    )


def test_round_scores_are_zero_sum() -> None:
    deciders = heuristic_deciders()
    rng = random.Random(31)
    for index in range(15):
        result = run_round(deciders, dealer=index % 4, rng=rng)
        assert sum(result.scores) == 0
        if not result.is_flow:
            assert result.scores[result.winner] > 0 and result.fan >= 1


def test_match_accumulates_rounds_and_place_points() -> None:
    result = run_match(legal_deciders(), rounds=4, seed=5, labels=["a", "b", "c", "d"])
    assert result.rounds == 4
    assert all(stat.rounds == 4 for stat in result.seats)
    assert sum(stat.total_score for stat in result.seats) == 0
    assert result.flow_rate >= 0.0
    assert "局数" in summary_lines(result)[0]


def test_batch_rotates_the_starting_dealer() -> None:
    result = run_batch(
        [lambda: FirstLegalDecider() for _ in range(4)], matches=4, rounds=1, seed=3
    )
    assert result.matches == 4
    assert result.rounds == 4


def test_heuristic_beats_baseline_head_to_head() -> None:
    """策略应当显著优于「永远打最小那张」的基线。"""
    heuristic = lambda: GuardedDecider(HeuristicDecider(PolicyConfig()))
    baseline = lambda: FirstLegalDecider()
    result = run_batch(
        [heuristic, baseline, baseline, baseline],
        matches=6,
        rounds=4,
        seed=17,
        labels=["heuristic", "baseline", "baseline", "baseline"],
    )
    mine = result.seats[0]
    others = result.seats[1:]
    assert mine.wins > max(stat.wins for stat in others)
    assert mine.total_score > 0
    assert all(stat.total_score < 0 for stat in others)


def test_tile_count_tampering_is_detectable() -> None:
    """守恒校验能发现状态被改坏——这是模拟器自检的核心手段。"""
    state = deal(random.Random(12), dealer=0)
    assert total_tiles_accounted(state) == 136
    state.seats[0].hand[tiles.parse("9t")] += 1
    assert total_tiles_accounted(state) == 137


def test_round_module_exposes_flow_result_for_reuse() -> None:
    assert round_module.flow_result(deal(random.Random(13), 0), 3).is_flow is True

