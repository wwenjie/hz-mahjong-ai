import pytest

from majiang.rules import fan as fan_module
from majiang.rules import score as score_module
from majiang.rules import tiles

from .helpers import counts_of


def test_plain_win_without_god_or_chain() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b")
    result = fan_module.compute_fan(waiting, tiles.parse("5b"))
    assert result.hu is True
    assert result.fan == 1
    assert result.branch_name == "平胡"
    assert result.baotou is False
    assert result.four_gods is False
    assert result.detail == ("平胡",)


def test_baotou_doubles_fan() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b白")
    result = fan_module.compute_fan(waiting, tiles.parse("5b"))
    assert result.hu is True and result.baotou is True
    assert result.four_gods is False
    assert result.fan == 2
    assert result.detail == ("平胡", "爆头")


def test_four_gods_stacks_with_baotou() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w白白白白")
    result = fan_module.compute_fan(waiting, tiles.parse("东"))
    assert result.hu is True and result.baotou is True and result.four_gods is True
    assert result.fan == 1 * 1 * 2 * 2
    assert result.detail == ("平胡", "4个白板", "爆头")


def test_four_gods_bonus_can_be_disabled() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w白白白白")
    result = fan_module.compute_fan(
        waiting, tiles.parse("东"), baotou_stacks_with_four_gods=False
    )
    assert result.four_gods is True and result.baotou is True
    assert result.fan == 2


def test_chain_multiplier_is_power_of_two() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b")
    draw = tiles.parse("5b")
    for chain in range(fan_module.CHAIN_MAX + 1):
        result = fan_module.compute_fan(waiting, draw, chain_count=chain, piao_count=0)
        assert result.fan == 2**chain
        assert result.gang_count == chain


def test_seven_pairs_with_chain_and_four_gods_scores_documented_maximum() -> None:
    waiting = counts_of("1w1w1w1w2w2w2w2w3w3w3w3w白")
    result = fan_module.compute_fan(
        waiting, tiles.parse("4w"), chain_count=3, piao_count=3
    )
    assert result.hu is True
    assert result.branch_name == "豪华七对×3" and result.branch == 16
    assert result.baotou is True and result.four_gods is True
    assert result.piao_count == 3 and result.gang_count == 0
    assert result.fan == 512
    assert result.detail == ("豪华七对×3", "三财飘", "4个白板", "爆头")


def test_plain_branch_scores_documented_maximum() -> None:
    waiting = counts_of("1w2w3w白")
    result = fan_module.compute_fan(
        waiting,
        tiles.parse("5b"),
        meld_count=3,
        chain_count=6,
        piao_count=3,
    )
    assert result.hu is True
    assert result.branch_name == "平胡" and result.branch == 1
    assert result.baotou is True and result.four_gods is True
    assert result.gang_count == 3 and result.piao_count == 3
    assert result.fan == 256
    assert result.detail == ("平胡", "杠飘链×6", "4个白板", "爆头")


def test_not_a_winning_draw_reports_not_hu() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b")
    result = fan_module.compute_fan(waiting, tiles.parse("9t"))
    assert result.hu is False and result.fan == 0 and result.branch_name == ""


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chain_count": 7}, "动作链次数"),
        ({"chain_count": -1}, "动作链次数"),
        ({"chain_count": 2, "piao_count": 3}, "飘次数"),
    ],
)
def test_fan_rejects_invalid_chain(kwargs: dict[str, int], match: str) -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w1b1b2b2b")
    with pytest.raises(fan_module.FanError, match=match):
        fan_module.compute_fan(waiting, tiles.parse("9t"), **kwargs)


def test_you_cai_bi_kao_blocks_plain_win_with_gods() -> None:
    waiting = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b白")
    blocked = fan_module.compute_fan(
        waiting, tiles.parse("5b"), you_cai_bi_kao=True, baotou=False
    )
    assert blocked.hu is False and blocked.fan == 0
    allowed_by_gang_draw = fan_module.compute_fan(
        waiting, tiles.parse("5b"), you_cai_bi_kao=True, baotou=False, is_gang_draw=True
    )
    assert allowed_by_gang_draw.hu is True and allowed_by_gang_draw.fan == 1
    allowed_by_baotou = fan_module.compute_fan(waiting, tiles.parse("5b"), you_cai_bi_kao=True)
    assert allowed_by_baotou.hu is True and allowed_by_baotou.fan == 2


def test_settlement_matches_documented_shape() -> None:
    settlement = score_module.settle(fan=1, base=1)
    assert settlement.dealer_hu == score_module.ScoreLine(win=24, lose=(8, 8, 8))
    assert settlement.nondealer_hu == score_module.ScoreLine(win=10, lose=(8, 1, 1))
    assert settlement.to_api_shape() == {
        "dealer_hu": {"win": 24, "lose": [8, 8, 8]},
        "nondealer_hu": {"win": 10, "lose": [8, 1, 1]},
    }


@pytest.mark.parametrize("fan", [1, 2, 4, 16, 512])
def test_settlement_is_zero_sum(fan: int) -> None:
    for winner in range(4):
        for dealer in range(4):
            deltas = score_module.seat_deltas(fan, 1, winner_seat=winner, dealer_seat=dealer)
            assert sum(deltas) == 0
            assert deltas[winner] > 0


def test_seat_deltas_match_documented_observation() -> None:
    deltas = score_module.seat_deltas(fan=4, base=1, winner_seat=1, dealer_seat=1)
    assert deltas == (-32, 96, -32, -32)


def test_seat_deltas_dealer_wins_collects_eight_from_each() -> None:
    deltas = score_module.seat_deltas(fan=2, base=1, winner_seat=0, dealer_seat=0)
    assert deltas == (48, -16, -16, -16)


def test_seat_deltas_non_dealer_wins_dealer_pays_eight() -> None:
    deltas = score_module.seat_deltas(fan=1, base=1, winner_seat=2, dealer_seat=0)
    assert deltas == (-8, -1, 10, -1)


@pytest.mark.parametrize("base", [0, 10001])
def test_settlement_rejects_out_of_range_base(base: int) -> None:
    with pytest.raises(score_module.ScoreError):
        score_module.settle(fan=1, base=base)


def test_fan_rejects_more_gods_than_the_wall_holds() -> None:
    waiting = counts_of("1w1w1w1w2w2w2w2w3w3w3w3w白")
    with pytest.raises(fan_module.FanError, match="财神"):
        fan_module.compute_fan(waiting, tiles.parse("4w"), chain_count=4, piao_count=4)
