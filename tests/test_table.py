import pytest

from majiang.rules import god as god_module
from majiang.rules import table as table_module
from majiang.rules.god import GodState
from majiang.rules.table import TableState


def test_wall_composition() -> None:
    assert table_module.DEALT_TILES == 53
    assert table_module.INITIAL_WALL == 83
    assert table_module.RESERVED_TILES == 20
    assert table_module.MAX_DRAWS == 63


def test_open_table_can_draw_and_gang() -> None:
    table = TableState.open(dealer_seat=2, round_no=3)
    assert table.draws_made == 0
    assert table.draws_left == 63
    assert table.can_draw and table.can_gang
    assert not table.is_exhausted
    assert table.dealer_seat == 2 and table.round_no == 3


def test_reserved_tiles_close_both_draw_and_gang() -> None:
    last_drawable = TableState(wall_remaining=table_module.RESERVED_TILES + 1)
    assert last_drawable.can_draw and last_drawable.can_gang
    assert last_drawable.draws_left == 1

    reserved_only = last_drawable.after_draw()
    assert reserved_only.wall_remaining == table_module.RESERVED_TILES
    assert not reserved_only.can_draw
    assert not reserved_only.can_gang
    assert reserved_only.is_exhausted
    assert reserved_only.draws_left == 0


def test_draw_beyond_wall_raises() -> None:
    reserved_only = TableState(wall_remaining=table_module.RESERVED_TILES)
    with pytest.raises(table_module.TableError):
        reserved_only.after_draw()
    with pytest.raises(table_module.TableError):
        TableState.open().after_draw(-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wall_remaining": -1},
        {"wall_remaining": table_module.INITIAL_WALL + 1},
        {"dealer_seat": 4},
        {"dealer_seat": -1},
        {"round_no": 0},
    ],
)
def test_table_state_rejects_invalid_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(table_module.TableError):
        TableState(**kwargs)


def test_dealer_keeps_seat_on_exhausted_or_dealer_win() -> None:
    assert table_module.next_dealer_seat(1, exhausted=True, dealer_won=False) == 1
    assert table_module.next_dealer_seat(1, exhausted=False, dealer_won=True) == 1
    assert table_module.next_dealer_seat(1, exhausted=False, dealer_won=False) == 2
    assert table_module.next_dealer_seat(3, exhausted=False, dealer_won=False) == 0


def test_next_round_resets_wall_and_advances() -> None:
    table = TableState(wall_remaining=30, dealer_seat=1, round_no=5)
    nxt = table_module.next_round(table, exhausted=False, dealer_won=False)
    assert nxt.round_no == 6 and nxt.dealer_seat == 2
    assert nxt.wall_remaining == table_module.INITIAL_WALL

    kept = table_module.next_round(table, exhausted=True, dealer_won=False)
    assert kept.dealer_seat == 1 and kept.round_no == 6


def test_god_state_defaults() -> None:
    state = GodState()
    assert state.hand_gods == 0 and state.chain_count == 0 and state.gang_count == 0
    assert state.catch_play is False and state.god_discarder_seat == god_module.NO_DISCARDER
    assert state.restricts(0) is False and state.is_exempt(0) is False


def test_gang_count_is_chain_minus_piao() -> None:
    assert GodState(chain_count=5, piao_count=3).gang_count == 2
    assert GodState(chain_count=3, piao_count=0).gang_count == 3


def test_catch_play_restricts_everyone_but_the_discarder() -> None:
    state = GodState(catch_play=True, god_discarder_seat=2)
    assert state.restricts(0) and state.restricts(1) and state.restricts(3)
    assert not state.restricts(2)
    assert state.is_exempt(2) and not state.is_exempt(0)


def test_after_gang_only_extends_chain() -> None:
    state = GodState(hand_gods=1, chain_count=2, piao_count=1, catch_play=True, god_discarder_seat=1)
    extended = state.after_gang()
    assert extended.chain_count == 3 and extended.piao_count == 1 and extended.gang_count == 2
    assert extended.catch_play and extended.god_discarder_seat == 1
    assert extended.hand_gods == 1


def test_after_piao_extends_chain_and_restarts_catch_play() -> None:
    state = GodState(hand_gods=2, chain_count=1, piao_count=1, baotou=True)
    piaoed = state.after_piao(3)
    assert piaoed.chain_count == 2 and piaoed.piao_count == 2
    assert piaoed.hand_gods == 1
    assert piaoed.catch_play and piaoed.god_discarder_seat == 3
    assert piaoed.baotou is True


def test_god_discard_without_piao_breaks_chain_but_opens_catch_play() -> None:
    state = GodState(hand_gods=3, chain_count=2, piao_count=1)
    broken = state.after_god_discard_without_piao(0)
    assert broken.chain_count == 0 and broken.piao_count == 0
    assert broken.catch_play and broken.god_discarder_seat == 0
    assert broken.hand_gods == 2


def test_after_break_clears_chain_and_catch_play() -> None:
    state = GodState(hand_gods=1, chain_count=4, piao_count=2, catch_play=True, god_discarder_seat=1)
    cleared = state.after_break()
    assert cleared.chain_count == 0 and cleared.piao_count == 0
    assert not cleared.catch_play
    assert cleared.god_discarder_seat == god_module.NO_DISCARDER
    assert cleared.hand_gods == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chain_count": 7},
        {"chain_count": 1, "piao_count": 2},
        {"hand_gods": 5},
        {"hand_gods": 4, "chain_count": 1, "piao_count": 1},
        {"catch_play": True, "god_discarder_seat": -1},
        {"catch_play": False, "god_discarder_seat": 1},
    ],
)
def test_god_state_rejects_invalid_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(god_module.GodStateError):
        GodState(**kwargs)


def test_from_god_field_reads_snapshot_keys() -> None:
    state = GodState.from_god_field(
        {"baotou": True, "chain_count": 2, "catch_play": True, "god_discarder_seat": 3},
        hand_gods=1,
        piao_count=1,
    )
    assert state.baotou is True and state.chain_count == 2 and state.piao_count == 1
    assert state.gang_count == 1
    assert state.god_discarder_seat == 3
    empty = GodState.from_god_field(None)
    assert empty.chain_count == 0 and not empty.catch_play
