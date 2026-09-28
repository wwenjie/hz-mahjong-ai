import pytest

from majiang.rules import action as action_module
from majiang.rules import tiles
from majiang.rules.action import (
    ANGANG,
    BUGANG,
    CHI,
    DISCARD,
    GANG,
    HU,
    MINGGANG,
    PASS,
    PENG,
    Action,
)
from majiang.rules.god import GodState
from majiang.rules.hand import Hand
from majiang.rules.melds import CHI as CHI_MELD
from majiang.rules.melds import PENG as PENG_MELD
from majiang.rules.melds import Meld
from majiang.rules.situation import (
    PHASE_DEAL,
    PHASE_DRAW,
    PHASE_FINISHED,
    PHASE_RESPONSE_CHI,
    PHASE_RESPONSE_PENG,
    PHASE_SETTLED,
    Situation,
    SituationError,
)
from majiang.rules.table import TableState

from .helpers import parse_spec

RESTRICTED_BY_SEAT_THREE = GodState(catch_play=True, god_discarder_seat=3)


def make_situation(
    codes: str,
    *,
    phase: str = PHASE_DRAW,
    seat: int = 0,
    turn: int = 0,
    melds: tuple[Meld, ...] = (),
    god: GodState | None = None,
    table: TableState | None = None,
    responding: tuple[int, ...] = (),
    offered: str | None = None,
    drawn: str | None = None,
) -> Situation:
    return Situation(
        seat=seat,
        phase=phase,
        turn=turn,
        hand=Hand.from_codes([tiles.to_code(t) for t in parse_spec(codes)], melds),
        god=god or GodState(),
        table=table or TableState.open(),
        responding_seats=responding,
        offered_tile=None if offered is None else tiles.parse(offered),
        drawn_tile=None if drawn is None else tiles.parse(drawn),
    )


def kinds(actions: tuple[Action, ...]) -> set[str]:
    return {a.kind for a in actions}


def test_discard_options_cover_every_held_tile() -> None:
    situation = make_situation("1w2w3w4w5w6w7w8w9w1b2b3b5b6b")
    actions = action_module.legal_actions(situation)
    assert HU not in kinds(actions)
    assert [a.tile for a in actions if a.kind == DISCARD] == parse_spec(
        "1w2w3w4w5w6w7w8w9w1b2b3b5b6b"
    )


def test_winning_hand_offers_hu() -> None:
    # 胡只发生在本人摸牌之后，因此必须给出 drawn_tile
    situation = make_situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    assert HU in kinds(action_module.legal_actions(situation))


def test_hu_not_offered_without_a_self_drawn_tile() -> None:
    """碰/吃之后轮到本人时手上会多一张但并非自摸，此时不可胡。"""
    situation = make_situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b")
    actions = action_module.legal_actions(situation)
    assert HU not in kinds(actions)
    assert any(action.kind == DISCARD for action in actions)


def test_catch_play_restricts_discard_to_the_drawn_tile() -> None:
    situation = make_situation(
        "1w2w3w4w5w6w7w8w9w1b2b3b5b5b",
        god=RESTRICTED_BY_SEAT_THREE,
        drawn="5b",
    )
    actions = action_module.legal_actions(situation)
    assert kinds(actions) == {DISCARD, HU}
    assert [a.tile for a in actions if a.kind == DISCARD] == [tiles.parse("5b")]


def test_restricted_discard_without_drawn_tile_raises() -> None:
    situation = make_situation(
        "1w2w3w4w5w6w7w8w9w1b2b3b5b5b",
        god=RESTRICTED_BY_SEAT_THREE,
    )
    with pytest.raises(SituationError):
        action_module.legal_actions(situation)


def test_concealed_gang_detected() -> None:
    situation = make_situation("9t9t9t9t1w2w3w4w5w6w7w8w1b1b")
    gangs = [a for a in action_module.legal_actions(situation) if a.kind == GANG]
    assert [(a.tile, a.gang_kind) for a in gangs] == [(tiles.parse("9t"), ANGANG)]


def test_god_is_never_a_concealed_gang() -> None:
    situation = make_situation("白白白白1w2w3w4w5w6w7w8w9w1b")
    assert [a for a in action_module.legal_actions(situation) if a.kind == GANG] == []


def test_added_gang_requires_an_existing_peng() -> None:
    with_peng = make_situation(
        "5b1w2w3w4w5w6w7w8w9w1b",
        melds=(Meld(kind=PENG_MELD, tiles=(tiles.parse("5b"),) * 3),),
    )
    assert {a.gang_kind for a in action_module.legal_actions(with_peng) if a.kind == GANG} == {
        BUGANG
    }
    without_peng = make_situation("5b1w2w3w4w5w6w7w8w9w1b")
    assert [a for a in action_module.legal_actions(without_peng) if a.kind == GANG] == []


def test_no_gang_options_in_the_reserved_tail() -> None:
    situation = make_situation(
        "9t9t9t9t1w2w3w4w5w6w7w8w1b1b",
        table=TableState(wall_remaining=20),
    )
    assert [a for a in action_module.legal_actions(situation) if a.kind == GANG] == []


def test_restricted_player_may_not_added_gang() -> None:
    situation = make_situation(
        "5b1w2w3w4w5w6w7w8w9w1b",
        melds=(Meld(kind=PENG_MELD, tiles=(tiles.parse("5b"),) * 3),),
        god=RESTRICTED_BY_SEAT_THREE,
        drawn="5b",
    )
    assert [a for a in action_module.legal_actions(situation) if a.kind == GANG] == []


@pytest.mark.parametrize(
    "situation_kwargs",
    [
        {"phase": PHASE_SETTLED},
        {"phase": PHASE_FINISHED},
        {"phase": PHASE_DEAL},
        {"turn": 2},
        {"seat": -1},
    ],
)
def test_no_actions_outside_my_turn(situation_kwargs: dict[str, object]) -> None:
    situation = make_situation("1w2w3w4w5w6w7w8w9w1b2b3b5b6b", **situation_kwargs)
    assert action_module.legal_actions(situation) == ()


def test_peng_window_offers_peng_and_melded_gang() -> None:
    situation = make_situation(
        "3b3b3b1w2w3w4w5w6w7w8w9w1b1b",
        phase=PHASE_RESPONSE_PENG,
        turn=3,
        responding=(0,),
        offered="3b",
    )
    actions = action_module.legal_actions(situation)
    assert kinds(actions) == {PASS, PENG, GANG}
    assert [a.gang_kind for a in actions if a.kind == GANG] == [MINGGANG]


def test_peng_window_needs_two_matching_tiles() -> None:
    situation = make_situation(
        "3b9t9t1w2w3w4w5w6w7w8w1b2b",
        phase=PHASE_RESPONSE_PENG,
        turn=3,
        responding=(0,),
        offered="3b",
    )
    assert kinds(action_module.legal_actions(situation)) == {PASS}


def test_peng_window_denied_when_not_responding() -> None:
    situation = make_situation(
        "3b3b3b1w2w3w4w5w6w7w8w9w1b1b",
        phase=PHASE_RESPONSE_PENG,
        turn=3,
        responding=(1,),
        offered="3b",
    )
    assert action_module.legal_actions(situation) == ()


def test_peng_window_ignores_god_as_offered_tile() -> None:
    situation = make_situation(
        "白白白白1w2w3w4w5w6w7w8w9w1b",
        phase=PHASE_RESPONSE_PENG,
        turn=3,
        responding=(0,),
        offered="白",
    )
    assert kinds(action_module.legal_actions(situation)) == {PASS}


def test_peng_window_restricted_player_can_only_pass() -> None:
    situation = make_situation(
        "3b3b3b1w2w3w4w5w6w7w8w9w1b1b",
        phase=PHASE_RESPONSE_PENG,
        turn=3,
        responding=(0,),
        offered="3b",
        god=RESTRICTED_BY_SEAT_THREE,
    )
    assert kinds(action_module.legal_actions(situation)) == {PASS}


def test_peng_window_has_no_melded_gang_in_the_reserved_tail() -> None:
    situation = make_situation(
        "3b3b3b1w2w3w4w5w6w7w8w9w1b1b",
        phase=PHASE_RESPONSE_PENG,
        turn=3,
        responding=(0,),
        offered="3b",
        table=TableState(wall_remaining=20),
    )
    assert kinds(action_module.legal_actions(situation)) == {PASS, PENG}


def test_chi_window_lists_every_valid_combination() -> None:
    situation = make_situation(
        "1w2w2w4w4w5w9t9t东东东中中中",
        phase=PHASE_RESPONSE_CHI,
        turn=3,
        responding=(0,),
        offered="3w",
    )
    combos = {tuple(sorted(a.tiles)) for a in action_module.legal_actions(situation) if a.kind == CHI}
    assert combos == {
        tuple(sorted(parse_spec("1w2w"))),
        tuple(sorted(parse_spec("2w4w"))),
        tuple(sorted(parse_spec("4w5w"))),
    }


def test_chi_window_respects_suit_boundaries() -> None:
    situation = make_situation(
        "1w2w3w4w5w6w7w8w9w1b1b1b1b9t",
        phase=PHASE_RESPONSE_CHI,
        turn=3,
        responding=(0,),
        offered="9w",
    )
    combos = {tuple(sorted(a.tiles)) for a in action_module.legal_actions(situation) if a.kind == CHI}
    assert combos == {tuple(sorted(parse_spec("7w8w")))}


def test_chi_window_blocked_after_two_chi_melds() -> None:
    melds = (
        Meld(kind=CHI_MELD, tiles=tuple(parse_spec("1w2w3w"))),
        Meld(kind=CHI_MELD, tiles=tuple(parse_spec("4w5w6w"))),
    )
    situation = make_situation(
        "7w8w9w1b2b3b9t9t",
        melds=melds,
        phase=PHASE_RESPONSE_CHI,
        turn=3,
        responding=(0,),
        offered="2w",
    )
    assert kinds(action_module.legal_actions(situation)) == {PASS}


def test_chi_window_denied_when_not_responding_or_restricted() -> None:
    not_responding = make_situation(
        "1w2w3w4w5w6w7w8w9w1b2b3b5b5b",
        phase=PHASE_RESPONSE_CHI,
        turn=3,
        responding=(1,),
        offered="3w",
    )
    assert action_module.legal_actions(not_responding) == ()
    restricted = make_situation(
        "1w2w3w4w5w6w7w8w9w1b2b3b5b5b",
        phase=PHASE_RESPONSE_CHI,
        turn=3,
        responding=(0,),
        offered="3w",
        god=RESTRICTED_BY_SEAT_THREE,
    )
    assert kinds(action_module.legal_actions(restricted)) == {PASS}


def test_response_phase_without_offered_tile_is_rejected() -> None:
    with pytest.raises(SituationError):
        make_situation(
            "1w2w3w4w5w6w7w8w9w1b2b3b5b5b",
            phase=PHASE_RESPONSE_CHI,
            turn=3,
            responding=(0,),
        )


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        (Action(DISCARD, tile=tiles.parse("1w")), {"action": "discard", "tile": "1w"}),
        (Action(PENG, tile=tiles.parse("3b")), {"action": "peng", "tile": "3b"}),
        (
            Action(CHI, tile=tiles.parse("3b"), tiles=tuple(parse_spec("1b2b"))),
            {"action": "chi", "tile": "3b", "tiles": ["1b", "2b"]},
        ),
        (Action(HU), {"action": "hu"}),
        (Action(PASS), {"action": "pass", "tile": ""}),
    ],
)
def test_action_payload_shape(action: Action, payload: dict[str, object]) -> None:
    assert action.to_payload() == payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "weird"},
        {"kind": DISCARD},
        {"kind": GANG, "tile": tiles.parse("1w")},
        {"kind": PENG, "tile": tiles.parse("1w"), "gang_kind": ANGANG},
    ],
)
def test_action_rejects_invalid_shapes(kwargs: dict[str, object]) -> None:
    with pytest.raises(action_module.ActionError):
        Action(**kwargs)


def test_situation_helper_properties() -> None:
    mine = make_situation("1w2w3w4w5w6w7w8w9w1b2b3b5b6b")
    assert mine.is_my_turn and not mine.in_response_window and not mine.is_restricted
    window = make_situation(
        "1w2w3w4w5w6w7w8w9w1b2b3b5b6b",
        phase=PHASE_RESPONSE_CHI,
        turn=3,
        responding=(0,),
        offered="2w",
    )
    assert window.in_response_window and not window.is_my_turn
    observer = make_situation("1w2w3w4w5w6w7w8w9w1b2b3b5b6b", seat=-1)
    assert observer.is_observer and not observer.is_restricted

def _restricted_situation(codes, melds=(), catch=True):
    from majiang.rules.god import GodState
    from majiang.rules.hand import Hand
    from majiang.rules.situation import PHASE_DRAW, Situation
    from majiang.rules.table import TableState
    from majiang.rules import tiles as tile_module

    counts = [0] * tile_module.TILE_KINDS
    for code in codes:
        counts[tile_module.parse(code)] += 1
    return Situation.from_parts(
        seat=0,
        phase=PHASE_DRAW,
        turn=0,
        hand=Hand.from_counts(counts, tuple(melds)),
        god=GodState(
            hand_gods=0,
            catch_play=catch,
            god_discarder_seat=(3 if catch else -1),
        ),
        table=TableState(wall_remaining=40, dealer_seat=0, round_no=1),
        drawn_tile=tile_module.parse("5b"),
    )


def test_restricted_player_may_declare_angang() -> None:
    """**抓打圈内暗杠必须放行**（agent C 复核发现的规则 bug）。

    平台规则 §1.1 原文是「不能吃、碰、明杠（**仅暗杠与自摸胡**）」——括号内是**允许**的
    动作。原实现两道闸门（`concealed_gang_options` + `_turn_actions`）都按「受限则无杠」
    处理，把暗杠一并禁掉，等于丢了一类合法动作。暗杠在本平台价值极高：
    链 +1、自带补牌、不经对手回合、且不暴露任何信息。
    """
    situation = _restricted_situation(
        ["9t", "9t", "9t", "9t", "1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "1b", "5b"]
    )
    assert situation.is_restricted is True
    kinds = {(action.kind, action.gang_kind) for action in action_module.legal_actions(situation)}
    assert (GANG, ANGANG) in kinds, f"圈内应可行暗杠，实际 {sorted(kinds)}"


def test_restricted_player_may_not_declare_bugang() -> None:
    """圈内**仍不可补杠**：规则只放行「暗杠与自摸胡」。

    补杠要经对手回合、且必须打出手上的牌，而圈内只能打刚摸到的那一张。
    """
    from majiang.rules.melds import Meld

    melds = [Meld(kind="peng", tiles=(tiles.parse("9t"),) * 3)]
    situation = _restricted_situation(
        ["9t", "1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "1b", "5b"], melds
    )
    kinds = {(action.kind, action.gang_kind) for action in action_module.legal_actions(situation)}
    assert (GANG, BUGANG) not in kinds, f"圈内不应可补杠，实际 {sorted(kinds)}"
    # 反向确认：不受限时可以补杠（否则上一条可能因构造错误而假通过）
    free = _restricted_situation(
        ["9t", "1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "1b", "5b"], melds, catch=False
    )
    free_kinds = {(action.kind, action.gang_kind) for action in action_module.legal_actions(free)}
    assert (GANG, BUGANG) in free_kinds, f"不受限时应可补杠，实际 {sorted(free_kinds)}"

