"""事件流重建器测试（tasks.md 6A.4）。

两个真 bug 的回归防线（都是在真实数据上跑出来的）：

1. **座位 0 的事件被整批丢弃**。原写法 ``seat = event.get("seat", -1) or -1`` 里 0 是 falsy，
   座位 0 会被变成 -1，于是它的摸打全部落进「未知事件」分支。表现为牌种计数超过 4 张、
   吃碰的牌在牌河里找不到。
2. **吃副露里被吃的那张重复计数**。实测 ``chi`` 的 ``data.tiles`` **已包含**被吃那张
   （``["7b","8b","9b"]`` 配 ``tile="9b"``），若再拼一次 ``tile``，副露就多一张，
   全牌种计数随之超过 4 张。
"""

from __future__ import annotations

import pytest

from majiang.rules import melds as melds_module
from majiang.rules import tiles
from majiang.sim import replay

# 四家起手：座位 0 是庄家，14 张（**庄家的第 14 张在 start_hands 里，不是摸来的**）
DEALER_HAND = ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "1b", "2b", "3b", "4b", "5b"]
OTHER_HAND = ["1t", "2t", "3t", "4t", "5t", "6t", "7t", "8t", "9t", "6b", "7b", "8b", "9b"]


def make_payload(events, *, start_hands=None, dealer=0, result=None, rounds=None):
    return {
        "game_id": "g1",
        "status": "finished",
        "seats": [{"user_id": f"u{i}"} for i in range(4)],
        "blocks": [
            {
                "seq_start": 1,
                "seq_end": 1 + len(events),
                "round_no": 1,
                "dealer": dealer,
                "start_hands": start_hands or [DEALER_HAND, OTHER_HAND, OTHER_HAND, OTHER_HAND],
                "truncated": False,
                "events": events,
            }
        ],
        "rounds": [result or {}] if rounds is None else rounds,
    }


def event(kind, seat, tile="", data=None):
    return {"seq": 0, "type": kind, "seat": seat, "tile": tile, "data": data, "ts": 0}


def run(events, **kwargs) -> replay.ReplayState:
    state = replay.from_payload(make_payload(events, **kwargs))
    for item in events:
        replay.apply_event(state, item)
    return state


def count_of(state: replay.ReplayState, code: str) -> int:
    target = tiles.parse(code)
    total = 0
    for seat in state.seats:
        total += seat.hand[target]
        total += sum(meld.counts()[target] for meld in seat.melds)
        total += seat.discards.count(target)
    return total


# --- 起手与庄家 ---------------------------------------------------------------------


def test_dealer_extra_tile_is_not_drawn_again() -> None:
    """庄家第 14 张已在 start_hands 里；若再补摸一张，牌数会整体多一张。"""
    state = run([event("tile_discarded", 0, "1w")])
    assert sum(state.seats[0].hand) == 13, "庄家出牌后应剩 13 张"
    assert state.draws == 0, "本局还没有任何摸牌事件"
    assert state.seats[0].discards == [tiles.parse("1w")]


def test_situation_refuses_states_before_the_first_draw() -> None:
    """未摸牌前牌墙是 84，TableState 按发牌 53 张定义，只接受 ≤83，因此此时不能构造局面。"""
    state = run([])
    assert state.opened is False
    with pytest.raises(ValueError, match="尚未摸第一张牌"):
        state.situation_for(0)


def test_situation_exposes_only_the_observer_hand() -> None:
    state = run([event("tile_drawn", 1, "3t")])
    situation = state.situation_for(0)
    assert sum(situation.hand.counts) == 14, "观察者只看得到自己的暗手"
    assert situation.hand_counts_for(1) == 14, "对手只暴露暗手张数"
    assert situation.melds_for(1) == ()


# --- 两个真 bug 的回归 ----------------------------------------------------------------


def test_seat_zero_events_are_applied() -> None:
    """回归：``seat or -1`` 会把座位 0 变成 -1，导致它的摸打全部被丢弃。"""
    state = run([event("tile_drawn", 0, "9b"), event("tile_discarded", 0, "9b")])
    assert dict(state.anomalies) == {}
    assert state.draws == 1, "座位 0 的摸牌必须被计入"
    assert state.seats[0].discards == [tiles.parse("9b")]


def test_chi_meld_does_not_duplicate_the_claimed_tile() -> None:
    """回归：``chi`` 的 data.tiles 已含被吃那张，再拼一次会让副露多一张。"""
    events = [
        event("tile_discarded", 2, "9b"),
        event("chi", 3, "9b", {"tiles": ["7b", "8b", "9b"]}),
    ]
    state = run(events)
    meld = state.seats[3].melds[0]
    assert meld.kind == melds_module.CHI
    assert len(meld.tiles) == 3, "吃副露只有三张"
    assert meld.counts()[tiles.parse("9b")] == 1
    assert dict(state.anomalies) == {}
    # 被吃那张要从牌河移除，否则会与副露重复计数
    assert state.seats[2].discards == []
    # 座位 3 手里本来也有一张 9b（它只用 7b/8b 去吃），所以它手 1 张 + 副露 1 张；
    # 加上座位 1 手里的 1 张，全场合计 3 张
    assert state.seats[3].hand[tiles.parse("9b")] == 1
    assert count_of(state, "9b") == 3


def test_chi_without_tiles_is_recorded_not_guessed() -> None:
    """没有 data.tiles 就无法确定用哪两张手牌——宁可留异常也不编一个错副露。"""
    events = [event("tile_discarded", 2, "9b"), event("chi", 3, "9b", None)]
    state = run(events)
    assert state.anomalies["chi-without-tiles"] == 1
    assert state.seats[3].melds == []


def test_peng_takes_two_from_hand_and_one_from_discards() -> None:
    # 碰需要手里有两张，因此给座位 2 一个含两张 9b 的起手
    seat_two = ["1t", "2t", "3t", "4t", "5t", "6t", "7t", "8t", "9t", "9b", "9b", "6b", "7b"]
    hands = [DEALER_HAND, OTHER_HAND, seat_two, OTHER_HAND]
    events = [
        event("tile_discarded", 1, "9b"),
        event("peng", 2, "9b"),
    ]
    state = run(events, start_hands=hands)
    meld = state.seats[2].melds[0]
    assert meld.kind == melds_module.PENG and len(meld.tiles) == 3
    assert state.seats[2].hand[tiles.parse("9b")] == 0
    assert state.seats[1].discards == []
    assert dict(state.anomalies) == {}
    assert count_of(state, "9b") == 4


# --- 抓打圈与链条 ---------------------------------------------------------------------


def test_discarding_a_god_starts_catch_play_and_only_that_seat_ends_it() -> None:
    """抓打圈由打出财神触发，且只有**该座位本人**再打非财神牌才解除。"""
    god = tiles.to_code(tiles.GOD)
    state = run([event("tile_drawn", 1, "6t"), event("tile_discarded", 1, "6t")])
    # 先给座位 1 一张财神再打出（起手里没有白板）
    state.seats[1].hand[tiles.GOD] = 1
    replay.apply_event(state, event("tile_discarded", 1, god))
    assert state.catch_play is True and state.god_discarder == 1

    # 别的座位打非财神牌不解除
    replay.apply_event(state, event("tile_discarded", 2, "6b"))
    assert state.catch_play is True

    # 打出者本人再打非财神牌才解除
    replay.apply_event(state, event("tile_discarded", 1, "7b"))
    assert state.catch_play is False and state.god_discarder == -1


def test_god_discard_restricts_other_seats() -> None:
    god = tiles.to_code(tiles.GOD)
    state = run([event("tile_drawn", 1, "6t")])
    state.seats[1].hand[tiles.GOD] = 1
    replay.apply_event(state, event("tile_discarded", 1, god))
    situation = state.situation_for(0)
    assert situation.god.restricts(0) is True
    assert situation.god.restricts(1) is False, "打出财神者本人不受限"


def test_non_god_discard_resets_chain_counts() -> None:
    state = run([event("tile_drawn", 1, "6t")])
    state.seats[1].chain_count = 3
    state.seats[1].piao_count = 2
    replay.apply_event(state, event("tile_discarded", 1, "6t"))
    assert state.seats[1].chain_count == 0 and state.seats[1].piao_count == 0


# --- 结果与守恒 -----------------------------------------------------------------------


def test_round_ended_supplies_the_official_result_and_game_ended_does_not_clobber_it() -> None:
    """回归：``game_ended`` 的 data 只有 final_scores，若直接覆盖会让 fan/detail 丢失。"""
    events = [
        event("round_ended", 2, data={"dealer": 0, "detail": ["平胡"], "draw": False,
                                      "fan": 1, "round_no": 1, "scores": [-8, -1, 10, -1]}),
        event("game_ended", -1, data={"final_scores": [-8, -1, 10, -1]}),
    ]
    state = run(events)
    assert state.result is not None
    assert state.result["fan"] == 1 and state.result["detail"] == ["平胡"]
    assert state.final_scores == (-8, -1, 10, -1)


def test_timeout_events_do_not_change_the_board() -> None:
    events = [event("tile_drawn", 1, "3t"), event("timeout", 1, data={"kind": "discard"})]
    state = run(events)
    assert state.draws == 1 and sum(state.seats[1].hand) == 14
    assert dict(state.anomalies) == {}


def test_wall_shrinks_by_one_per_draw() -> None:
    state = run([event("tile_drawn", 1, "3t"), event("tile_drawn", 2, "3t")])
    assert state.wall_remaining == replay.table_rules.INITIAL_WALL - 2


def test_unknown_event_types_are_counted_not_guessed() -> None:
    state = run([event("gang", 1, "9b")])
    assert state.anomalies["gang-unhandled"] == 1


def test_iter_before_each_event_yields_the_state_prior_to_the_event() -> None:
    events = [event("tile_drawn", 1, "3t"), event("tile_discarded", 1, "3t")]
    seen = []
    for item, state in replay.iter_before_each_event(make_payload(events)):
        seen.append((item["type"], state.draws, sum(state.seats[1].hand)))
    assert seen == [("tile_drawn", 0, 13), ("tile_discarded", 1, 14)]


def test_all_events_concatenates_blocks_in_seq_order() -> None:
    payload = make_payload([event("tile_drawn", 1, "3t")])
    payload["blocks"].append(
        {**payload["blocks"][0], "seq_start": 100, "start_hands": [None] * 4,
         "events": [event("tile_discarded", 1, "3t")]}
    )
    payload["blocks"].reverse()  # 故意乱序，应按 seq_start 还原
    kinds = [item["type"] for item in replay.all_events(payload)]
    assert kinds == ["tile_drawn", "tile_discarded"]


def test_conservation_reports_per_kind_over_copies() -> None:
    state = run([event("tile_drawn", 1, "3t")])
    report = replay.conservation(state)
    assert report["over_copies"] == 0
    assert report["held_and_discarded"] > 0
    # 人为造一个越界：两个座位各塞 4 张同样的牌（over_copies 统计的是**牌种数**，不是超出的张数）
    state.seats[0].hand[tiles.parse("1w")] = 4
    state.seats[1].hand[tiles.parse("1w")] = 4
    assert replay.conservation(state)["over_copies"] == 1
