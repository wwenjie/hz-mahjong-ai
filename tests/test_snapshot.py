"""快照模型与桥接测试。

夹具是 2026-09-23 测试房真实对局中原样抓取的快照（指南 v34），因此这些用例同时充当
「协议结构」的回归：一旦平台字段变化，这里会先炸。
"""

from majiang.client.snapshot import Snapshot
from majiang.rules import action as action_module
from majiang.rules import tiles
from majiang.rules.action import DISCARD, HU
from majiang.rules.situation import PHASE_DRAW, PHASE_RESPONSE_PENG

# 本人回合：my_hand 为 14 张，drawn_tile 标注刚摸到的那张
REAL_DRAW_SNAPSHOT = {
    "game_id": "t_0cfde5a00075_r1_b9_t0",
    "phase": "draw",
    "round_no": 1,
    "dealer": 0,
    "turn": 1,
    "waited_seat": 1,
    "wall_remaining": 74,
    "discards": [["1t", "东", "4b"], ["1b", "1w"], ["5t", "7w"], ["7b", "南"]],
    "melds": [[], [], [], []],
    "hand_counts": [13, 14, 13, 13],
    "my_hand": ["7t", "中", "东", "5t", "5b", "北", "2b", "9b", "7w", "6w", "1t", "白", "4b", "2t"],
    "seat": 1,
    "last_discard": "4b",
    "drawn_tile": "2t",
    "god": {
        "baotou": False,
        "chain_count": 0,
        "catch_play": False,
        "god_discarder_seat": -1,
    },
    "scores": [0, 0, 0, 0],
}

# 碰响应窗口：多出 responding_seats 与 window_deadline_ms；本人刚打出 2w，不参与响应
REAL_PENG_SNAPSHOT = {
    "game_id": "t_0cfde5a00075_r1_b6_t0",
    "phase": "response_peng",
    "round_no": 1,
    "dealer": 0,
    "turn": 1,
    "waited_seat": 0,
    "responding_seats": [0],
    "window_deadline_ms": 1790137339816,
    "wall_remaining": 74,
    "discards": [["6t", "5t", "白"], ["1t", "8w", "2w"], ["2t", "7t"], ["4b", "5t"]],
    "melds": [[], [], [], []],
    "hand_counts": [13, 13, 13, 13],
    "my_hand": ["8t", "中", "南", "8b", "6b", "中", "6b", "8w", "9w", "3w", "6b", "4t", "8w"],
    "seat": 1,
    "last_discard": "2w",
    "drawn_tile": "",
    "god": {
        "baotou": False,
        "chain_count": 0,
        "catch_play": True,
        "god_discarder_seat": 0,
    },
    "scores": [0, 0, 0, 0],
}


def test_my_hand_is_complete_and_must_not_add_drawn_tile() -> None:
    snapshot = Snapshot.parse(REAL_DRAW_SNAPSHOT)
    assert len(snapshot.my_hand) == 14
    assert snapshot.hand_counts[snapshot.seat] == 14
    hand = snapshot.hand()
    assert hand.tile_count == 14
    assert hand.god_count == 1
    assert snapshot.drawn_tile == tiles.parse("2t")
    assert hand.counts[tiles.parse("2t")] == 1


def test_draw_snapshot_marks_my_turn() -> None:
    snapshot = Snapshot.parse(REAL_DRAW_SNAPSHOT)
    assert snapshot.phase == PHASE_DRAW
    assert snapshot.is_my_turn is True
    assert snapshot.in_response_window is False
    assert snapshot.expects_action is True


def test_draw_snapshot_yields_discard_options() -> None:
    situation = Snapshot.parse(REAL_DRAW_SNAPSHOT).to_situation()
    actions = action_module.legal_actions(situation)
    assert HU not in {a.kind for a in actions}
    assert len([a for a in actions if a.kind == DISCARD]) == 14


def test_response_snapshot_exposes_window_fields() -> None:
    snapshot = Snapshot.parse(REAL_PENG_SNAPSHOT)
    assert snapshot.phase == PHASE_RESPONSE_PENG
    assert snapshot.responding_seats == (0,)
    assert snapshot.window_deadline_ms == 1790137339816
    assert snapshot.turn == 1  # 响应阶段 turn 是打出牌的那一家
    assert snapshot.last_discard == tiles.parse("2w")


def test_response_snapshot_offered_tile_is_last_discard() -> None:
    situation = Snapshot.parse(REAL_PENG_SNAPSHOT).to_situation()
    assert situation.offered_tile == tiles.parse("2w")
    assert situation.responding_seats == (0,)
    assert situation.window_deadline_ms == 1790137339816


def test_seat_outside_window_has_no_actions() -> None:
    situation = Snapshot.parse(REAL_PENG_SNAPSHOT).to_situation()
    assert situation.seat == 1
    assert action_module.legal_actions(situation) == ()


def test_catch_play_is_carried_through() -> None:
    snapshot = Snapshot.parse(REAL_PENG_SNAPSHOT)
    god = snapshot.god_state()
    assert god.catch_play is True and god.god_discarder_seat == 0
    assert god.restricts(1) is True
    assert god.is_exempt(0) is True
    assert snapshot.to_situation().is_restricted is True


def test_wall_remaining_becomes_table_state() -> None:
    table = Snapshot.parse(REAL_DRAW_SNAPSHOT).table_state()
    assert table.wall_remaining == 74
    assert table.draws_left == 54
    assert table.can_draw is True and table.can_gang is True
    assert table.dealer_seat == 0 and table.round_no == 1


def test_discards_are_parsed_per_seat() -> None:
    snapshot = Snapshot.parse(REAL_PENG_SNAPSHOT)
    assert len(snapshot.discards) == 4
    assert snapshot.discards[0] == tuple(tiles.parse_all(["6t", "5t", "白"]))
    assert snapshot.discards[3] == tuple(tiles.parse_all(["4b", "5t"]))


def test_empty_drawn_tile_becomes_none() -> None:
    snapshot = Snapshot.parse(REAL_PENG_SNAPSHOT)
    assert snapshot.drawn_tile is None


def test_unknown_keys_are_preserved() -> None:
    snapshot = Snapshot.parse({**REAL_DRAW_SNAPSHOT, "future_field": {"a": 1}})
    assert snapshot.extra == {"future_field": {"a": 1}}


def test_snapshot_round_trips_into_situation() -> None:
    situation = Snapshot.parse(REAL_DRAW_SNAPSHOT).to_situation()
    assert situation.seat == 1
    assert situation.phase == PHASE_DRAW
    assert situation.turn == 1
    assert situation.drawn_tile == tiles.parse("2t")
    assert situation.table.wall_remaining == 74
    assert len(situation.discards) == 4
    assert situation.hand.tile_count == 14


def test_platform_placeholder_zero_tile_means_not_applicable() -> None:
    """平台用点数为 0 的牌码表示「本字段不适用」，只认空串会在开局丢快照。

    实测依据（`logs/a_*.jsonl`）：307 场里 **125 场**出现过
    `TileCodeError: 非法牌码: '0w'`，共 1795 次，**全部落在开局 2.6 秒内**
    （每场对局的前几个快照），中局 **0 次**。每场因此多花约 0.6 秒重拉快照。
    """
    from majiang.client.snapshot import nullable_tile

    assert nullable_tile("") is None
    assert nullable_tile(None) is None
    assert nullable_tile("0w") is None and nullable_tile("0b") is None
    assert nullable_tile("2t") == tiles.parse("2t"), "正常牌码不得被当成占位吞掉"

    for field in ("drawn_tile", "last_discard"):
        snapshot = Snapshot.parse({**REAL_DRAW_SNAPSHOT, field: "0w"})
        assert getattr(snapshot, field) is None, f"{field} 的占位码应解析为 None"
    assert Snapshot.parse({**REAL_DRAW_SNAPSHOT, "drawn_tile": "0w"}).to_situation()


def test_placeholder_is_not_tolerated_in_hand() -> None:
    """**暗手必须继续严格报错**：若哪天 `my_hand` 里真出现 `0w`（例如平台启用赤 5），
    把它当占位就会**静默丢掉一张真牌**——那比报错糟得多。"""
    import pytest

    from majiang.rules.tiles import TileCodeError

    with pytest.raises(TileCodeError):
        Snapshot.parse({**REAL_DRAW_SNAPSHOT, "my_hand": ["0w"] + REAL_DRAW_SNAPSHOT["my_hand"][1:]})
    with pytest.raises(TileCodeError):
        Snapshot.parse({**REAL_DRAW_SNAPSHOT, "discards": [["0w"], [], [], []]})
