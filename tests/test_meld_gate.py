"""吃碰闸门的回归测试（tasks.md 5.5）。

背景：改变闸门松紧是**有实测依据**的方向（我们副露 0.591/局 vs 对手 1.093/局），但这类
开关极易静默失效——`meld_tolerance` 用 `is` 比较 StrEnum 时，
`for_mode(..., meld_tolerance="equal")` 传进来的普通 str 会让档位退化成 strict
而**没有任何报错**。所以这里既测「档位真的被读到」，也直接用随机局面测闸门行为，
不做手工构造向听数（那正是容易写错的地方）。
"""

from __future__ import annotations

import random

import pytest

from majiang.cli import DECIDERS, make_decider
from majiang.rules import tiles
from majiang.rules.action import CHI, GANG, PASS, PENG, legal_actions
from majiang.rules.melds import chi_count
from majiang.rules.situation import PHASE_RESPONSE_CHI, PHASE_RESPONSE_PENG
from majiang.sim import round as round_module
from majiang.strategy.policy import (
    VARIANT_FIELDS,
    HeuristicDecider,
    MeldTolerance,
    Mode,
    PolicyConfig,
    _meld_cell_allows,
)


def test_default_is_strict_and_for_mode_keeps_the_string_comparable() -> None:
    assert PolicyConfig().meld_tolerance is MeldTolerance.STRICT
    decider = make_decider("meld-equal", Mode.QUALIFIER)
    assert decider.config.meld_tolerance == MeldTolerance.EQUAL, (
        "for_mode 传入普通 str；若闸门用 `is` 比较就会静默退化成 strict"
    )


def test_meld_equal_is_registered() -> None:
    assert "meld-equal" in DECIDERS


class _Tournament:
    base_score = 3
    you_cai_bi_kao = True


def test_configure_preserves_every_variant_knob() -> None:
    """**回归防线**：`configure()` 曾在重建 PolicyConfig 时丢掉所有变体开关。

    `configure` 只在真机路径被调用（`engine.py` 的 `Runtime.run`），自对弈不调用它。
    于是真机上 no-chase / ukeire / meld-equal **全都跑成了默认档**，而自对弈跑的是真档位——
    表现为「自对弈有差异、真机无差异」，看起来像噪声，实际是档位根本没生效。
    """
    from dataclasses import fields, replace

    from majiang.strategy.policy import PolicyConfig, _DEFAULT_POLICY

    custom = PolicyConfig(
        meld_tolerance=MeldTolerance.EQUAL,
        tiebreak="exact-ukeire",
        chase_baotou=False,
        route_aware=True,
        preserve_god=True,
        natural_route=True,
        pair_route_pairs=4,
        shanten_weight=7.5,
        feed_weight=1.5,
        god_discard_penalty=10.0,
        value_weight=4.0,
        piao_threshold_scale=0.85,
    )
    decider = HeuristicDecider(custom)
    decider.configure(_Tournament())

    # 服务端注入的字段必须生效
    assert decider.config.base_score == 3
    assert decider.config.you_cai_bi_kao is True
    # 其余每一个字段都必须与 configure 前逐值相同（不逐个列举字段名，避免以后新增
    # 开关时漏测——那正是这个 bug 的成因）
    expected = replace(custom, base_score=3, you_cai_bi_kao=True)
    for field in fields(PolicyConfig):
        assert getattr(decider.config, field.name) == getattr(expected, field.name), (
            f"configure 丢掉了 {field.name}"
        )
    assert decider.config != _DEFAULT_POLICY, "变体不该被 configuration 抹平"


def test_decider_name_reveals_the_variant() -> None:
    """日志记的是 `decider=<name>`；名字若恒为 heuristic，事后无法分辨跑的是哪个档位。"""
    from majiang.strategy.policy import PolicyConfig as Config

    assert HeuristicDecider(Config()).name == "heuristic"
    name = HeuristicDecider(Config(meld_tolerance=MeldTolerance.EQUAL)).name
    assert name.startswith("heuristic[") and "meld-tolerance" in name
    # 用**明确的非默认值**构造，否则默认值一变这个断言就失效（刚才就被默认 tiebreak
    # 从 blocks 改为 exact-ukeire 抓到一次）
    multi = HeuristicDecider(Config(tiebreak="blocks", chase_baotou=False)).name
    assert multi.count("=") == 2, multi
    assert "tiebreak" in multi and "chase-baotou" in multi


def _situations(seed: int, count: int = 400):
    """从随机发牌里造出「我们面对别人打出的一张牌」的响应局面。"""
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        state = round_module.deal(rng, dealer=rng.randrange(4))
        # 未摸第一张牌时牌墙为 84，TableState 按「发牌后已摸走一张」定义，不接受 84
        round_module._draw(state, state.turn)  # noqa: SLF001
        mine = rng.randrange(4)
        for discarder in range(4):
            if discarder == mine:
                continue
            for tile in range(tiles.TILE_KINDS):
                if tile == tiles.GOD or state.seats[discarder].hand[tile] <= 0:
                    continue
                for kind, phase in ((PENG, PHASE_RESPONSE_PENG), (CHI, PHASE_RESPONSE_CHI)):
                    state.seats[discarder].hand[tile] -= 1
                    situation = round_module.situation_for(
                        state, mine, phase, offered=tile, responding=(mine,)
                    )
                    state.seats[discarder].hand[tile] += 1
                    actions = legal_actions(situation)
                    if any(action.kind == kind for action in actions):
                        yield situation, actions
                        produced += 1
                        if produced >= count:
                            return


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gate_behavior_matches_the_declared_rule(seed: int) -> None:
    strict = HeuristicDecider(PolicyConfig())
    relaxed = HeuristicDecider(PolicyConfig(meld_tolerance=MeldTolerance.EQUAL))
    seen_equal = seen_strict = 0

    for situation, actions in _situations(seed):
        offered = situation.offered_tile
        assert offered is not None
        hand = situation.hand
        from majiang.rules import shanten as shanten_module

        current = shanten_module.shanten_any(hand.counts, hand.meld_count)
        deltas = {}
        for action in actions:
            if action.kind not in (PENG, CHI):
                continue
            after = strict._shanten_after_meld(situation, offered, action.kind)
            if after is not None:
                deltas[action.kind] = after - current
        if not deltas:
            continue

        chosen_strict = strict.choose(situation, actions, budget_ms=1000)
        chosen_relaxed = relaxed.choose(situation, actions, budget_ms=1000)

        if chosen_strict is not None and chosen_strict.kind == GANG:
            # 明杠在吃碰之前单独判断（自带补牌、进动作链），不在本测试范围内
            continue

        better = [kind for kind, delta in deltas.items() if delta < 0]
        if better:
            # 有真正改善的选项时，两档都必须选它（放宽档不得因为新增选项而改变选择）
            assert chosen_strict is not None and chosen_strict.kind in better
            assert chosen_relaxed is not None and chosen_relaxed.kind in better
            seen_strict += 1
            continue

        # 没有改善选项：严格档一律不副露
        assert chosen_strict is None or chosen_strict.kind == PASS

        equal = [kind for kind, delta in deltas.items() if delta == 0]
        if current >= 1 and equal:
            assert chosen_relaxed is not None and chosen_relaxed.kind in equal, (
                f"放宽档应接受向听不变的副露（current={current}, deltas={deltas}）"
            )
            seen_equal += 1
        else:
            # 已听牌（0 向听）或只剩变差的选项：放宽档也必须拒绝
            assert chosen_relaxed is None or chosen_relaxed.kind == PASS, (
                f"current={current}, deltas={deltas} 不该副露"
            )

    assert seen_strict > 0 and seen_equal > 0, (
        f"样本未覆盖两种分支（strict={seen_strict}, equal={seen_equal}），测试无效"
    )


def test_relaxed_gate_never_breaks_the_two_chi_limit() -> None:
    """放宽档只动「向听判断」，不得绕过吃摊上限（规则层由 legal_actions 保证）。"""
    relaxed = HeuristicDecider(PolicyConfig(meld_tolerance=MeldTolerance.EQUAL))
    for situation, actions in _situations(7, 300):
        chosen = relaxed.choose(situation, actions, budget_ms=1000)
        if chosen is None or chosen.kind != CHI:
            continue
        assert chi_count(situation.hand.melds) < 2, "吃之前必须还没到上限"


def test_meld_conditional_cell_table_denies_late_and_deep() -> None:
    """`meld_conditional` 的 cell 表：早中盘 × 浅向听放开，尾盘与深向听维持 STRICT。

    表照抄 C 的 1a（`agent/out/meld-cond.log`，7056 房 / 3190 个副露事件）里头部 bot 的副露密度形状：
    1-4 巡 ≤3 向听 24.7%~41.7% 有副露，9-12 巡 3 向听 0%，13+ 巡 ≤11%、3 向听 0.4%。
    这条测试钉的是**边界**——「一律放开」（`v6-equal`）已实测为负，本档的全部价值就在这个边界上。

    只用到 `situation.seat` 与 `situation.discards`（见 `_meld_cell_allows` 的说明），故用最小替身。
    """

    class _Stub:
        def __init__(self, played: int) -> None:
            self.seat = 0
            self.discards = (tuple([0] * played), (), (), ())

    for played in (0, 4, 8):
        for current in (1, 2, 3):
            assert _meld_cell_allows(_Stub(played), current), (played, current)
        assert not _meld_cell_allows(_Stub(played), 4), played
    for played in (9, 12):
        for current in (1, 2):
            assert _meld_cell_allows(_Stub(played), current), (played, current)
        assert not _meld_cell_allows(_Stub(played), 3), played
    for played in (13, 20):
        for current in (1, 2, 3, 4):
            assert not _meld_cell_allows(_Stub(played), current), (played, current)


# --------------------------------------------------------------------------- #
# `meld_chi_best`（①根因修复）+ `meld_chi_tiebreak`（②同向听次排序），2026-10-08
#
# 修的是一处**代码缺陷**：`_shanten_after_meld` 无视 `action.tiles`，对每个 CHI 选项
# 都取 `chi_combinations()[0]`。同一张 `offered` 的两种吃法可以有**不同**的「吃后最小向听」
# （扫 4000 副：≥2 种吃法的 2904 个局面里 1155 个不同）。后果包含**漏吃**。
# A 22:12 裁定：②是①的**下游**、须**单独成第二臂**，故两开关拆开：
#   `v7-chibest` = ①（根因）；`v7-keepchi` = ①+②（次排序）。一次只动一个主导项。
# 三条防线：① 默认关、逐位不变；② 已知漏吃局面被修正；③ 仍走 v5 的路径（未开时不调用）。
# --------------------------------------------------------------------------- #


def _response_situation(hand_codes, offered_code):
    from majiang.rules import tiles as _t
    from majiang.rules.god import GodState
    from majiang.rules.hand import Hand
    from majiang.rules.situation import Situation
    from majiang.rules.table import TableState

    return Situation.from_parts(
        seat=0,
        phase=PHASE_RESPONSE_CHI,
        turn=1,
        hand=Hand.from_codes(list(hand_codes)),
        god=GodState(),
        table=TableState(wall_remaining=60, dealer_seat=1, round_no=1),
        offered_tile=_t.parse(offered_code),
        responding_seats=(0,),
        discards=(),
        melds=(),
    )


def test_meld_chi_best_knob_default_off_and_registered() -> None:
    from majiang.strategy.policy import PolicyConfig as Config
    from majiang.cli import DECIDERS

    assert Config().meld_chi_best is False, "默认必须为 False，否则等于换了冠军"
    assert Config().meld_chi_tiebreak is False, "次排序默认也必须关（A ③ 拆臂）"
    assert "meld_chi_best" in VARIANT_FIELDS, "新开关必须进 VARIANT_FIELDS（configure 会丢字段）"
    assert "meld_chi_tiebreak" in VARIANT_FIELDS
    assert "v7-chibest" in DECIDERS and "v7-keepchi" in DECIDERS
    # 两臂必须分开：`v7-chibest` 只动①，`v7-keepchi` = ①+②
    assert make_decider("v7-chibest", Mode.QUALIFIER).config.meld_chi_best is True
    assert make_decider("v7-chibest", Mode.QUALIFIER).config.meld_chi_tiebreak is False
    assert make_decider("v7-keepchi", Mode.QUALIFIER).config.meld_chi_best is True
    assert make_decider("v7-keepchi", Mode.QUALIFIER).config.meld_chi_tiebreak is True
    assert make_decider("v5", Mode.QUALIFIER).config.meld_chi_best is False
    assert make_decider("v5", Mode.QUALIFIER).config.meld_chi_tiebreak is False


def test_meld_chi_best_fixes_a_known_missed_chi() -> None:
    """**硬证据**：已知会漏吃的局面，开启后必须改判为吃。

    手牌 `1w 6w 6w 8w 9w 2b 3b 9b 1t 北 北 白 白`（2 财神），上家出 `7w`，当前向听 2。
    吃 `8w9w` 真实能到**向听 1**（默认 STRICT 也该吃）；但 `combos[0]=(6w,8w)` 只到向听 2。
    """
    hand = ["1w", "6w", "6w", "8w", "9w", "2b", "3b", "9b", "1t", "北", "北", "白", "白"]
    situation = _response_situation(hand, "7w")
    actions = legal_actions(situation)

    off = make_decider("v5", Mode.QUALIFIER)
    assert off.choose(situation, actions, budget_ms=3000).kind == PASS, (
        "默认档（开关关）必须保持旧行为：漏吃 ⇒ PASS"
    )

    on = make_decider("v7-chibest", Mode.QUALIFIER)   # 只修根因① 就能吃到向听 1
    choice = on.choose(situation, actions, budget_ms=3000)
    assert choice.kind == CHI, "开启后必须吃到向听 1，而不是漏吃"
    from majiang.rules import tiles as _t

    assert set(_t.to_codes(choice.tiles)) == {"8w", "9w"}, "应选能降到向听 1 的吃法（8w9w）"


def test_meld_chi_best_default_is_bit_identical() -> None:
    """**逐位不变**：默认关的决策器对随机响应局面与旧路径完全一致。

    旧路径 = 显式调用 `_shanten_after_meld(..., kind)`（不传 chi_tiles），
    新路径 = `_meld_after_shanten`（开关关时也应回退到同一支）。
    """
    for situation, actions in _situations(11, 150):
        offered = situation.offered_tile
        assert offered is not None
        decider = make_decider("v5", Mode.QUALIFIER)
        for action in actions:
            if action.kind not in (PENG, CHI):
                continue
            legacy = decider._shanten_after_meld(situation, offered, action.kind)
            routed = decider._meld_after_shanten(situation, offered, action)
            assert legacy == routed, (action.describe(), legacy, routed)


def test_meld_chi_best_prefers_wider_ukeire_on_tie() -> None:
    """同向听的多种吃法：开启后选**留下更宽进张**的那手。

    用户局面 `5w 4b5b6b7b7b8b 1t4t6t7t8t9t`，上家出 `7t`：吃 `8t9t`（留 6t7t 两面）
    的进张宽度 > 吃 `6t8t`。
    """
    from majiang.rules import tiles as _t

    hand = ["5w", "4b", "5b", "6b", "7b", "7b", "8b", "1t", "4t", "6t", "7t", "8t", "9t"]
    situation = _response_situation(hand, "7t")
    actions = [a for a in legal_actions(situation) if a.kind == CHI]
    assert len(actions) >= 2, "该局面应有多种吃法"

    on = make_decider("v7-keepchi", Mode.QUALIFIER)
    offered = situation.offered_tile
    assert offered is not None
    widths = {
        tuple(_t.to_codes(a.tiles)): on._meld_ukeire_copies(situation, offered, a, 2)
        for a in actions
    }
    best = max(widths, key=lambda key: widths[key])
    assert best == ("8t", "9t"), f"进张最宽的吃法应为 8t9t，实测 {widths}"
    assert widths[("8t", "9t")] > widths[("6t", "8t")], widths
