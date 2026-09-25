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
    HeuristicDecider,
    MeldTolerance,
    Mode,
    PolicyConfig,
)


def test_default_is_strict_and_for_mode_keeps_the_string_comparable() -> None:
    assert PolicyConfig().meld_tolerance is MeldTolerance.STRICT
    decider = make_decider("meld-equal", Mode.QUALIFIER)
    assert decider.config.meld_tolerance == MeldTolerance.EQUAL, (
        "for_mode 传入普通 str；若闸门用 `is` 比较就会静默退化成 strict"
    )


def test_meld_equal_is_registered() -> None:
    assert "meld-equal" in DECIDERS


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
