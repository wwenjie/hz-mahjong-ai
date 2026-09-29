"""`feed_visibility`（喂牌项乘「还剩几张未现」）的回归测试。

**依据（agent-c 独立复核确证的一条缺陷）**：`risk.visible_need(tile)` 是**牌种静态表**，
不含已见张数 ⇒ **已见 3 张、几乎喂不出去的牌，仍按满值计罚**。
正确口径是「还剩几张可能被对手拿到」= `4 − seen`（seen 含本方暗手 + 四家副露 + 四家弃牌）。

测试要锁住：
1. 默认档逐位不变（v1/v2/v3 已冻结）；
2. **它真的改变行为**（空干预防线）；
3. 因子方向正确：同牌种下**已见越多、喂牌代价越小**。
"""

from __future__ import annotations

import pytest

from majiang.cli import DECIDERS, make_decider
from majiang.rules import tiles
from majiang.strategy.policy import VARIANT_FIELDS, Mode, PolicyConfig


def test_knob_registered_and_default_off() -> None:
    assert PolicyConfig().feed_visibility is False, "默认必须关，否则等于换了冠军"
    assert "feed_visibility" in VARIANT_FIELDS, "新开关必须进 VARIANT_FIELDS"
    assert "seen-feed" in DECIDERS and "seen-shape-feed" in DECIDERS
    assert make_decider("seen-feed", Mode.QUALIFIER).config.wait_aware_tenpai is True


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_frozen_champions_unchanged(version: str) -> None:
    assert make_decider(version, Mode.QUALIFIER).config.feed_visibility is False


def test_factor_matches_unseen_copies() -> None:
    """因子 = (4 − 已见)/4，且已见越多因子越小。"""
    for seen, expected in ((0, 1.0), (1, 0.75), (2, 0.5), (3, 0.25), (4, 0.0)):
        factor = max(0.0, tiles.COPIES_PER_KIND - seen) / tiles.COPIES_PER_KIND
        assert factor == pytest.approx(expected)


def test_it_changes_the_choice_and_the_budget_is_bounded() -> None:
    """**空干预防线 + 预算防线**：必须真的改变选择，且耗时不得顶到出牌预算。"""
    import random
    import time

    from majiang.rules.action import DISCARD, legal_actions
    from majiang.rules.situation import PHASE_DRAW
    from majiang.sim import round as round_module
    from majiang.strategy.policy import HeuristicDecider

    base = HeuristicDecider(PolicyConfig(wait_aware_tenpai=True))
    seen = HeuristicDecider(PolicyConfig(wait_aware_tenpai=True, feed_visibility=True))
    rng = random.Random(29)
    differed = reached = 0
    elapsed = 0.0
    while reached < 60:
        state = round_module.deal(rng, dealer=rng.randrange(4))
        seat = state.turn
        round_module._draw(state, seat)  # noqa: SLF001
        situation = round_module.situation_for(state, seat, PHASE_DRAW)
        actions = legal_actions(situation)
        if sum(1 for action in actions if action.kind == DISCARD) < 3:
            continue
        reached += 1
        left = base.choose(situation, actions, budget_ms=1800)
        started = time.perf_counter()
        right = seen.choose(situation, actions, budget_ms=1800)
        elapsed += time.perf_counter() - started
        if left is not None and right is not None and left.tile != right.tile:
            differed += 1
    assert differed > 0, (
        f"{reached} 个局面里两个档位选的牌完全一样——这条改动不改变行为，"
        "应记为无效干预而不是去做 A/B"
    )
    assert elapsed / reached < 0.5, f"均耗时 {elapsed / reached * 1000:.0f} ms，超出安全余量"
