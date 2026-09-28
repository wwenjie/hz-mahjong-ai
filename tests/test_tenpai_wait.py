"""听牌口径修正的回归测试（`wait_aware_tenpai`）。

**这是一条 bug 级发现的修补，不是调参**：`shanten.ukeire` 在 `current == 0` 时
**直接返回空元组**（向听不能再降，`shanten.py:296`），而 `_break_ties_by_ukeire` 把
「每个候选 copies 都是 0」当成平局，于是 `best` 停在 `total` 排序的第一名上——
**听牌时那个「精确进张次排序」什么都没做**，出牌完全由
`total = -10×向听 + 骨架厚度 - 3×喂牌 - 财神罚` 决定，**根本不看听口**。

实测依据（`tools/analyze_wait_ceiling.py`，v2 时代 150 文件 / 1298 个听牌出牌点）：
我们的听口可见张数离同一手牌的上限平均差 **0.96 张（6.7%，分位 27.6%）**，
而对手只差 **0.20 张（1.1%，分位 5.2%）**。

测试要锁住三件事：

1. **`ukeire` 在向听 0 确实返回空**——这是根因，必须直接钉住，否则以后有人「优化」
   掉这条测试就再也没人知道它为什么存在。
2. **修正真的改变行为**（不能像 `preserve-god` 那样是个空干预——那个档位在自对弈里与
   默认档逐位相同，白烧了 2×120 场）。
3. **默认档逐位不变**（v1/v2 已冻结，改默认等于悄悄换冠军）。
"""

from __future__ import annotations

import random

import pytest

from majiang.cli import DECIDERS, make_decider
from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win as win_module
from majiang.rules.action import DISCARD, legal_actions
from majiang.rules.situation import PHASE_DRAW
from majiang.sim import round as round_module
from majiang.strategy import policy as policy_module
from majiang.strategy.policy import (
    VARIANT_FIELDS,
    HeuristicDecider,
    Mode,
    PolicyConfig,
    _DEFAULT_POLICY,
)


def test_knob_is_registered_and_defaults_off() -> None:
    assert PolicyConfig().wait_aware_tenpai is False, "默认必须关，否则等于换了冠军"
    assert "wait_aware_tenpai" in VARIANT_FIELDS, (
        "新开关必须进 VARIANT_FIELDS，否则 configure() 会把它丢掉（踩过一次）"
    )
    assert "tenpai-wait" in DECIDERS and "tenpai-wait-6" in DECIDERS
    decider = make_decider("tenpai-wait", Mode.QUALIFIER)
    assert decider.config.wait_aware_tenpai is True
    assert decider.config.tiebreak == "exact-ukeire", "这条改动只在精确进张口径下有意义"


def _tenpai_situations(seed: int, count: int = 60):
    """随机发牌里挑出「本人摸完牌且已听牌」的局面（14 张、`shanten == 0`）。"""
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        state = round_module.deal(rng, dealer=rng.randrange(4))
        seat = state.turn
        round_module._draw(state, seat)  # noqa: SLF001
        hand = state.seats[seat].hand
        try:
            value = shanten_module.shanten_any(hand, len(state.seats[seat].melds))
        except shanten_module.ShantenError:
            continue
        if value != 0:
            continue
        situation = round_module.situation_for(state, seat, PHASE_DRAW)
        actions = legal_actions(situation)
        if sum(1 for action in actions if action.kind == DISCARD) < 2:
            continue
        produced += 1
        yield situation, actions


def test_ukeire_is_blind_at_tenpai_which_is_the_root_cause() -> None:
    """**根因测试**：向听 0 的 13 张手牌，`ukeire` 对每个牌种都给不出任何进张。

    这条一改（例如有人让 `ukeire` 在向听 0 返回听口），`wait_aware_tenpai` 的理由
    就变了，必须重新审——所以把它钉成测试而不是注释。
    """
    seen = 0
    for situation, _actions in _tenpai_situations(5):
        counts = list(situation.hand.counts)
        for tile in range(tiles.TILE_KINDS):
            if counts[tile] <= 0:
                continue
            after = list(counts)
            after[tile] -= 1
            if shanten_module.shanten_any(after, 0) != 0:
                continue
            assert shanten_module.ukeire(after, 0) == (), (
                "ukeire 在向听 0 竟然返回了非空——听牌口径修正的前提变了"
            )
            seen += 1
    assert seen > 50, f"样本太少（{seen}），测试无效"


def test_wait_aware_uses_real_wait_copies(monkeypatch) -> None:
    """修正版必须真的去数「还能摸到几张」，且默认档一次都不数。"""
    counted: list[int] = []
    real = policy_module.win.winning_draws

    def wrapper(counts, meld_count=0):  # noqa: ANN001, ANN201
        counted.append(1)
        return real(counts, meld_count)

    monkeypatch.setattr(policy_module.win, "winning_draws", wrapper)
    default = HeuristicDecider(PolicyConfig())
    wide = HeuristicDecider(PolicyConfig(wait_aware_tenpai=True))
    default_hits = wide_hits = situations = 0
    for situation, actions in _tenpai_situations(7):
        situations += 1
        counted.clear()
        default.choose(situation, actions, budget_ms=1800)
        default_hits += len(counted)
        counted.clear()
        wide.choose(situation, actions, budget_ms=1800)
        wide_hits += len(counted)
    assert situations >= 20, f"样本太少（{situations}）"
    assert default_hits == 0, (
        f"默认档不该数听口张数（数了 {default_hits} 次）——那说明默认行为被改了"
    )
    assert wide_hits > 0, "修正版一次都没数听口张数，说明这条路没被走到"


def test_it_really_changes_the_choice_somewhere() -> None:
    """**空干预防线**：若两个档位在所有样本上选同一张牌，这条改动就没有意义。

    对应 `preserve-god` 的教训——那个档位在自对弈里与默认档逐座逐位相同，等于白烧算力。
    """
    default = HeuristicDecider(PolicyConfig())
    wide = HeuristicDecider(PolicyConfig(wait_aware_tenpai=True))
    differed = reached = 0
    for situation, actions in _tenpai_situations(23, count=120):
        tied = [
            score
            for score in (
                default._score_discard(situation, a) for a in actions if a.kind == DISCARD
            )
            if score.shanten == 0
        ]
        if len(tied) < 3:
            continue
        reached += 1
        left = default.choose(situation, actions, budget_ms=1800)
        right = wide.choose(situation, actions, budget_ms=1800)
        if left is not None and right is not None and left.tile != right.tile:
            differed += 1
    assert reached >= 10, f"适用样本太少（{reached}），测试无效"
    assert differed > 0, (
        f"{reached} 个适用局面里两个档位选的牌完全一样——这条改动不改变行为，"
        "应记为无效干预而不是去做 A/B"
    )


def test_default_matches_the_frozen_champion_after_every_knob() -> None:
    """`heuristic`（= 冻结的 v2）在默认参数下必须与新开关的默认值一致。"""
    decider = make_decider("heuristic", Mode.QUALIFIER)
    assert decider.config.wait_aware_tenpai == _DEFAULT_POLICY.wait_aware_tenpai
    assert decider.config.tiebreak == _DEFAULT_POLICY.tiebreak


def test_god_tiles_are_still_excluded_only_by_config() -> None:
    """听牌档不得顺手改变财神处理——那属于另一个开关，混进来会让结果无法归因。"""
    wide = PolicyConfig(wait_aware_tenpai=True)
    assert wide.preserve_god == _DEFAULT_POLICY.preserve_god
    assert wide.god_discard_penalty == _DEFAULT_POLICY.god_discard_penalty
    assert wide.ukeire_candidates == _DEFAULT_POLICY.ukeire_candidates
    assert tiles.TILE_KINDS == 34


def test_wait_copies_falls_back_to_the_shipped_wait_set() -> None:
    """`_wait_copies` 与 `winning_draws` 必须同口径（同一个函数，不另写一套听口定义）。"""
    from majiang.strategy.policy import _wait_copies

    seen = 0
    for situation, _actions in _tenpai_situations(31):
        counts = list(situation.hand.counts)
        for tile in range(tiles.TILE_KINDS):
            if counts[tile] <= 0:
                continue
            after = list(counts)
            after[tile] -= 1
            if shanten_module.shanten_any(after, 0) != 0:
                continue
            waits = win_module.winning_draws(after, 0)
            if not waits:
                continue
            got = _wait_copies(after, 0, counts, tile)
            assert got is not None
            # visible 缺省只用本手牌：这里传的是 14 张手牌，故每个听口至少还剩
            # (4 − 本手持有量) 张，且必须为正
            assert got > 0, "听口张数为 0 说明口径写反了"
            seen += 1
    assert seen > 20, f"样本太少（{seen}）"
