"""两拍档（`two_ply_shanten1`）的回归测试。

**为什么单独测**：这条改动动的是 `_break_ties_by_ukeire` 的核心排序，而且它的收益是
「灰区」（同位置 regret 11.6%，落在 10–25%），所以**不能靠 A/B 正号来判它该不该留**——
要先由机制量（自对弈实际听口宽度）判。测试要锁住三件事：

1. **默认档逐位不变**（v1/v2/v3 都已冻结，改默认等于悄悄换冠军）；
2. **只在向听 1 生效**（向听 0 走 `wait_aware_tenpai`，向听 ≥2 不该被它碰到）；
3. **它真的改变行为**（`preserve-god` 那种空干预的教训）。
"""

from __future__ import annotations

import random

import pytest

from majiang.cli import DECIDERS, make_decider
from majiang.rules import shanten as shanten_module
from majiang.rules.action import DISCARD, legal_actions
from majiang.rules.situation import PHASE_DRAW
from majiang.sim import round as round_module
from majiang.strategy.policy import (
    VARIANT_FIELDS,
    HeuristicDecider,
    Mode,
    PolicyConfig,
    _DEFAULT_POLICY,
)


def test_knob_registered_and_default_off() -> None:
    assert PolicyConfig().two_ply_shanten1 is False, "默认必须关，否则等于换了冠军"
    assert "two_ply_shanten1" in VARIANT_FIELDS, (
        "新开关必须进 VARIANT_FIELDS，否则 configure() 会把它丢掉（踩过一次）"
    )
    assert "two-ply" in DECIDERS and "two-ply-only" in DECIDERS
    assert make_decider("two-ply", Mode.QUALIFIER).config.wait_aware_tenpai is True
    assert make_decider("two-ply-only", Mode.QUALIFIER).config.wait_aware_tenpai is False


def _situations(seed: int, want_shanten: int, count: int = 40):
    """随机发牌里挑出「本人摸完牌后，**打完所有候选里的最小向听**恰好等于目标」的局面。

    注意不能只筛「存在某张打完是目标向听」——决策器用的是**最小**向听（`top_shanten`），
    若另一张能打到更低的向听，两拍就不该触发。第一版测试就是这么写错的，
    它报出「向听 2 却触发了 3 次」，查下来是筛选太松而不是代码有问题。
    """
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        state = round_module.deal(rng, dealer=rng.randrange(4))
        seat = state.turn
        round_module._draw(state, seat)  # noqa: SLF001
        hand = list(state.seats[seat].hand)
        values = []
        for tile in range(34):
            if hand[tile] <= 0:
                continue
            after = list(hand)
            after[tile] -= 1
            try:
                values.append(shanten_module.shanten_any(after, 0))
            except Exception:  # noqa: BLE001
                continue
        if not values or min(values) != want_shanten:
            continue
        situation = round_module.situation_for(state, seat, PHASE_DRAW)
        actions = legal_actions(situation)
        if sum(1 for a in actions if a.kind == DISCARD) < 3:
            continue
        produced += 1
        yield situation, actions


def test_default_never_calls_two_ply(monkeypatch) -> None:
    """默认档一次都不该算两拍值。"""
    from majiang.strategy import policy as policy_module

    calls: list[int] = []
    real = policy_module._two_ply_value

    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(policy_module, "_two_ply_value", wrapper)
    default = HeuristicDecider(PolicyConfig())
    seen = 0
    for situation, actions in _situations(3, 1):
        default.choose(situation, actions, budget_ms=1800)
        seen += 1
    assert seen >= 20, f"样本太少（{seen}）"
    assert calls == [], f"默认档调用了 {len(calls)} 次两拍——默认行为被改了"


def test_two_ply_fires_only_at_shanten_one(monkeypatch) -> None:
    """只在向听 1 调用；向听 0 与向听 2 都不该触发。"""
    from majiang.strategy import policy as policy_module

    calls: list[int] = []
    real = policy_module._two_ply_value

    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(policy_module, "_two_ply_value", wrapper)
    decider = HeuristicDecider(PolicyConfig(two_ply_shanten1=True))
    for want in (0, 1, 2):
        calls.clear()
        seen = 0
        for situation, actions in _situations(5 + want, want, count=25):
            decider.choose(situation, actions, budget_ms=1800)
            seen += 1
        assert seen >= 15, f"向听 {want} 的样本太少（{seen}）"
        if want == 1:
            assert calls, "向听 1 一次都没算两拍，说明这条路没被走到"
        else:
            assert not calls, f"向听 {want} 不该算两拍，却算了 {len(calls)} 次"


def test_two_ply_changes_the_choice_and_fits_the_budget() -> None:
    """**空干预防线 + 预算防线**：它必须真的改变选择，且耗时不得顶到出牌预算。"""
    import time

    base = HeuristicDecider(PolicyConfig(wait_aware_tenpai=True))
    two = HeuristicDecider(PolicyConfig(wait_aware_tenpai=True, two_ply_shanten1=True))
    differed = seen = 0
    elapsed = 0.0
    for situation, actions in _situations(23, 1, count=40):
        seen += 1
        left = base.choose(situation, actions, budget_ms=1800)
        started = time.perf_counter()
        right = two.choose(situation, actions, budget_ms=1800)
        elapsed += time.perf_counter() - started
        if left is not None and right is not None and left.tile != right.tile:
            differed += 1
    assert seen >= 15, f"适用样本太少（{seen}），测试无效"
    assert differed > 0, (
        f"{seen} 个向听 1 局面里两个档位选的牌完全一样——这条改动不改变行为，"
        "应记为无效干预而不是去做 A/B"
    )
    # 实测约 150 ms/候选；单次决策不宜超过 1 秒（出牌预算 1800 ms，平台超时 3 s）
    assert elapsed / seen < 1.0, f"两拍决策均耗时 {elapsed / seen * 1000:.0f} ms，超出安全余量"


def test_default_matches_frozen_champions() -> None:
    for version in ("v1", "v2", "v3"):
        decider = make_decider(version, Mode.QUALIFIER)
        assert decider.config.two_ply_shanten1 is False, f"{version} 行为被改动了"


@pytest.mark.parametrize("knob", ["wait_aware_tenpai", "two_ply_shanten1"])
def test_knobs_are_independent(knob: str) -> None:
    other = "two_ply_shanten1" if knob == "wait_aware_tenpai" else "wait_aware_tenpai"
    config = PolicyConfig(**{knob: True})  # type: ignore[arg-type]
    assert getattr(config, knob) is True
    assert getattr(config, other) is _DEFAULT_POLICY.__getattribute__(other)
