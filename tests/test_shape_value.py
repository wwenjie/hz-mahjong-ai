"""`shape_value`（加权形质值）的回归测试。

**为什么单独测**：这条改动修的是一处**度量退化**——`quick_blocks` 的裁剪恒饱和，
导致同听候选之间 77~93% 完全并列，次排序退化成「只看喂牌」。
所以测试必须锁住的是**分辨力本身**，而不是「某个具体手牌算得对不对」：

1. **分辨力**：同一批真机式手牌上，全并列占比必须显著下降（否则等于空干预）；
2. **不越界**：形质值极小化后仍 < 10，保证 `-10×向听` 依旧是主导项
   （即「只打破并列、不覆盖块数差 1、不改变向听优先」）；
3. **默认档逐位不变**（v1/v2/v3 已冻结）。
"""

from __future__ import annotations

import random
import statistics
import time

import pytest

from majiang.cli import DECIDERS, make_decider
from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.strategy.policy import (
    VARIANT_FIELDS,
    HeuristicDecider,
    Mode,
    PolicyConfig,
)


def _hand(*codes: str) -> list[int]:
    counts = [0] * tiles.TILE_KINDS
    for code in codes:
        counts[tiles.parse(code)] += 1
    return counts


def test_discriminates_ryanmen_from_kanchan_where_old_metric_ties() -> None:
    """旧口径给两面和坎张**同一个分**，新口径必须分开。"""
    ryanmen = _hand("1w", "2w", "5b", "6b", "3t", "4t", "东", "南")
    kanchan = _hand("1w", "3w", "5b", "7b", "3t", "5t", "东", "南")
    old_a = 2 * shanten_module.quick_blocks(ryanmen)[0] + shanten_module.quick_blocks(ryanmen)[1]
    old_b = 2 * shanten_module.quick_blocks(kanchan)[0] + shanten_module.quick_blocks(kanchan)[1]
    new_a = shanten_module.shape_value(ryanmen, 0)
    new_b = shanten_module.shape_value(kanchan, 0)
    assert old_a == old_b, "旧口径应当并列（这正是要修的现象）"
    assert new_a > new_b, "两面手的形质值应高于坎张手"


def test_shape_value_stays_below_shanten_weight() -> None:
    """**不越界**：形质值必须 < 10（`shanten_weight`），否则会覆盖向听优先。"""
    rng = random.Random(7)
    worst = 0.0
    for _ in range(400):
        state = round_module_deal(rng)
        hand = list(state.seats[0].hand)
        for melds in (0, 1, 2):
            worst = max(worst, shanten_module.shape_value(hand, melds))
    assert worst < 10.0, f"形质值上界 {worst:.2f} 已达 10，会覆盖 -10×向听 的主导地位"
    assert worst > 4.0, f"形质值上界只有 {worst:.2f}，分辨范围过窄"


def round_module_deal(rng):
    from majiang.sim import round as round_module

    return round_module.deal(rng, dealer=0)


def _tie_rate(hands: list[list[int]], melds: int, *, new: bool) -> tuple[float, float]:
    """返回 (全并列占比, 平均不同值数)。候选集＝该手牌所有「同最小向听」的打法。"""
    flat = 0
    distinct_sum = 0.0
    total = 0
    for hand in hands:
        if sum(hand) % 3 != 2:
            continue
        best = 99
        rows = []
        for drop in range(tiles.TILE_KINDS):
            if hand[drop] <= 0:
                continue
            after = list(hand)
            after[drop] -= 1
            try:
                value = shanten_module.shanten_any(after, melds)
            except Exception:  # noqa: BLE001
                continue
            best = min(best, value)
            rows.append(after)
        picked = [
            (shanten_module.shape_value(after, melds) if new
             else 2 * shanten_module.quick_blocks(after)[0]
             + shanten_module.quick_blocks(after)[1])
            for after in rows
            if shanten_module.shanten_any(after, melds) == best
        ]
        if len(picked) < 2:
            continue
        total += 1
        distinct_sum += len(set(picked))
        if len(set(picked)) == 1:
            flat += 1
    return (flat / total if total else 0.0, distinct_sum / total if total else 0.0)


def _midgame_hands(count: int = 90) -> list[list[int]]:
    """从中局采样手牌——**不能用随机发牌**。

    第一版测试就是拿随机发牌手牌算并列率的，得到 0% 并列（前提不成立）。
    原因是：刚发的 13 张是散的、块少，`min(partials, 4-sets)` 根本不会饱和；
    **饱和是中局现象**（手牌结构化了、块多到 4 个以上）。所以这里跑自对弈取真实中局局面。
    """
    import random as _random

    from majiang.sim import round as round_module

    rng = _random.Random(23)
    hands: list[list[int]] = []
    while len(hands) < count:
        deciders = [_HeuristicForTest() for _ in range(4)]
        outcome = round_module.run_round(deciders, dealer=0, round_no=1, base_score=1, rng=rng)
        if outcome is None:  # pragma: no cover - 防御
            continue
        hands.extend(_HeuristicForTest.last_hands[0:6])
        _HeuristicForTest.last_hands = []
    return hands[:count]


class _HeuristicForTest(HeuristicDecider):
    """在真实对局里顺手记录「我方 14 张局面」的决策器。"""

    last_hands: list[list[int]] = []

    def __init__(self) -> None:
        super().__init__(PolicyConfig())

    def choose(self, situation, actions, *, budget_ms: int = 0):  # noqa: ANN001, ANN201
        counts = list(situation.hand.counts)
        if sum(counts) % 3 == 2:
            _HeuristicForTest.last_hands.append(counts)
        return super().choose(situation, actions, budget_ms=budget_ms)


def test_resolution_improves_on_realistic_hands() -> None:
    """**分辨力防线**：全并列占比必须显著下降——否则这条改动是空干预。"""
    hands = _midgame_hands(90)
    assert len(hands) >= 60, f"中局手牌只采到 {len(hands)} 副，测试无效"
    old_flat, old_distinct = _tie_rate(hands, 0, new=False)
    new_flat, new_distinct = _tie_rate(hands, 0, new=True)
    assert old_flat > 0.55, f"旧口径中局并列率只有 {old_flat:.1%}，前提不成立"
    assert new_flat < old_flat * 0.6, (
        f"全并列占比 {old_flat:.1%} -> {new_flat:.1%} 降幅不足——这条改动没解决退化"
    )
    assert new_distinct > old_distinct + 0.3, (
        f"平均不同值数 {old_distinct:.2f} -> {new_distinct:.2f} 提升不足"
    )


def test_shape_value_is_fast_enough() -> None:
    """热路径上的成本必须仍是微秒级（它是每个候选都要算的）。"""
    rng = random.Random(13)
    hands = [list(round_module_deal(rng).seats[0].hand) for _ in range(200)]
    started = time.perf_counter()
    for hand in hands:
        for drop in range(tiles.TILE_KINDS):
            if hand[drop] <= 0:
                continue
            after = list(hand)
            after[drop] -= 1
            shanten_module.shape_value(after, 0)
    per_call = (time.perf_counter() - started) / (len(hands) * 10) * 1e6
    assert per_call < 200, f"单次 {per_call:.0f} µs，超出热路径预算"


def test_knob_registered_and_default_off() -> None:
    assert PolicyConfig().shape_value is False, "默认必须关，否则等于换了冠军"
    assert "shape_value" in VARIANT_FIELDS, "新开关必须进 VARIANT_FIELDS（configure 会丢字段）"
    assert "shape" in DECIDERS and "shape-only" in DECIDERS
    assert make_decider("shape", Mode.QUALIFIER).config.wait_aware_tenpai is True
    assert make_decider("shape-only", Mode.QUALIFIER).config.wait_aware_tenpai is False


def test_default_never_calls_shape_value(monkeypatch) -> None:
    """默认档一次都不该调用新函数。"""
    from majiang.strategy import policy as policy_module

    calls: list[int] = []
    real = policy_module.shanten_module.shape_value

    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(policy_module.shanten_module, "shape_value", wrapper)
    from majiang.rules.action import DISCARD, legal_actions
    from majiang.rules.situation import PHASE_DRAW
    from majiang.sim import round as round_module

    decider = HeuristicDecider(PolicyConfig())
    rng = random.Random(3)
    seen = 0
    while seen < 25:
        state = round_module.deal(rng, dealer=rng.randrange(4))
        seat = state.turn
        round_module._draw(state, seat)  # noqa: SLF001
        situation = round_module.situation_for(state, seat, PHASE_DRAW)
        actions = legal_actions(situation)
        if sum(1 for a in actions if a.kind == DISCARD) < 2:
            continue
        decider.choose(situation, actions, budget_ms=1800)
        seen += 1
    assert calls == [], f"默认档调用了 {len(calls)} 次 shape_value，默认行为被改了"


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_frozen_champions_unchanged(version: str) -> None:
    assert make_decider(version, Mode.QUALIFIER).config.shape_value is False
