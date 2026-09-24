"""性能回归护栏（tasks.md 2.15）。

阈值取得比实测值宽一个数量级，只用于拦住灾难性退化：实测如下（Python 3.12，
单核，本机）：

- ``is_winning_shape``（14 张随机）约 21 µs
- ``is_baotou``（13 张随机，绝大多数在第一次判定就失败）约 18 µs
- ``is_baotou`` / ``compute_fan``（真听任意，需跑满 34 次完整搜索）约 2.1 ms
- ``compute_fan``（普通听牌）约 41 µs

结论：决策路径上单次判定都在毫秒级以内，3 秒决策预算、同时 M 场并发都够用；
但约 2 ms 的最坏情形不适合放进前瞻搜索的 rollout 内层，那里需要单独的快速通道。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from majiang.rules import fan, tiles, win

from .helpers import counts_of

DECISION_PATH_BUDGET_MS = 20.0


def _per_call_ms(function: Callable[[list[int]], object], data: Iterable[list[int]]) -> float:
    items = list(data)
    start = time.perf_counter()
    for item in items:
        function(item)
    return (time.perf_counter() - start) / len(items) * 1000.0


def test_baotou_worst_case_stays_within_decision_budget() -> None:
    baotou_hand = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b白")
    assert win.is_baotou(baotou_hand, 0) is True
    elapsed = _per_call_ms(lambda hand: win.is_baotou(hand, 0), [baotou_hand] * 200)
    assert elapsed < DECISION_PATH_BUDGET_MS


def test_compute_fan_worst_case_stays_within_decision_budget() -> None:
    baotou_hand = counts_of("1w2w3w4w5w6w7w8w9w1b2b3b白")
    draw = tiles.parse("5b")
    elapsed = _per_call_ms(lambda hand: fan.compute_fan(hand, draw), [baotou_hand] * 200)
    assert elapsed < DECISION_PATH_BUDGET_MS


def test_winning_shape_stays_far_below_decision_budget() -> None:
    drawn = counts_of("1w1w1w1w2w2w2w2w3w3w3w3w4w4w")
    elapsed = _per_call_ms(lambda hand: win.is_winning_shape(hand, 0), [drawn] * 2000)
    assert elapsed < DECISION_PATH_BUDGET_MS / 10
