"""`candidate_features` 公共构造器的单元测试。

背景：该模块在 `94d5b1a`（A 2026-10-06 03:41 裁决）从「原型 extract(counts, meld_count)」
收敛为**唯一**的候选字段构造入口 `candidate_dicts` / `candidate_features(decider, situation,
candidates)`（训练导出器与 `botlike` 推理共用，消除 train-serve skew）。旧测试仍按已删除的
`extract` / `FEATURE_COUNT` / `FEATURE_NAMES` 写，5 项全部 AttributeError——
**改前改后同结果，与本轮改动无关**（A 2026-10-08 22:30 回归立据，指派 B' 决定删/补）。

本文件按现 API 重写，锁定的是**新公共口径**（B' 2026-10-08 22:35 补 API，非删测试）：

- 返回 `(rows, kept)`，`kept` 与 `candidates` 中 `tile is not None` 者一一对应；
- 每个 row = 29 维局面特征 + 5 个候选字段；`None` 语义在 `candidate_dicts` 保留、在
  `candidate_features` 里落成 0.0；
- 主字段 `main_total`/`shanten` 恒有；`wait_*` 只在**并列层且听牌态**计算，
  `ukeire_exact` 只在**并列层非听牌**计算（其余为 None）——这是口径的关键不变量。
"""
from __future__ import annotations

import time

from majiang.rules import tiles
from majiang.strategy import candidate_features, features
from majiang.strategy.policy import HeuristicDecider

from tests.test_policy import situation


def _candidates(situation_obj):
    """把全部可弃牌做成 DISCARD 候选（与 `ab_test`/对局中的合法弃牌同构）。"""
    from majiang.rules.action import DISCARD, Action

    return tuple(
        Action(DISCARD, tile=tile)
        for tile, amount in enumerate(situation_obj.hand.counts)
        if amount
    )


def _decider() -> HeuristicDecider:
    return HeuristicDecider()


def _fields(situation_obj):
    return candidate_features.candidate_dicts(
        _decider(), situation_obj, _candidates(situation_obj)
    )


def test_rows_align_with_kept_actions() -> None:
    """rows 与 kept 一一对应；row 宽度 = 29 维局面特征 + 5 个候选字段。"""
    obj = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b")
    rows, kept = candidate_features.candidate_features(
        _decider(), obj, _candidates(obj)
    )
    assert len(rows) == len(kept) == len(_candidates(obj))
    n_situation = len(features.extract(obj))
    assert all(len(row) == n_situation + 5 for row in rows), (n_situation, len(rows[0]))


def test_main_fields_always_present() -> None:
    """每个候选都有 tile/main_total/shanten；数字有限。"""
    obj = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b")
    fields, kept = _fields(obj)
    assert len(fields) == len(kept)
    for field in fields:
        assert field["tile"] is not None
        assert isinstance(field["main_total"], float)
        assert isinstance(field["shanten"], float)


def test_ukeire_only_on_non_tenpai_tie() -> None:
    """非听牌并列层：`ukeire_exact` 计算、`wait_copies`/`wait_kinds` 为 None。

    构造一手向听 2–3 的散牌（多并列候选、非听牌），断言听口类字段留 None 而进张字段被填。
    """
    obj = situation("1w2w2w3w3w6w7w9w2b5b5b6b7b9b", drawn="9b")
    fields, _ = _fields(obj)
    assert fields
    assert any(f["ukeire_exact"] is not None for f in fields), "非听牌并列层应算 ukeire"
    assert all(f["wait_copies"] is None for f in fields)
    assert all(f["wait_kinds"] is None for f in fields)


def test_tenpai_tie_uses_wait_fields() -> None:
    """听牌并列层：`wait_copies`/`wait_kinds` 计算、`ukeire_exact` 为 None。"""
    # 已听牌（0 向听）并列型：多张打后仍听 ⇒ wait_aware=True，走 wait_* 分支
    obj = situation("2w2w3w3w6w6w6w3b3b5b5b5b7b7b", drawn="7b")
    fields, _ = _fields(obj)
    assert fields
    top = max(f["main_total"] for f in fields)
    tied = [f for f in fields if f["main_total"] == top and f["shanten"] == 0]
    assert len(tied) >= 2, "该构造应为听牌并列层"
    assert any(f["wait_copies"] is not None for f in tied)
    assert any(f["wait_kinds"] is not None for f in tied)
    assert all(f["ukeire_exact"] is None for f in tied)


def test_cost_is_microsecond_scale() -> None:
    """热路径预算：单次构造 14 张 12 候选应远在 1ms 量级（quick_shanten 微秒级）。"""
    obj = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b")
    decider = _decider()
    candidates = _candidates(obj)
    n = 50
    start = time.perf_counter()
    for _ in range(n):
        candidate_features.candidate_features(decider, obj, candidates)
    per_call_ms = (time.perf_counter() - start) / n * 1000
    assert per_call_ms < 50.0, f"单次 {per_call_ms:.3f}ms 超预算"


def test_empty_candidates_returns_empty() -> None:
    """无候选 ⇒ 空返回，不抛。"""
    obj = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b")
    rows, kept = candidate_features.candidate_features(_decider(), obj, ())
    assert rows == [] and kept == []
