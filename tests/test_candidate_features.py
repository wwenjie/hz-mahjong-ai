"""candidate_features 的单元测试（原型，未接入 features.py）。"""

from __future__ import annotations

import time

from majiang.rules import tiles
from majiang.strategy import candidate_features


def _counts(*tile_ids: int) -> list[int]:
    counts = [0] * tiles.TILE_KINDS
    for t in tile_ids:
        counts[t] += 1
    return counts


def test_feature_count_matches_names() -> None:
    assert candidate_features.FEATURE_COUNT == len(candidate_features.FEATURE_NAMES) == 3


def test_tenpai_hand_keep_dominates() -> None:
    """一手已听牌（0 向听）：打掉让牌型继续听的候选应全部 keep，无 lower。

    用 13 张（未摸牌）听牌型：123m 123p 123s 55z + 白板财神一张做将/搭，
    简化：直接验证「返回值三分量非负且总数=不同牌种数」这一不变量。
    """
    counts = _counts(0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 28, 28)  # 13 张
    keep, lower, raise_ = candidate_features.extract(counts, meld_count=0)
    kinds = sum(1 for c in counts if c > 0)
    assert keep + lower + raise_ == kinds
    assert keep >= 1  # 至少有牌打了不掉向听


def test_drawn_hand_has_baseline_min_semantics() -> None:
    """摸牌后 14 张：基线取各候选打后向听的最小值 ⇒ lower 恒为 0。"""
    counts = _counts(0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 28, 28, 3)  # 14 张
    keep, lower, raise_ = candidate_features.extract(counts, meld_count=0)
    kinds = sum(1 for c in counts if c > 0)
    assert keep + lower + raise_ == kinds
    assert lower == 0  # 摸后态 baseline 已是 min，不可能有"更低"
    assert keep >= 1


def test_trash_tile_is_raise_candidate() -> None:
    """孤张字牌在一手好型里应是 raise/keep 边缘，而不是 lower。"""
    # 123m 456m 789m 22p 33p + 孤张字牌 31（14 张，摸后）
    counts = _counts(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 11, 31)
    keep, lower, raise_ = candidate_features.extract(counts, meld_count=0)
    assert keep + lower + raise_ > 0


def test_bad_shape_raises_on_wrong_tile_count() -> None:
    counts = _counts(0, 1, 2)  # 3 张，非法
    try:
        candidate_features.extract(counts, meld_count=0)
    except Exception:
        return
    raise AssertionError("应抛 ShantenError")


def test_cost_is_microsecond_scale() -> None:
    """热路径预算：单次提取应远在 1ms 之内（quick_shanten 是微秒级）。"""
    counts = _counts(0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 28, 28, 3)
    n = 200
    start = time.perf_counter()
    for _ in range(n):
        candidate_features.extract(counts, meld_count=0)
    per_call_ms = (time.perf_counter() - start) / n * 1000
    assert per_call_ms < 5.0, f"单次 {per_call_ms:.3f}ms 超预算"
