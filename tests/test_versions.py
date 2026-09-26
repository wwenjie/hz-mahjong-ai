"""版本库的回归测试（`src/majiang/strategy/versions.py`）。

重点不是「版本能构造」，而是**纪律**：每次改默认档都必须新增快照，否则旧默认档的行为
会永久丢失（`blocks` 那次只能靠留一个近似档位补救）。
"""

from __future__ import annotations

from majiang.cli import DECIDERS, make_decider
from majiang.strategy import versions
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig


def test_version_ids_are_unique_and_addressable() -> None:
    ids = [version.id for version in versions.VERSIONS]
    assert len(ids) == len(set(ids)), "版本号必须唯一"
    for version_id in ids:
        assert versions.is_version(version_id)
        assert make_decider(version_id, Mode.QUALIFIER) is not None
    assert not versions.is_version("v999")


def test_v1_is_the_pre_change_baseline_and_differs_from_current_default() -> None:
    """v1 必须真的等于「改动前」的行为，且与当前默认**不同**——否则它没有对照价值。"""
    old = versions.build("v1", Mode.QUALIFIER)
    now = HeuristicDecider(PolicyConfig())
    assert old.config.tiebreak == "blocks"
    assert now.config.tiebreak == "exact-ukeire"
    assert old.config != now.config


def test_every_version_builds_in_both_modes() -> None:
    for version in versions.VERSIONS:
        for mode in (Mode.QUALIFIER, Mode.FINAL):
            decider = make_decider(version.id, mode)
            assert isinstance(decider, HeuristicDecider)


def test_versions_override_same_named_declared_deciders() -> None:
    """版本库优先于 DECIDERS：同名时以「已胜出并冻结」的版本为准。"""
    assert versions.is_version("v2")
    # `heuristic` 走 DECIDERS，`v2` 走版本库，但两者当前应等价（v2 就是当前默认）
    assert make_decider("v2", Mode.QUALIFIER).config == make_decider(
        "heuristic", Mode.QUALIFIER
    ).config
    assert "v1" not in DECIDERS
