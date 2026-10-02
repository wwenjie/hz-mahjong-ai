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


def test_v4_is_bit_identical_to_the_arm_that_passed_the_ab() -> None:
    """**换档的承重保证**：v4 必须与跑过 A/B 的那个档位（`shape-blocks`）在
    **每一个变体开关**上逐位一致——否则「n=10、t+4.66」那份证据就挂在别的档位上了。

    `VARIANT_FIELDS` 是「会改变打法、且必须在 `configure()` 里保留」的开关清单
    （`policy.py` 的 docstring 记着它的来历：漏一个字段就会让真机上所有档位静默跑成默认档）。
    这里只比这份清单——`base_score`/`you_cai_bi_kao` 由服务端注入，**刻意不冻结**，
    比它们会让版本在真机上与服务端实际规则不符。
    """
    from majiang.strategy.policy import VARIANT_FIELDS

    frozen = versions.build("v4", Mode.QUALIFIER).config
    tested = make_decider("shape-blocks", Mode.QUALIFIER).config
    for field in VARIANT_FIELDS:
        assert getattr(frozen, field) == getattr(tested, field), (
            f"v4 与 shape-blocks 在 {field} 上不一致："
            f"{getattr(frozen, field)!r} vs {getattr(tested, field)!r}"
        )


def test_v3_is_still_addressable_after_v4_was_added() -> None:
    """**保留 v3**：加 v4 不能动到 v3 的开关，且任意两版仍可直接对打（本文件的存在理由）。"""
    assert versions.is_version("v3")
    v3 = versions.build("v3", Mode.QUALIFIER).config
    assert v3.wait_aware_tenpai is True
    assert v3.shape_value is False, "v3 的开关不得被 v4 的改动污染"
    assert v3.ukeire_max_shanten == PolicyConfig().ukeire_max_shanten
    # 两版确实不同（否则这次的「换档」是空的）
    assert versions.build("v4", Mode.QUALIFIER).config != v3


def test_v5_is_bit_identical_to_the_arm_that_measured_the_marginal() -> None:
    """v5 必须与跑出边际的那个档位（`v4-cand3`）逐位一致——否则「配对 t4.18」的证据就挂空了。"""
    from majiang.strategy.policy import VARIANT_FIELDS

    frozen = versions.build("v5", Mode.QUALIFIER).config
    tested = make_decider("v4-cand3", Mode.QUALIFIER).config
    for field in VARIANT_FIELDS:
        assert getattr(frozen, field) == getattr(tested, field), field
    # v4 仍然可达且未被 v5 的改动污染（只是多了一个候选面开关）
    v4 = versions.build("v4", Mode.QUALIFIER).config
    assert v4.ukeire_candidates == PolicyConfig().ukeire_candidates
    assert frozen.ukeire_candidates == 3


def test_v6_is_bit_identical_to_the_arm_that_produced_the_additivity_result() -> None:
    """v6 必须与跑出 +0.734 的 `v6a` 逐字段一致——否则「可加性」的证据就挂在空处。"""
    from majiang.strategy.policy import VARIANT_FIELDS

    frozen = versions.build("v6", Mode.QUALIFIER).config
    tested = make_decider("v6a", Mode.QUALIFIER).config
    for field in VARIANT_FIELDS:
        assert getattr(frozen, field) == getattr(tested, field), field
    # v5 仍可达且未被污染（v6 只是在它之上多两个开关）
    v5 = versions.build("v5", Mode.QUALIFIER).config
    assert v5.ukeire_preselect == 0 and v5.piao_threshold_scale == 1.0
    assert frozen.ukeire_preselect == 5 and frozen.piao_threshold_scale == 1.3
