"""S3 局况姿态的极性与不变量测试（`PolicyConfig.standing_*`，A 2026-10-04）。

**为什么先钉这两条**：`standing_*` 的三个系数（1.4 / 0.7 / 阈值 29 分 / 余局 1）里，
「1.4 与 0.7」是**拍**的初值 ⇒ 如果连**极性**（落后更敢搏、领先更保守）都没出现，
说明接入根本没生效（仪表问题），而不是「效果不够」。而极性可测、**不受 n 小的影响**
（不像「每场名次分」这类重尾指标）。第二条钉**安全兜底**：拿不到比分/并列/非边界态 ⇒ 必须**逐位等于 v5**。
"""

from __future__ import annotations

from types import SimpleNamespace

from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig


def _stub(scores, *, me=0, rounds_total=8, round_no=8):
    table = SimpleNamespace(scores=tuple(scores), rounds_total=rounds_total, round_no=round_no)
    return SimpleNamespace(table=table, seat=me)


def _decider(lead=1.4, behind=0.7):
    return HeuristicDecider(
        PolicyConfig.for_mode(
            Mode.QUALIFIER, standing_lead_scale=lead, standing_behind_scale=behind
        )
    )


def test_standing_scale_polarity_and_neutrality() -> None:
    d = _decider()
    # 领先（名次 1）且分差可竞争 ⇒ 守（>1）
    assert d._standing_scale(_stub((100, 90, 80, 70), me=0)) == 1.4
    # 落后（名次 4）且分差可竞争 ⇒ 搏（<1）
    assert d._standing_scale(_stub((100, 90, 80, 70), me=3)) == 0.7
    # 名次 3 ⇒ 搏
    assert d._standing_scale(_stub((100, 90, 80, 70), me=2)) == 0.7
    # 名次 2 ⇒ 中性（不在边界定义里）
    assert d._standing_scale(_stub((100, 90, 80, 70), me=1)) == 1.0
    # 分差 > 阈值 29 ⇒ 中性（不可竞争）
    assert d._standing_scale(_stub((100, 60, 50, 40), me=0)) == 1.0
    assert d._standing_scale(_stub((100, 90, 80, 40), me=3)) == 1.0
    # 余局 > 1 ⇒ 中性
    assert d._standing_scale(_stub((100, 90, 80, 70), me=0, round_no=7)) == 1.0
    # 并列（分差 0）⇒ 中性
    assert d._standing_scale(_stub((100, 100, 80, 70), me=0)) == 1.0
    assert d._standing_scale(_stub((100, 100, 80, 70), me=1)) == 1.0
    # 拿不到比分 / 赛制未知 ⇒ 中性（真机安全兜底）
    assert d._standing_scale(_stub((), me=0)) == 1.0
    assert d._standing_scale(_stub((100, 90, 80, 70), me=0, rounds_total=0)) == 1.0
    # 关闭（默认 1.0/1.0）⇒ 恒中性
    off = _decider(lead=1.0, behind=1.0)
    assert off._standing_scale(_stub((100, 90, 80, 70), me=0)) == 1.0
    assert off._standing_scale(_stub((100, 90, 80, 70), me=3)) == 1.0


def test_standing_defaults_are_bit_identical_to_v5_threshold() -> None:
    """默认 `standing_*` 下，弃胡阈值必须与 v5 的公式**逐位一致**（不变量）。"""
    v5 = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
    armed = _decider()
    non_boundary = _stub((100, 90, 80, 70), me=1)  # 名次 2 ⇒ 中性
    for gain, loss in ((2.0, 3.0), (1.0, 1.0), (8.0, 2.0)):
        assert v5._piao_threshold(gain, loss, non_boundary) == armed._piao_threshold(
            gain, loss, non_boundary
        )
        # 边界态上必须**不同**（否则接入没生效）
        assert armed._piao_threshold(gain, loss, _stub((100, 90, 80, 70), me=3)) != (
            v5._piao_threshold(gain, loss, _stub((100, 90, 80, 70), me=3))
        )


def test_standing_feed_apply_is_exclusive_with_the_piao_entry_point() -> None:
    """`standing_feed_apply=True` 时，弃胡阈值必须**退回 v5 公式**（局况只计一次）。

    为什么钉这条：S3 v1（接弃胡阈值）与 v2（接出牌层喂牌项）**用的是同一组 `standing_*` 系数**，
    如果两个接入点同时生效，局况会被乘两次——那不是「更有姿态」，是**参数被放大成未被验证的形态**。
    """
    v5 = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
    v2 = HeuristicDecider(
        PolicyConfig.for_mode(
            Mode.QUALIFIER,
            standing_lead_scale=1.4,
            standing_behind_scale=0.7,
            standing_feed_apply=True,
        )
    )
    boundary = _stub((100, 90, 80, 70), me=3)  # 边界态：v1 在这里会改阈值
    for gain, loss in ((2.0, 3.0), (8.0, 2.0)):
        assert v2._piao_threshold(gain, loss, boundary) == v5._piao_threshold(
            gain, loss, boundary
        )
    # 而 `_standing_scale` 本身仍非中性（说明它是要去喂牌项那边生效的）
    assert v2._standing_scale(boundary) == 0.7
