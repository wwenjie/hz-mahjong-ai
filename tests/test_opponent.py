"""对手听牌模型测试（tasks.md 6B）。

最关键的一条是**分类器的纯 Python 求值必须与 sklearn 的 `predict_proba` 逐值一致**——
二分类多了一个环节：初始化项要从先验概率转成 logit。若这里错了，概率就整体偏移，
而模型输出要**直接与阈值比较**，偏移会直接变成错误的阈值决策。
"""

import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from majiang.rules import shanten as shanten_module
from majiang.strategy import gbdt, opponent_features, risk
from majiang.strategy.opponent import ModelReadyModel, OpponentModel, load_or_none
from majiang.strategy.policy import HeuristicDecider, PolicyConfig

from .test_policy import situation


def train_tiny(tmp_path, *, rows: int = 600, trees: int = 40):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(rows, opponent_features.FEATURE_COUNT)).astype(np.float32)
    logits = x[:, 0] * 1.5 - x[:, 4] * 0.5
    y = (rng.random(rows) < 1 / (1 + np.exp(-logits))).astype(int)
    model = GradientBoostingClassifier(n_estimators=trees, max_depth=3, learning_rate=0.1)
    model.fit(x, y)
    path = tmp_path / "opponent_model.json"
    gbdt.export_sklearn_model(
        model, path, feature_names=opponent_features.FEATURE_NAMES, link=gbdt.SIGMOID
    )
    return model, OpponentModel.load(path), x


def test_python_evaluator_matches_sklearn_predict_proba(tmp_path) -> None:
    model, loaded, x = train_tiny(tmp_path)
    expected = model.predict_proba(x[:30])[:, 1]
    got = np.asarray([loaded.predict(row) for row in x[:30]])
    assert np.allclose(got, expected, atol=1e-6), np.abs(got - expected).max()


def test_classifier_export_carries_sigmoid_link(tmp_path) -> None:
    _, loaded, _ = train_tiny(tmp_path, rows=80, trees=3)
    assert loaded.link == gbdt.SIGMOID
    assert loaded.n_features == opponent_features.FEATURE_COUNT


def test_predictions_are_probabilities(tmp_path) -> None:
    _, loaded, x = train_tiny(tmp_path)
    values = np.asarray([loaded.predict(row) for row in x[:50]])
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_model_rejects_wrong_feature_count() -> None:
    with pytest.raises(gbdt.ModelError, match="特征数"):
        OpponentModel({"version": gbdt.MODEL_VERSION, "n_features": 3, "link": gbdt.SIGMOID})


def test_model_rejects_wrong_link() -> None:
    with pytest.raises(gbdt.ModelError, match="链接函数"):
        OpponentModel(
            {
                "version": gbdt.MODEL_VERSION,
                "n_features": opponent_features.FEATURE_COUNT,
                "link": "softmax",
            }
        )


def test_load_or_none_falls_back_on_missing_file(tmp_path) -> None:
    ready_model = load_or_none(tmp_path / "nope.json")
    assert ready_model.using_model is False


def test_load_or_none_uses_the_model_when_available(tmp_path) -> None:
    _, _, _ = train_tiny(tmp_path, rows=80, trees=3)
    assert load_or_none(tmp_path / "opponent_model.json").using_model is True


def test_ready_model_without_model_matches_heuristic() -> None:
    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    opponents = opponent_features.opponents_of(obj.seat)
    fallback = ModelReadyModel(None).estimate(obj, opponents=opponents)
    direct = risk.HeuristicReadyModel().estimate(obj, opponents=opponents)
    assert fallback == direct


def test_ready_model_with_model_changes_the_estimate(tmp_path) -> None:
    """模型确实接管了风险估计（而不是仍走启发式）。"""
    _, model, _ = train_tiny(tmp_path)
    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    opponents = opponent_features.opponents_of(obj.seat)
    with_model = ModelReadyModel(model).estimate(obj, opponents=opponents)
    heuristic = ModelReadyModel(None).estimate(obj, opponents=opponents)
    assert len(with_model) == len(heuristic) == 3
    assert all(0.0 <= value <= 1.0 for value in with_model)


def test_constant_high_risk_model_lowers_lap_survival() -> None:
    """管道验证：把听牌概率拉到 0.9，q 必须显著下降。"""

    class AlwaysReady:
        def estimate(self, situation, *, opponents):
            return tuple(0.9 for _ in opponents)

    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    baseline = risk.lap_survival(risk.assess(obj))
    stressed = risk.lap_survival(risk.assess(obj, model=AlwaysReady()))
    assert stressed < baseline


def test_heuristic_decider_accepts_a_risk_model() -> None:
    class AlwaysCalm:
        def estimate(self, situation, *, opponents):
            return tuple(0.01 for _ in opponents)

    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    decider = HeuristicDecider(PolicyConfig(), risk_model=AlwaysCalm())
    risks = decider._risks(obj)
    assert len(risks) == 3
    assert all(item.ready_probability >= 0.0 for item in risks)


def test_opponent_features_only_describe_the_target_and_public_state() -> None:
    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    row = opponent_features.extract(obj, 2)
    assert len(row) == opponent_features.FEATURE_COUNT
    assert row == opponent_features.extract(obj, 2)


def test_is_ready_label_matches_shanten_zero() -> None:
    import random

    from majiang.sim.round import deal
    from tools.gen_opponent_data import is_ready

    state = deal(random.Random(3), dealer=0)
    for seat in range(4):
        expected = shanten_module.shanten(state.seats[seat].hand, 0) == 0
        assert is_ready(state, seat) is expected


def train_tiny_calibrated(tmp_path, *, rows: int = 600, trees: int = 40):
    """用训练集自身拟合等渗校准——单元测试只验证链路，不验证泛化。"""
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(11)
    x = rng.normal(size=(rows, opponent_features.FEATURE_COUNT)).astype(np.float32)
    logits = x[:, 0] * 1.5 - x[:, 4] * 0.5
    y = (rng.random(rows) < 1 / (1 + np.exp(-logits))).astype(int)
    model = GradientBoostingClassifier(n_estimators=trees, max_depth=3, learning_rate=0.1).fit(x, y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(x)[:, 1], y)
    path = tmp_path / "calibrated.json"
    gbdt.export_sklearn_model(
        model,
        path,
        feature_names=opponent_features.FEATURE_NAMES,
        link=gbdt.SIGMOID,
        calibration=isotonic,
    )
    return model, isotonic, OpponentModel.load(path), x


def test_calibrated_export_matches_sklearn_isotonic(tmp_path) -> None:
    model, isotonic, loaded, x = train_tiny_calibrated(tmp_path)
    expected = isotonic.predict(model.predict_proba(x[:40])[:, 1])
    got = np.asarray([loaded.predict(row) for row in x[:40]])
    assert np.allclose(got, expected, atol=1e-6), np.abs(got - expected).max()


def test_calibration_actually_changes_predictions(tmp_path) -> None:
    """回归防线：校准段必须真的生效，而不是被解析后遗忘。"""
    _, _, loaded, x = train_tiny_calibrated(tmp_path)
    assert loaded.has_calibration is True
    calibrated = np.asarray([loaded.predict(row) for row in x[:60]])
    payload = json.loads((tmp_path / "calibrated.json").read_text(encoding="utf-8"))
    del payload["calibration"]
    plain = np.asarray([OpponentModel(payload).predict(row) for row in x[:60]])
    assert not np.allclose(calibrated, plain)


def test_model_without_calibration_reports_no_calibration(tmp_path) -> None:
    _, loaded, _ = train_tiny(tmp_path)
    assert loaded.has_calibration is False


def test_calibration_rejects_non_monotonic_x() -> None:
    with pytest.raises(gbdt.ModelError, match="单调"):
        OpponentModel(
            {
                "version": gbdt.MODEL_VERSION,
                "n_features": opponent_features.FEATURE_COUNT,
                "link": gbdt.SIGMOID,
                "init": 0.0,
                "learning_rate": 1.0,
                "trees": [],
                "calibration": {"kind": "isotonic", "x": [0.5, 0.3], "y": [0.1, 0.2]},
            }
        )


def test_calibration_rejects_unknown_kind() -> None:
    with pytest.raises(gbdt.ModelError, match="校准类型"):
        OpponentModel(
            {
                "version": gbdt.MODEL_VERSION,
                "n_features": opponent_features.FEATURE_COUNT,
                "link": gbdt.SIGMOID,
                "init": 0.0,
                "learning_rate": 1.0,
                "trees": [],
                "calibration": {"kind": "platt", "x": [0.1], "y": [0.1]},
            }
        )


def test_calibration_rejects_mismatched_lengths() -> None:
    with pytest.raises(gbdt.ModelError, match="长度"):
        OpponentModel(
            {
                "version": gbdt.MODEL_VERSION,
                "n_features": opponent_features.FEATURE_COUNT,
                "link": gbdt.SIGMOID,
                "init": 0.0,
                "learning_rate": 1.0,
                "trees": [],
                "calibration": {"kind": "isotonic", "x": [0.1, 0.2], "y": [0.1]},
            }
        )


def test_interpolate_matches_numpy_interp() -> None:
    xs = np.array([0.0, 0.2, 0.5, 1.0])
    ys = np.array([0.1, 0.3, 0.3, 0.9])
    probe = np.linspace(-0.2, 1.2, 57)
    expected = np.interp(probe, xs, ys)
    got = np.asarray([gbdt.interpolate(float(v), list(xs), list(ys)) for v in probe])
    assert np.allclose(got, expected)


def mid_round_state(seed: int = 5):
    """发牌后先完成庄家首摸。

    ``sim.round.deal`` 只发 13×4 张，庄家第 14 张仍留在牌墙里（``wall=84``）。该中间态
    不对应平台任何可观测状态（平台把这张算作发牌），因此 ``TableState`` 会拒绝它。
    这里先摸一张，把状态推进到与平台一致的可观测点。
    """
    import random

    from majiang.sim import round as round_module

    state = round_module.deal(random.Random(seed), dealer=0)
    assert round_module._draw(state, state.dealer) is not None
    return state


def test_features_ignore_opponent_hand_composition() -> None:
    """6B.3 泄漏回归：换成完全不同的牌但**张数不变**，特征必须逐值不变。

    对手手牌的**张数**是公开信息（由副露可推、平台也在快照里直接下发），所以特征可以
    依赖它；手牌的**构成**是隐藏信息，一旦依赖就越过合规红线。这条测试就是守住这条线：
    手牌数组里装什么牌都不影响输出，只有它的总和可以。

    重排时牌种总数守恒（牌从牌墙换到手上，反之亦然），以免连带改动 ``wall_remaining``。
    """
    from majiang.rules import tiles as tiles_module
    from majiang.sim import round as round_module

    state = mid_round_state()
    observer, target = 0, 2
    before = opponent_features.extract(round_module.situation_for(state, observer, "draw"), target)

    seat = state.seats[target]
    held = list(seat.hand)
    total = sum(held)

    # 可分配的牌种余量：全场 4 张，扣掉所有暗手、副露与弃牌，再把目标自己的牌放回来
    pool = [tiles_module.COPIES_PER_KIND] * tiles_module.TILE_KINDS
    for other in state.seats:
        for tile, amount in enumerate(other.hand):
            pool[tile] -= amount
        for meld in other.melds:
            for tile, amount in enumerate(meld.counts()):
                pool[tile] -= amount
        for tile in other.discards:
            pool[tile] -= 1
    for tile, amount in enumerate(held):
        pool[tile] += amount
    assert min(pool) >= 0

    # 优先用目标手上没有的牌种，保证构成真的变了
    replacement = [0] * tiles_module.TILE_KINDS
    remaining = total
    for tile in sorted(range(tiles_module.TILE_KINDS), key=lambda value: (held[value], value)):
        if remaining == 0:
            break
        take = min(remaining, pool[tile])
        replacement[tile] = take
        remaining -= take
    assert remaining == 0
    assert sum(replacement) == total
    assert replacement != held, "重排后构成未变化，测试未起到作用"
    seat.hand[:] = replacement

    after = opponent_features.extract(round_module.situation_for(state, observer, "draw"), target)
    assert after == before


def test_feature_row_reacts_to_the_target_concealed_count() -> None:
    """对手手牌张数变化时特征**应当**变化——否则上一条测试可能只是因为特征全是常数。"""
    from majiang.sim import round as round_module

    state = mid_round_state()
    observer, target = 0, 2
    before = opponent_features.extract(round_module.situation_for(state, observer, "draw"), target)
    state.seats[target].hand[:] = [0] * len(state.seats[target].hand)
    after = opponent_features.extract(round_module.situation_for(state, observer, "draw"), target)
    assert after != before, "特征对手牌张数变化无反应，说明该特征未被真正读取"
