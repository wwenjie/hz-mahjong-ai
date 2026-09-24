"""价值模型测试（tasks.md 5.15）。

最关键的一条是**纯 Python 求值必须与 sklearn 的预测逐值一致**——交付物靠纯 Python
求值上线，若与训练侧不等价，离线指标就没有意义。
"""

import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from majiang.rules import tiles
from majiang.rules.action import DISCARD, GANG, HU, Action
from majiang.rules.tiles import GOD
from majiang.strategy import features
from majiang.strategy.policy import HeuristicDecider, PolicyConfig
from majiang.strategy.value import (
    MODEL_VERSION,
    ValueDecider,
    ValueModel,
    ValueModelError,
    export_sklearn_model,
)

from .test_policy import situation


def train_tiny(tmp_path, *, rows: int = 400, trees: int = 25):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(rows, features.FEATURE_COUNT)).astype(np.float32)
    y = (x[:, 0] * 2.0 - x[:, 5] + rng.normal(scale=0.3, size=rows)).astype(np.float32)
    model = GradientBoostingRegressor(n_estimators=trees, max_depth=3, learning_rate=0.1)
    model.fit(x, y)
    path = tmp_path / "value_model.json"
    export_sklearn_model(model, path)
    return model, ValueModel.load(path), x


def test_python_evaluator_matches_sklearn(tmp_path) -> None:
    model, loaded, x = train_tiny(tmp_path)
    expected = model.predict(x[:25])
    got = np.asarray([loaded.predict(row) for row in x[:25]])
    assert np.allclose(got, expected, atol=1e-6), np.abs(got - expected).max()


def test_export_contains_version_and_feature_names(tmp_path) -> None:
    _, _, _ = train_tiny(tmp_path, rows=50, trees=3)
    payload = json.loads((tmp_path / "value_model.json").read_text())
    assert payload["version"] == MODEL_VERSION
    assert payload["n_features"] == features.FEATURE_COUNT
    assert payload["feature_names"] == list(features.FEATURE_NAMES)


def test_model_rejects_wrong_version() -> None:
    with pytest.raises(ValueModelError, match="版本"):
        ValueModel({"version": 999, "n_features": features.FEATURE_COUNT})


def test_model_rejects_wrong_feature_count() -> None:
    with pytest.raises(ValueModelError, match="特征数"):
        ValueModel({"version": MODEL_VERSION, "n_features": 7})


def test_model_load_reports_missing_file(tmp_path) -> None:
    with pytest.raises(ValueModelError, match="无法加载"):
        ValueModel.load(tmp_path / "nope.json")


def test_extract_is_deterministic_and_fixed_length() -> None:
    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    first = features.extract(obj)
    assert len(first) == features.FEATURE_COUNT
    assert first == features.extract(obj)


def test_value_decider_without_model_defers_to_heuristic() -> None:
    obj = situation("1w1w2w3w4w5w6w7w9w1b2b3b9w9w", drawn="9w")
    actions = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj.hand.counts) if n)
    heuristic = HeuristicDecider(PolicyConfig())
    decider = ValueDecider(model=None, heuristic=heuristic)
    chosen = decider.choose(obj, actions, budget_ms=1500)
    assert chosen in actions


def test_value_decider_never_bypasses_hu_or_gang(tmp_path) -> None:
    """同一教训：只覆盖出牌，胡与杠必须照旧走启发式，否则策略永不自摸。"""
    _, model, _ = train_tiny(tmp_path)
    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn="5b")
    actions = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj.hand.counts) if n)
    decider = ValueDecider(model=model)
    assert decider.choose(obj, (Action(HU), *actions), budget_ms=1500).kind == HU

    gang_hand = "9t9t9t9t1w2w3w4w5w6w7w8w1b1b"
    obj2 = situation(gang_hand, drawn="1b")
    gang = Action(GANG, tile=tiles.parse("9t"), gang_kind="angang")
    actions2 = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj2.hand.counts) if n)
    assert ValueDecider(model=model).choose(obj2, (gang, *actions2), budget_ms=1500).kind == GANG


def test_value_decider_scores_every_candidate(tmp_path) -> None:
    _, model, _ = train_tiny(tmp_path)
    obj = situation("1w1w2w3w4w5w6w7w9w1b2b3b9w9w", drawn="9w")
    actions = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj.hand.counts) if n)
    decider = ValueDecider(model=model)
    chosen = decider.choose(obj, actions, budget_ms=1500)
    assert chosen in actions
    assert chosen.tile != GOD
    assert decider.last_detail["value"]
    assert len(decider.last_detail["value"]) <= 4


def test_value_decider_is_deterministic(tmp_path) -> None:
    _, model, _ = train_tiny(tmp_path)
    obj = situation("1w1w2w3w4w5w6w7w9w1b2b3b9w9w", drawn="9w")
    actions = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj.hand.counts) if n)
    first = ValueDecider(model=model).choose(obj, actions, budget_ms=1500)
    second = ValueDecider(model=model).choose(obj, actions, budget_ms=1500)
    assert first == second
