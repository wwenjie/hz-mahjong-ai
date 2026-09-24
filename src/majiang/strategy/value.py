"""价值模型（tasks.md 5.15）。

学到的 `E[本局本人最终得分 | 公开局面]`，用来给出牌候选打分——替代此前那个
「用极速策略滚出」的评估器（实测它就是搜索无效的瓶颈：在粗糙评估器上搜更多候选，
只会按噪声选得更差）。

三个刻意的工程约束：

1. **零运行时依赖**：模型导出为 JSON，用本模块的纯 Python 树求值，不把 scikit-learn
   带进交付路径。
2. **确定性 + 无网络**：纯计算，同一输入必得同一输出，满足「对局复盘可复现」。
3. **版本与特征数校验**：不兼容时抛错，由调用方回退到启发式，绝不静默用错的特征顺序。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from majiang.rules.action import DISCARD, Action
from majiang.rules.situation import Situation

from . import features, gbdt
from .gbdt import TreeEnsemble
from .policy import HeuristicDecider

MODEL_VERSION = gbdt.MODEL_VERSION
ValueModelError = gbdt.ModelError


class ValueModel(TreeEnsemble):
    """`E[本局本人最终得分 | 公开局面]` 的回归模型（纯 Python 求值）。"""

    def __init__(self, payload: dict) -> None:
        super().__init__(payload, expected_features=features.FEATURE_COUNT)

    @classmethod
    def load(cls, path: str | Path) -> ValueModel:
        payload = _load_payload(path)
        return cls(payload)


def _load_payload(path: str | Path) -> dict:
    import json

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise gbdt.ModelError(f"无法加载价值模型 {path}: {exc}") from exc


def export_sklearn_model(model: object, path: str | Path) -> None:
    """把训练好的 ``GradientBoostingRegressor`` 导出为 JSON（离线训练用）。"""
    gbdt.export_sklearn_model(
        model, path, feature_names=features.FEATURE_NAMES, link=gbdt.IDENTITY
    )


class ValueDecider:
    """用价值模型给**出牌**打分；胡/杠/弃胡飘仍交给启发式。

    只覆盖出牌那一步是踩过坑的结论：早期实现直接从候选排序入手，绕过了胡/杠判定，
    导致策略永不自摸（实测 0% 胡率）。
    """

    name = "value"

    def __init__(
        self,
        model: ValueModel | None = None,
        heuristic: HeuristicDecider | None = None,
        *,
        model_path: str | Path | None = None,
    ) -> None:
        if model is None and model_path is not None:
            model = ValueModel.load(model_path)
        self.model = model
        self.heuristic = heuristic or HeuristicDecider()
        self.last_reason = ""
        self.last_detail: dict[str, object] = {}
        self.last_model_error = ""

    def configure(self, tournament) -> None:
        self.heuristic.configure(tournament)

    def choose(self, situation: Situation, actions, *, budget_ms: int = 0) -> Action | None:
        self.last_reason = ""
        self.last_detail = {}
        first = self.heuristic.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = self.heuristic.last_reason
        self.last_detail = dict(self.heuristic.last_detail)
        if first is None or first.kind != DISCARD or self.model is None:
            return first
        if situation.drawn_tile is None:
            return first

        best: tuple[float, Action] | None = None
        scored: list[tuple[float, str]] = []
        for action in actions:
            if action.kind != DISCARD or action.tile is None:
                continue
            if situation.hand.counts[action.tile] <= 0:
                continue
            after = replace(situation, hand=situation.hand.without_tile(action.tile))
            value = self.model.predict(features.extract(after))
            scored.append((value, action.describe()))
            if best is None or value > best[0]:
                best = (value, action)
        if best is None:
            return first
        scored.sort(reverse=True)
        self.last_detail["value"] = [f"{name}={value:.2f}" for value, name in scored[:4]]
        self.last_reason = f"价值模型：{scored[0][1]} 价值 {scored[0][0]:.2f}"
        return best[1]


__all__ = [
    "MODEL_VERSION",
    "ValueDecider",
    "ValueModel",
    "ValueModelError",
    "export_sklearn_model",
]
