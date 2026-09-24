"""对手听牌模型（tasks.md 6B.5 / 6B.8 / 6B.9）。

学 `P(对手听牌 | 公开信息)`，替换 :mod:`majiang.strategy.risk` 里那个手写启发式。

为什么这个模型比价值模型可学得多：

- **标签是隐藏状态的确定值**（对手是否听牌），由完整手牌给出、**无噪声**；
  而价值模型的目标是局分——随机变量的实现值，实测 96% 的方差是运气。
- **正例率 20–35%**，标签稠密。
- 公开信息与听牌之间是确定的统计关系（副露多、弃牌连贯 → 更可能已听）。

替换后的收益流：`sᵢ` → `rᵢ` → `q` → 财飘阈值 / 抓打圈时机 / 喂牌代价，
全部经由 :func:`majiang.strategy.risk.assess` 的 `model` 参数注入，上层无需改动。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from majiang.rules.situation import Situation

from . import gbdt, opponent_features
from .gbdt import TreeEnsemble
from .risk import HeuristicReadyModel, ReadyModel

MODEL_VERSION = gbdt.MODEL_VERSION
OpponentModelError = gbdt.ModelError


class OpponentModel(TreeEnsemble):
    """对手听牌概率模型（纯 Python 求值）。"""

    def __init__(self, payload: dict) -> None:
        super().__init__(payload, expected_features=opponent_features.FEATURE_COUNT)

    @classmethod
    def load(cls, path: str | Path) -> OpponentModel:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise gbdt.ModelError(f"无法加载对手模型 {path}: {exc}") from exc
        return cls(payload)

    def ready_probability(self, situation: Situation, target: int) -> float:
        return float(self.predict(opponent_features.extract(situation, target)))


class ModelReadyModel:
    """把模型包装成 ``risk.ReadyModel``；模型不可用时回退手写启发式。

    回退是硬要求（tasks.md 6B.9）：绝不因模型缺失而中止决策。
    """

    def __init__(self, model: OpponentModel | None) -> None:
        self.model = model
        self._fallback = HeuristicReadyModel()

    @property
    def using_model(self) -> bool:
        return self.model is not None

    def estimate(self, situation: Situation, *, opponents: Sequence[int]) -> tuple[float, ...]:
        if self.model is None:
            return self._fallback.estimate(situation, opponents=opponents)
        return tuple(self.model.ready_probability(situation, seat) for seat in opponents)


def load_or_none(path: str | Path | None) -> ModelReadyModel:
    """加载模型；路径为空或不可用时返回回退版本（不抛错）。"""
    if not path:
        return ModelReadyModel(None)
    try:
        return ModelReadyModel(OpponentModel.load(path))
    except gbdt.ModelError:
        return ModelReadyModel(None)


__all__ = [
    "MODEL_VERSION",
    "ModelReadyModel",
    "OpponentModel",
    "OpponentModelError",
    "load_or_none",
]
