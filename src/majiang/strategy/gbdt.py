"""梯度提升树的通用求值器与导出（价值模型、对手模型共用）。

三个刻意的工程约束（交付物要求）：

1. **零运行时依赖**：模型导出为 JSON，用本模块的纯 Python 求值，不把 scikit-learn
   带进交付路径（它只在离线训练时被惰性导入）。
2. **确定性 + 无网络**：纯计算，同一输入必得同一输出，满足「对局复盘可复现」。
3. **版本与特征数校验**：不兼容时抛错，由调用方回退到启发式，绝不静默用错的特征顺序。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

MODEL_VERSION = 1
IDENTITY = "identity"
SIGMOID = "sigmoid"
LINKS = (IDENTITY, SIGMOID)


class ModelError(ValueError):
    """模型文件缺失、版本、特征数或链接函数不兼容。"""


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class TreeEnsemble:
    """梯度提升树的纯 Python 求值器。

    ``link="sigmoid"`` 用于二分类（输出概率），``link="identity"`` 用于回归。
    """

    def __init__(self, payload: dict, *, expected_features: int | None = None) -> None:
        version = int(payload.get("version", 0))
        if version != MODEL_VERSION:
            raise ModelError(f"模型版本不兼容: {version} != {MODEL_VERSION}")
        count = int(payload.get("n_features", 0))
        if expected_features is not None and count != expected_features:
            raise ModelError(f"特征数不兼容: {count} != {expected_features}")
        link = str(payload.get("link", IDENTITY))
        if link not in LINKS:
            raise ModelError(f"未知链接函数: {link!r}")
        self.n_features = count
        self.feature_names: tuple[str, ...] = tuple(payload.get("feature_names", ()))
        self.link = link
        self._init = float(payload["init"])
        self._lr = float(payload["learning_rate"])
        self._trees = payload["trees"]

    @classmethod
    def load(cls, path: str | Path, *, expected_features: int | None = None) -> TreeEnsemble:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"无法加载模型 {path}: {exc}") from exc
        return cls(payload, expected_features=expected_features)

    def raw(self, vector: Sequence[float]) -> float:
        total = self._init
        for tree in self._trees:
            node = 0
            left = tree["left"]
            right = tree["right"]
            threshold = tree["threshold"]
            feature = tree["feature"]
            while left[node] != -1:
                node = left[node] if vector[feature[node]] <= threshold[node] else right[node]
            total += self._lr * tree["value"][node]
        return total

    def predict(self, vector: Sequence[float]) -> float:
        value = self.raw(vector)
        return sigmoid(value) if self.link == SIGMOID else value


def export_sklearn_model(
    model: object,
    path: str | Path,
    *,
    feature_names: Sequence[str],
    link: str = IDENTITY,
) -> None:
    """把训练好的 sklearn 梯度提升模型导出为 JSON。"""
    if link not in LINKS:
        raise ModelError(f"未知链接函数: {link!r}")
    trees = []
    for stage in model.estimators_:  # type: ignore[attr-defined]
        tree = stage[0].tree_
        trees.append(
            {
                "feature": tree.feature.tolist(),
                "threshold": [float(value) for value in tree.threshold],
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "value": [float(value) for value in tree.value.reshape(-1)],
            }
        )
    init = model.init_  # type: ignore[attr-defined]
    constant = getattr(init, "constant_", None)
    if constant is None:
        # 二分类的 init 是 DummyClassifier，先验概率 → 转成 logit
        prior = float(init.class_prior_[1])  # type: ignore[attr-defined]
        prior = min(max(prior, 1e-6), 1 - 1e-6)
        init_value = math.log(prior / (1 - prior))
    else:
        init_value = float(constant.reshape(-1)[0])
    payload = {
        "version": MODEL_VERSION,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "link": link,
        "learning_rate": float(model.learning_rate),  # type: ignore[attr-defined]
        "init": init_value,
        "trees": trees,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


__all__ = [
    "IDENTITY",
    "MODEL_VERSION",
    "SIGMOID",
    "ModelError",
    "TreeEnsemble",
    "export_sklearn_model",
    "sigmoid",
]
