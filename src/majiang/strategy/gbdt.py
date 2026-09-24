"""梯度提升树的通用求值器与导出（价值模型、对手模型共用）。

四个刻意的工程约束（交付物要求）：

1. **零运行时依赖**：模型导出为 JSON，用本模块的纯 Python 求值，不把 scikit-learn
   带进交付路径（它只在离线训练时被惰性导入）。
2. **确定性 + 无网络**：纯计算，同一输入必得同一输出，满足「对局复盘可复现」。
3. **版本与特征数校验**：不兼容时抛错，由调用方回退到启发式，绝不静默用错的特征顺序。
4. **可选等渗校准**：模型 JSON 可内嵌一张等渗映射表，在链接函数之后施加，用于修正
   高概率尾部的系统性低估。运行时同样是纯 Python 分段线性插值。

校准的动机（实测）：对手听牌模型在 ``[0.8,0.9)`` 区间预测 0.847 而实际 0.908，
低估 6 个百分点。由于风险聚合是 ``q = ∏(1 − pᵢ)``，尾部低估会被连乘放大
（三家 0.847 → q=0.0036，真实 0.908 → q=0.0008，**高估近 4.6 倍**），
而高概率区间恰好对应残局的追链决策窗口。
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
ISOTONIC = "isotonic"


class ModelError(ValueError):
    """模型文件缺失、版本、特征数、链接函数或校准表不兼容。"""


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def interpolate(value: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """纯 Python 版 ``numpy.interp``：线性插值，越界取端点值。

    与 ``sklearn.isotonic.IsotonicRegression`` 在 ``out_of_bounds="clip"`` 下的预测
    语义一致（先插值、再由调用方裁剪到 ``[y_min, y_max]``）。
    """
    count = len(xs)
    if count == 0:
        return value
    if value <= xs[0]:
        return ys[0]
    if value >= xs[count - 1]:
        return ys[count - 1]
    low, high = 0, count - 1
    while high - low > 1:
        middle = (low + high) // 2
        if xs[middle] <= value:
            low = middle
        else:
            high = middle
    span = xs[high] - xs[low]
    if span <= 0:
        return ys[high]
    ratio = (value - xs[low]) / span
    return ys[low] + ratio * (ys[high] - ys[low])


class TreeEnsemble:
    """梯度提升树的纯 Python 求值器。

    ``link="sigmoid"`` 用于二分类（输出概率），``link="identity"`` 用于回归。
    若模型含 ``calibration`` 段，则在链接函数之后施加等渗校准。
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
        self._cal_x: tuple[float, ...] | None = None
        self._cal_y: tuple[float, ...] | None = None
        self._cal_min = 0.0
        self._cal_max = 1.0
        self._cal_from = float("-inf")
        calibration = payload.get("calibration")
        if calibration is not None:
            self._load_calibration(calibration)

    def _load_calibration(self, calibration: dict) -> None:
        kind = str(calibration.get("kind", ""))
        if kind != ISOTONIC:
            raise ModelError(f"未知校准类型: {kind!r}")
        xs = tuple(float(value) for value in calibration.get("x", ()))
        ys = tuple(float(value) for value in calibration.get("y", ()))
        if len(xs) != len(ys) or not xs:
            raise ModelError(f"校准表长度不匹配: x={len(xs)} y={len(ys)}")
        if any(b < a for a, b in zip(xs, xs[1:])):
            raise ModelError("校准表的 x 必须单调不减（等渗约束）")
        self._cal_x, self._cal_y = xs, ys
        self._cal_min = float(calibration.get("y_min", ys[0]))
        self._cal_max = float(calibration.get("y_max", ys[-1]))
        # 只在概率不低于该值时施加校准——用于「只修尾部、不动中段」的定向校准
        self._cal_from = float(calibration.get("applies_from", float("-inf")))

    @property
    def has_calibration(self) -> bool:
        return self._cal_x is not None

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
        if self.link == SIGMOID:
            value = sigmoid(value)
        if self._cal_x is not None and value >= self._cal_from:
            value = interpolate(value, self._cal_x, self._cal_y)
            value = min(max(value, self._cal_min), self._cal_max)
        return value


def export_sklearn_model(
    model: object,
    path: str | Path,
    *,
    feature_names: Sequence[str],
    link: str = IDENTITY,
    calibration: object | None = None,
    calibration_from: float | None = None,
) -> None:
    """把训练好的 sklearn 梯度提升模型导出为 JSON。

    ``calibration`` 若给定，应为已拟合的 ``IsotonicRegression``（``out_of_bounds="clip"``）。
    ``calibration_from`` 若给定，则只有概率不低于该值时才施加校准；用于「只修尾部、
    不动中段」的定向校准——全量等渗会为了追尾部而把方差灌进主体（实测净亏）。
    """
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
    payload: dict[str, object] = {
        "version": MODEL_VERSION,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "link": link,
        "learning_rate": float(model.learning_rate),  # type: ignore[attr-defined]
        "init": init_value,
        "trees": trees,
    }
    if calibration is not None:
        # 该版本 sklearn 的 IsotonicRegression 没有 y_min_/y_max_，裁剪边界即
        # y_thresholds_ 的两个端点（等渗映射单调，端点就是极值）。由于
        # ``numpy.interp`` 本身已把越界值夹到端点，这一步裁剪恒为无操作，但仍显式保留，
        # 以免将来换用非端点裁剪的语义。
        ys = [float(value) for value in calibration.y_thresholds_]  # type: ignore[attr-defined]
        block: dict[str, object] = {
            "kind": ISOTONIC,
            "x": [float(value) for value in calibration.X_thresholds_],  # type: ignore[attr-defined]
            "y": ys,
            "y_min": min(ys),
            "y_max": max(ys),
        }
        if calibration_from is not None:
            block["applies_from"] = float(calibration_from)
        payload["calibration"] = block
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


__all__ = [
    "IDENTITY",
    "ISOTONIC",
    "MODEL_VERSION",
    "SIGMOID",
    "ModelError",
    "TreeEnsemble",
    "export_sklearn_model",
    "interpolate",
    "sigmoid",
]
