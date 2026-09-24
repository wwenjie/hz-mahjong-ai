"""训练价值模型并导出为纯 Python 可求值的 JSON（tasks.md 5.15）。

目标：学一个 `E[本局本人最终得分 | 公开局面]`，用来给出牌候选打分——替代现在那个
「用极速策略滚出」的评估器（实测它就是搜索无效的瓶颈）。

两个刻意的工程选择：

1. **数据划分按整局分开**：同一局内的样本高度相关，随机切分会泄漏。这里用**另一个种子
   单独生成验证集**，训练/验证天然不共享任何一局。
2. **模型导出为 JSON + 纯 Python 求值**：交付物要保持零运行时依赖、确定性推理、无网络，
   因此不把 scikit-learn 带进线上路径。

用法::

    uv run python tools/train_value.py \\
        --train 'data/value_train.part*.npz' --valid 'data/value_valid.part*.npz' \\
        --out models/value_model.json
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from majiang.strategy.features import FEATURE_COUNT, FEATURE_NAMES
from majiang.strategy.value import export_sklearn_model


def load(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"没有匹配到数据分片: {pattern}")
    xs = [np.load(name)["x"] for name in files]
    ys = [np.load(name)["y"] for name in files]
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    if x.shape[1] != FEATURE_COUNT:
        raise SystemExit(f"特征数不符：数据 {x.shape[1]}，代码 {FEATURE_COUNT}")
    return x, y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练价值模型")
    parser.add_argument("--train", default="data/value_train.part*.npz")
    parser.add_argument("--valid", default="data/value_valid.part*.npz")
    parser.add_argument("--out", default="models/value_model.json")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args(argv)

    x_train, y_train = load(args.train)
    print(f"训练集 {x_train.shape[0]} 条，目标均值 {y_train.mean():.2f}、标准差 {y_train.std():.2f}")

    model = GradientBoostingRegressor(
        n_estimators=args.trees,
        max_depth=args.depth,
        learning_rate=args.lr,
        subsample=0.8,
        random_state=20260924,
    )
    model.fit(x_train, y_train)

    try:
        x_valid, y_valid = load(args.valid)
    except SystemExit as exc:
        print(f"跳过验证：{exc}", file=sys.stderr)
        x_valid = y_valid = None

    if x_valid is not None:
        pred = model.predict(x_valid)
        print(f"验证集 {x_valid.shape[0]} 条（独立种子，按整局分开）")
        print(f"  MAE {mean_absolute_error(y_valid, pred):.3f}   R² {r2_score(y_valid, pred):.4f}")
        baseline = np.full_like(pred, y_train.mean())
        print(f"  常量基线 MAE {mean_absolute_error(y_valid, baseline):.3f}")
        order = np.argsort(pred)
        low = y_valid[order[: len(order) // 10]].mean()
        high = y_valid[order[-len(order) // 10 :]].mean()
        print(f"  预设值十分位的实际均值：最低 {low:+.2f} → 最高 {high:+.2f}（差 {high - low:+.2f}）")

    importance = np.argsort(model.feature_importances_)[::-1][:8]
    print("  特征重要度 top8:", ", ".join(f"{FEATURE_NAMES[i]}={model.feature_importances_[i]:.3f}" for i in importance))

    export_sklearn_model(model, args.out)
    print(f"已导出 {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
