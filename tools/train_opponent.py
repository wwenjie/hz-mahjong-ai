"""训练对手听牌模型并导出（tasks.md 6B.4–6B.6）。

按规格要求的评估口径：

- **概率质量**：对数损失、布里尔分数、可靠性曲线（不能只看 AUC——输出要直接与阈值
  比较，排序好但未校准的概率会给出错误的阈值决策）。
- **基线对照**：固定常数、仅「剩余可摸牌数」的单特征模型、逻辑回归。
- **特征消融**：按组移除后重训，输出各组贡献。
- **数据划分按整局分开**：验证集用另一个种子单独生成。

用法::

    uv run python tools/train_opponent.py \\
        --train 'data/opponent_train.part*.npz' --valid 'data/opponent_valid.part*.npz' \\
        --out models/opponent_model.json
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from majiang.strategy import gbdt
from majiang.strategy.opponent_features import FEATURE_COUNT, FEATURE_NAMES

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "target_melds": ("target_melds", "target_chi", "target_peng", "target_gang"),
    "target_discards": (
        "target_discards",
        "target_suit_concentration",
        "target_middle_share",
        "target_recent_same_suit",
        "target_discarded_god",
        "target_discard_rate",
    ),
    "target_state": ("target_concealed", "target_is_dealer", "target_restricted"),
    "table": (
        "wall_remaining",
        "draws_left",
        "progress",
        "gods_seen",
        "table_melds",
    ),
    "observer": ("my_concealed", "my_melds", "my_gods"),
}


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


def make_model(trees: int, depth: int, lr: float) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=trees, max_depth=depth, learning_rate=lr, subsample=0.8, random_state=20260924
    )


def report(tag: str, y: np.ndarray, probability: np.ndarray) -> None:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    print(
        f"  {tag:34s} AUC {roc_auc_score(y, probability):.4f}  "
        f"对数损失 {log_loss(y, clipped):.4f}  布里尔 {brier_score_loss(y, probability):.4f}"
    )


def calibration_table(y: np.ndarray, probability: np.ndarray, bins: int = 5) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    print("    可靠性（预测区间 → 实际频率 / 样本数）")
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (probability >= low) & (probability < high if index < bins - 1 else probability <= high)
        if mask.sum() == 0:
            continue
        print(
            f"      [{low:.1f},{high:.1f}) → {y[mask].mean():.3f} / {int(mask.sum())}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练对手听牌模型")
    parser.add_argument("--train", default="data/opponent_train.part*.npz")
    parser.add_argument("--valid", default="data/opponent_valid.part*.npz")
    parser.add_argument("--out", default="models/opponent_model.json")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args(argv)

    x_train, y_train = load(args.train)
    print(f"训练集 {x_train.shape[0]} 条，正例率 {y_train.mean():.1%}")

    model = make_model(args.trees, args.depth, args.lr)
    model.fit(x_train, y_train)

    try:
        x_valid, y_valid = load(args.valid)
    except SystemExit as exc:
        print(f"跳过验证：{exc}", file=sys.stderr)
        x_valid = y_valid = None

    if x_valid is not None:
        probability = model.predict_proba(x_valid)[:, 1]
        print(f"验证集 {x_valid.shape[0]} 条（独立种子，按整局分开），正例率 {y_valid.mean():.1%}")
        report("模型（GBDT）", y_valid, probability)
        report("基线：固定常数", y_valid, np.full_like(probability, y_train.mean()))
        single = FEATURE_NAMES.index("draws_left")
        report("基线：仅剩余可摸牌数", y_valid, 1.0 - x_valid[:, single] / 63.0)
        logistic = LogisticRegression(max_iter=1000).fit(x_train, y_train)
        report("基线：逻辑回归", y_valid, logistic.predict_proba(x_valid)[:, 1])
        calibration_table(y_valid, probability)

    importance = np.argsort(model.feature_importances_)[::-1]
    print("  特征重要度 top8:", ", ".join(f"{FEATURE_NAMES[i]}={model.feature_importances_[i]:.3f}" for i in importance[:8]))

    if x_valid is not None:
        print("  特征消融（按组移除后重训）")
        full = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
        for group, names in FEATURE_GROUPS.items():
            drop = [FEATURE_NAMES.index(name) for name in names]
            keep = [index for index in range(FEATURE_COUNT) if index not in drop]
            ablated = make_model(args.trees, args.depth, args.lr).fit(x_train[:, keep], y_train)
            auc = roc_auc_score(y_valid, ablated.predict_proba(x_valid[:, keep])[:, 1])
            print(f"    去掉 {group:16s} AUC {auc:.4f}  （全量 {full:.4f}，Δ {auc - full:+.4f}）")

    gbdt.export_sklearn_model(
        model, Path(args.out), feature_names=FEATURE_NAMES, link=gbdt.SIGMOID
    )
    print(f"已导出 {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
