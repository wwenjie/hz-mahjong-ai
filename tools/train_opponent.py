"""训练对手听牌模型并导出（tasks.md 6B.4–6B.6）。

按规格要求的评估口径：

- **概率质量**：对数损失、布里尔分数、可靠性曲线（不能只看 AUC——输出要直接与阈值
  比较，排序好但未校准的概率会给出错误的阈值决策）。
- **基线对照**：固定常数、仅「剩余可摸牌数」的单特征模型、逻辑回归。
- **特征消融**：按组移除后重训，输出各组贡献。
- **数据划分按整局分开**：训练 / 校准 / 验证各用独立种子单独生成。

**为什么需要独立的校准集**：树模型在尾部会系统性低估（实测 ``[0.8,0.9)`` 预测 0.847
而实际 0.908）。等渗回归能修正它，但**必须在模型没见过、也不是评估用的第三份数据上拟合**，
否则校准本身就是过拟合，验证集上的好看数字没有意义。

用法::

    uv run python tools/train_opponent.py \\
        --train 'data/opponent_train.part*.npz' \\
        --calib 'data/opponent_calib.part*.npz' \\
        --valid 'data/opponent_valid.part*.npz' \\
        --out models/opponent_model.json
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
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
    "clock": ("wall_remaining", "draws_left", "progress"),
    "table_other": ("gods_seen", "table_melds"),
    "observer": ("my_concealed", "my_melds", "my_gods"),
}

# 校准的关注区间：风险聚合 q = ∏(1 − pᵢ) 会把这里的误差连乘放大
TAIL_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


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


def load_or_none(pattern: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        return load(pattern)
    except SystemExit as exc:
        print(f"  跳过：{exc}", file=sys.stderr)
        return None


def make_model(trees: int, depth: int, lr: float) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=trees, max_depth=depth, learning_rate=lr, subsample=0.8, random_state=20260924
    )


def report(tag: str, y: np.ndarray, probability: np.ndarray) -> None:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    print(
        f"  {tag:26s} AUC {roc_auc_score(y, probability):.4f}  "
        f"对数损失 {log_loss(y, clipped):.4f}  布里尔 {brier_score_loss(y, probability):.4f}"
    )


def bin_table(y: np.ndarray, probability: np.ndarray, tag: str) -> None:
    print(f"    {tag}")
    for low, high in zip(TAIL_EDGES, TAIL_EDGES[1:]):
        mask = (probability >= low) & (
            probability < high if high < 1.0 else probability <= high
        )
        if mask.sum() == 0:
            continue
        predicted = float(probability[mask].mean())
        actual = float(y[mask].mean())
        flag = "  <== 偏高" if actual - predicted > 0.03 else ""
        print(
            f"      [{low:.1f},{high:.1f}) → 预测 {predicted:.3f}  实际 {actual:.3f}  "
            f"偏差 {actual - predicted:+.3f}  样本 {int(mask.sum()):6d}{flag}"
        )


def apply_tail(probability: np.ndarray, isotonic: IsotonicRegression, threshold: float) -> np.ndarray:
    """只在概率不低于 ``threshold`` 处施加校准，与运行时 ``applies_from`` 语义一致。"""
    out = probability.copy()
    mask = probability >= threshold
    out[mask] = isotonic.predict(probability[mask])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练对手听牌模型")
    parser.add_argument("--train", default="data/opponent_train.part*.npz")
    parser.add_argument("--calib", default="data/opponent_calib.part*.npz")
    parser.add_argument("--valid", default="data/opponent_valid.part*.npz")
    parser.add_argument("--out", default="models/opponent_model.json")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--skip-ablation", action="store_true", help="跳过特征消融以节省时间")
    parser.add_argument(
        "--calibration-mode",
        choices=("none", "isotonic", "tail"),
        default="none",
        help="导出时采用哪种校准：none 原始概率 / isotonic 全量等渗 / tail 仅修高概率尾部",
    )
    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=0.75,
        help="tail 模式下施加校准的概率下限",
    )
    args = parser.parse_args(argv)

    x_train, y_train = load(args.train)
    print(f"训练集 {x_train.shape[0]} 条，正例率 {y_train.mean():.1%}")

    model = make_model(args.trees, args.depth, args.lr)
    model.fit(x_train, y_train)

    calibration_split = load_or_none(args.calib)
    validation_split = load_or_none(args.valid)

    full_isotonic: IsotonicRegression | None = None
    tail_isotonic: IsotonicRegression | None = None
    if calibration_split is not None:
        x_calib, y_calib = calibration_split
        print(f"校准集 {x_calib.shape[0]} 条（独立种子），正例率 {y_calib.mean():.1%}")
        raw_calib = model.predict_proba(x_calib)[:, 1]
        full_isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw_calib, y_calib)
        print(f"  全量等渗分段数 {len(full_isotonic.X_thresholds_)}")
        mask = raw_calib >= args.tail_threshold
        if int(mask.sum()) >= 100:
            tail_isotonic = IsotonicRegression(out_of_bounds="clip").fit(
                raw_calib[mask], y_calib[mask]
            )
            print(
                f"  尾部等渗（p ≥ {args.tail_threshold}）样本 {int(mask.sum())}，"
                f"分段数 {len(tail_isotonic.X_thresholds_)}"
            )
        else:
            print(f"  尾部样本仅 {int(mask.sum())}，不足以拟合尾部校准")

    if validation_split is not None:
        x_valid, y_valid = validation_split
        probability = model.predict_proba(x_valid)[:, 1]
        print(f"验证集 {x_valid.shape[0]} 条（独立种子），正例率 {y_valid.mean():.1%}")
        report("模型 GBDT（未校准）", y_valid, probability)
        if full_isotonic is not None:
            report("模型 GBDT + 全量等渗", y_valid, full_isotonic.predict(probability))
        if tail_isotonic is not None:
            report(
                f"模型 GBDT + 尾部等渗(p≥{args.tail_threshold})",
                y_valid,
                apply_tail(probability, tail_isotonic, args.tail_threshold),
            )
        report("基线：固定常数", y_valid, np.full_like(probability, y_train.mean()))
        single = FEATURE_NAMES.index("draws_left")
        report("基线：仅剩余可摸牌数", y_valid, 1.0 - x_valid[:, single] / 63.0)
        logistic = LogisticRegression(max_iter=1000).fit(x_train, y_train)
        report("基线：逻辑回归", y_valid, logistic.predict_proba(x_valid)[:, 1])

        print("\n  可靠性（10 分箱）")
        bin_table(y_valid, probability, "未校准")
        if full_isotonic is not None:
            bin_table(y_valid, full_isotonic.predict(probability), "全量等渗后")
        if tail_isotonic is not None:
            bin_table(
                y_valid,
                apply_tail(probability, tail_isotonic, args.tail_threshold),
                f"尾部等渗后（仅 p ≥ {args.tail_threshold} 被改动）",
            )

    importance = np.argsort(model.feature_importances_)[::-1]
    print(
        "\n  特征重要度（不纯度，在训练集上计算，仅作参考——它偏向连续高基数特征，"
        "判断重要性请以消融为准）"
    )
    print(
        "    top8:",
        ", ".join(f"{FEATURE_NAMES[i]}={model.feature_importances_[i]:.3f}" for i in importance[:8]),
    )

    if validation_split is not None and not args.skip_ablation:
        x_valid, y_valid = validation_split
        print("  特征消融（按组移除后重训，在验证集上评估）")
        full = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
        for group, names in FEATURE_GROUPS.items():
            drop = [FEATURE_NAMES.index(name) for name in names]
            keep = [index for index in range(FEATURE_COUNT) if index not in drop]
            ablated = make_model(args.trees, args.depth, args.lr).fit(x_train[:, keep], y_train)
            auc = roc_auc_score(y_valid, ablated.predict_proba(x_valid[:, keep])[:, 1])
            print(f"    去掉 {group:16s} AUC {auc:.4f}  （全量 {full:.4f}，Δ {auc - full:+.4f}）")

    chosen = {"none": None, "isotonic": full_isotonic, "tail": tail_isotonic}[args.calibration_mode]
    applied_from = args.tail_threshold if args.calibration_mode == "tail" else None
    gbdt.export_sklearn_model(
        model,
        Path(args.out),
        feature_names=FEATURE_NAMES,
        link=gbdt.SIGMOID,
        calibration=chosen,
        calibration_from=applied_from,
    )
    print(f"\n已导出 {Path(args.out).resolve()}（校准模式 {args.calibration_mode}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
