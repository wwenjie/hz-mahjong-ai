"""对手听牌模型诊断：它学到的是「对手行为」还是「牌墙时钟」？（tasks.md 6B.6 补充）

首轮训练暴露两个疑点：

1. GBDT（AUC 0.8937 / 对数损失 0.3462）**没有跑赢逻辑回归**（0.8941 / 0.3454）。
2. 基于不纯度的特征重要度把 ``progress``(0.260)、``wall_remaining``(0.241)、
   ``draws_left``(0.122) 排在最前——三者都是「牌局进行到多晚」的共线编码，合计 0.62。
   而消融实验（在验证集上测）却说去掉整组 ``table`` 反而微增（Δ +0.0006）。

重要度与消融结论相反，是梯度提升树的典型陷阱：不纯度重要度在**训练集**上计算，
且偏向连续高基数特征。因此必须用一个能直接回答问题的实验来定论：

- **时钟单独**能到多少 AUC → 给出「什么都不懂，只看时间」的基线上限。
- **对手行为单独**能到多少 AUC → 对手特征本身的表观能力。
- **时钟分层后的 AUC**（在每个 ``progress`` 分层内部分别算 AUC 再按样本数加权）
  → 这才是关键：在**时间被固定**的情况下，对手行为特征是否仍能把样本排序？
  若接近 0.5，说明模型只是时钟的代理，对手特征没有增量信息。

同时用导出的 JSON 模型跑一遍全量验证集，既校验导出保真度，也验证纯 Python 求值器
（tasks.md 6B.8）在大样本下的可用性。

用法::

    uv run python tools/diagnose_opponent.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from majiang.strategy import gbdt
from majiang.strategy.opponent_features import FEATURE_NAMES

CLOCK = ("progress", "wall_remaining", "draws_left")
TARGET = tuple(name for name in FEATURE_NAMES if name.startswith("target_"))
MODEL_PATH = Path("models/opponent_model.json")


def load(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"没有匹配到数据分片: {pattern}")
    x = np.concatenate([np.load(name)["x"] for name in files]).astype(np.float32)
    y = np.concatenate([np.load(name)["y"] for name in files]).astype(np.float32)
    return x, y


def columns(names: tuple[str, ...]) -> list[int]:
    return [FEATURE_NAMES.index(name) for name in names]


def report(tag: str, y: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, probability)
    print(
        f"  {tag:26s} AUC {auc:.4f}  对数损失 {log_loss(y, clipped):.4f}  "
        f"布里尔 {brier_score_loss(y, probability):.4f}"
    )
    return auc


def stratified_auc(y: np.ndarray, probability: np.ndarray, strata: np.ndarray, bins: int = 10) -> float:
    """在分层内部分别计算 AUC 再按样本数加权——即「固定分层变量后」的排序能力。"""
    edges = np.quantile(strata, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    total, weighted = 0, 0.0
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        mask = (strata >= low) & (strata < high if index < len(edges) - 2 else strata <= high)
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            continue
        total += int(mask.sum())
        weighted += roc_auc_score(y[mask], probability[mask]) * int(mask.sum())
        print(f"      分层 [{low:.3f},{high:.3f})  样本 {int(mask.sum()):6d}  "
              f"AUC {roc_auc_score(y[mask], probability[mask]):.4f}")
    return weighted / total if total else float("nan")


def make_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=20260924
    )


def main() -> int:
    x_train, y_train = load("data/opponent_train.part*.npz")
    x_valid, y_valid = load("data/opponent_valid.part*.npz")
    print(f"训练集 {x_train.shape[0]} 条 / 验证集 {x_valid.shape[0]} 条")

    clock = columns(CLOCK)
    target = columns(TARGET)

    print("\n[1] 特征子集的表观能力（同一套超参，仅换输入特征）")
    subsets: dict[str, np.ndarray] = {
        "仅时钟(3 特征)": np.array(clock),
        "仅对手行为(13 特征)": np.array(target),
        "时钟+对手行为": np.array(sorted(clock + list(target))),
        "全量(21 特征)": np.arange(len(FEATURE_NAMES)),
    }
    fits: dict[str, np.ndarray] = {}
    for tag, cols in subsets.items():
        model = make_model().fit(x_train[:, cols], y_train)
        fits[tag] = model.predict_proba(x_valid[:, cols])[:, 1]
        report(tag, y_valid, fits[tag])

    print("\n[2] 时钟分层后的 AUC —— 关键判据")
    strata = x_valid[:, FEATURE_NAMES.index("progress")]
    print("    仅对手行为模型（在固定 progress 分层内）")
    conditional = stratified_auc(y_valid, fits["仅对手行为(13 特征)"], strata)
    print(f"    → 加权条件 AUC {conditional:.4f}")

    print("\n[3] 对手特征是否带来增量：时钟 vs 时钟+对手行为")
    clock_only = log_loss(y_valid, np.clip(fits["仅时钟(3 特征)"], 1e-6, 1 - 1e-6))
    both = log_loss(y_valid, np.clip(fits["时钟+对手行为"], 1e-6, 1 - 1e-6))
    print(f"    对数损失 {clock_only:.4f} → {both:.4f}（Δ {both - clock_only:+.4f}，负值表示对手特征有增量）")

    print("\n[4] 导出模型保真度（纯 Python 求值器跑全量验证集，tasks.md 6B.8）")
    ensemble = gbdt.TreeEnsemble.load(MODEL_PATH, expected_features=len(FEATURE_NAMES))
    python_probability = np.array([ensemble.predict(row) for row in x_valid])
    report("导出 JSON（纯 Python）", y_valid, python_probability)
    gap = float(np.max(np.abs(python_probability - fits["全量(21 特征)"])))
    print(f"    与 sklearn 预测的最大绝对偏差 {gap:.2e}")

    print("\n[5] 细粒度校准（10 分箱，验证集）")
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        low, high = edges[index], edges[index + 1]
        mask = (python_probability >= low) & (
            python_probability < high if index < 9 else python_probability <= high
        )
        if mask.sum() == 0:
            continue
        actual = y_valid[mask].mean()
        print(f"    [{low:.1f},{high:.1f}) → 预测均值 {python_probability[mask].mean():.3f}  "
              f"实际 {actual:.3f}  偏差 {actual - python_probability[mask].mean():+.3f}  样本 {int(mask.sum()):6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
