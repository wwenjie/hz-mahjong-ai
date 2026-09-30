"""训练「终局首名概率」模型（P(首名) 目标对齐线）。

数据来自 ``tools/gen_pfirst_data.py``（整场自对弈、局况已接线）。特征 = 局面特征
（``features.extract``，与线上同一份）+ 4 个局况特征（rank / gap_to_first /
rounds_left / score_share，定义见生成脚本）。

双目标（采纳评审建议）：

- 主目标 ``y_first``：终局是否首名（0/1），GBDT 分类。
- 辅助目标 ``y_score``：整场累计净分（回归），用于双目标权衡（首名概率 vs 净分
  期望），由调用方决定如何融合。

模型与线上价值模型同构：``gbdt.TreeEnsemble`` 纯 Python 求值，零运行时依赖，
导出 JSON 可直接进交付路径。

用法::

    uv run python tools/train_pfirst.py data/pfirst_train.part*.npz --out models/pfirst.json
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from majiang.strategy import features
from majiang.strategy.gbdt import SIGMOID, export_sklearn_model

# 与 gen_pfirst_data._extra 一致
EXTRA_NAMES = ("rank", "gap_to_first", "rounds_left", "score_share")
ALL_NAMES = list(features.FEATURE_NAMES) + list(EXTRA_NAMES)


def _load(pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, yf, ys = [], [], []
    for path in sorted(glob.glob(pattern)):
        with np.load(path) as d:
            xs.append(d["x"])
            yf.append(d["y_first"])
            ys.append(d["y_score"])
    return (
        np.concatenate(xs),
        np.concatenate(yf),
        np.concatenate(ys),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 P(首名) 模型")
    parser.add_argument("data", help="数据分片 glob，如 data/pfirst_train.part*.npz")
    parser.add_argument("--out", default="models/pfirst.json")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="消融：分别训练「全部特征」与「去掉 4 个局况特征」两个模型，比 AUC——"
        "直接检验局况特征是否在手牌特征之上带来增量信号（本线的核心问题）。",
    )
    args = parser.parse_args(argv)

    x, y_first, y_score = _load(args.data)
    print(f"样本 {x.shape[0]} 条，特征 {x.shape[1]} 维", file=sys.stderr)
    print(f"y_first 阳性率 {y_first.mean():.3f}", file=sys.stderr)
    print(f"y_score 范围 [{y_score.min():.0f}, {y_score.max():.0f}]", file=sys.stderr)

    if args.ablate:
        _run_ablation(x, y_first, args)
        return 0

    # 简单按场次切分验证集（样本按场次连续写入，尾部 10% 做验证）
    n_val = max(1, x.shape[0] // 10)
    x_train, x_val = x[:-n_val], x[-n_val:]
    y_train, y_val = y_first[:-n_val], y_first[-n_val:]

    auc, acc, model = _fit(x_train, y_train, x_val, y_val, args)

    # 导出为线上可用的 JSON（与 value 模型同构；SIGMOID 链接 ⇒ predict 输出概率）
    export_sklearn_model(
        model,
        args.out,
        feature_names=ALL_NAMES,
        link=SIGMOID,
    )
    print(f"模型已写入 {Path(args.out).resolve()}（AUC {auc:.4f}）")
    return 0


def _fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, HistGradientBoostingClassifier]:
    started = time.perf_counter()
    model = HistGradientBoostingClassifier(
        max_depth=args.max_depth,
        max_iter=args.max_iter,
        learning_rate=args.lr,
        early_stopping=True,
        random_state=20260930,
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_val)[:, 1]
    pred = (proba > 0.5).astype(float)
    auc = roc_auc_score(y_val, proba)
    acc = accuracy_score(y_val, pred)
    elapsed = time.perf_counter() - started
    print(
        f"验证集 AUC {auc:.4f} 准确率 {acc:.4f}（n={len(y_val)}），用时 {elapsed:.1f}s",
        file=sys.stderr,
    )
    return auc, acc, model


def _run_ablation(x: np.ndarray, y_first: np.ndarray, args: argparse.Namespace) -> None:
    """同一切分上对比「全特征」vs「去局况特征」的验证 AUC。"""
    n_val = max(1, x.shape[0] // 10)
    x_train, x_val = x[:-n_val], x[-n_val:]
    y_train, y_val = y_first[:-n_val], y_first[-n_val:]
    n_base = features.FEATURE_COUNT

    auc_full, _, _ = _fit(x_train, y_train, x_val, y_val, args)
    print(f"[消融] 全特征（{x.shape[1]} 维）AUC = {auc_full:.4f}")

    auc_base, _, _ = _fit(
        x_train[:, :n_base], y_train, x_val[:, :n_base], y_val, args
    )
    print(f"[消融] 仅手牌特征（{n_base} 维，去局况）AUC = {auc_base:.4f}")
    print(f"[消融] 局况特征增量 ΔAUC = {auc_full - auc_base:+.4f}")
    if auc_full - auc_base < 0.005:
        print("[消融] 增量 < 0.005：局况特征几乎无独立信号，P(首名)线应止损重审")
    else:
        print("[消融] 局况特征有独立信号，值得继续推进接线")


if __name__ == "__main__":
    sys.exit(main())
