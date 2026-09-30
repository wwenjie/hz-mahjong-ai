"""P(首名) vs E(整场净分) 分歧分析——双目标接入决策链前的安全第一步。

回答一个问题：**「P(首名) 高但 E(净分) 低」（或反过来）的局面长什么样？**
若分歧集中在特定局况（如 rank=1 领先守分 / rank=4 落后搏杀），说明双目标修正项
有真实增益空间；若两目标高度一致（分歧全是噪声），则说明 E(净分) 已经隐含了
首名目标，P(首名) 模型不值得接入决策链——这正是 A 15:50 悬置的「P(首名) 模型
用法定位」问题。

方法：同一份 33 维特征（29 手牌 + 4 局况）上，
- P(首名)：已训好的 ``models/pfirst.json``（GBDT 分类，AUC 0.8248）
- E(净分)：同数据训一个 GBDT 回归器（y_score 是整场累计净分）

分歧度量：把两个预测各自转成百分位（rank-normalize 到 [0,1]），
``divergence = pct(P_first) - pct(E_score)``。看 |d| 最大的局面在 4 个局况
特征（rank / gap_to_first / rounds_left / score_share）上的分布 vs 全体。

用法::

    uv run python tools/analyze_pfirst_divergence.py data/pfirst_train.part000.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import rankdata

from majiang.strategy.gbdt import TreeEnsemble


def _pct(values: np.ndarray) -> np.ndarray:
    """秩归一化到 [0,1]：消除两个目标量纲差异，只比较「相对好坏」。"""
    return (rankdata(values) - 1) / max(len(values) - 1, 1)


def _profile(
    name: str,
    mask: np.ndarray,
    extra: np.ndarray,
    y_first: np.ndarray,
    y_score: np.ndarray,
    diverg: np.ndarray,
) -> None:
    n = int(mask.sum())
    if n == 0:
        print(f"\n[{name}] 空集", file=sys.stderr)
        return
    cols = ("rank", "gap_to_first", "rounds_left", "score_share")
    print(f"\n[{name}] n={n}（占全体 {n / len(mask) * 100:.1f}%）")
    print(f"  y_first 均值 {y_first[mask].mean():.3f}（全体 {y_first.mean():.3f}）")
    print(f"  y_score 均值 {y_score[mask].mean():+.1f}（全体 {y_score.mean():+.1f}）")
    for i, col in enumerate(cols):
        v = extra[mask, i]
        a = extra[:, i]
        print(f"  {col:>12}: 均值 {v.mean():+.3f}（全体 {a.mean():+.3f}）")
    print(f"  divergence 均值 {diverg[mask].mean():+.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P(首名) vs E(净分) 分歧分析")
    parser.add_argument("data", help="pfirst npz 分片")
    parser.add_argument("--model", default="models/pfirst.json")
    args = parser.parse_args(argv)

    with np.load(args.data) as d:
        x = d["x"]
        y_first = d["y_first"]
        y_score = d["y_score"]
    extra = x[:, -4:]  # 4 个局况特征在尾部（与 gen_pfirst_data 约定一致）
    print(f"样本 {x.shape[0]}，特征 {x.shape[1]} 维", file=sys.stderr)

    # P(首名)：加载已训模型
    pf = TreeEnsemble.load(args.model)
    p_first = np.array([pf.predict(row) for row in x[::1]], dtype=np.float64)

    # E(净分)：同特征训回归器（分析用，80/20 切分防过拟合读数）
    n_val = x.shape[0] // 5
    reg = GradientBoostingRegressor(
        max_depth=6, n_estimators=200, learning_rate=0.05, random_state=20261001
    )
    reg.fit(x[:-n_val], y_score[:-n_val])
    e_score = reg.predict(x)
    r2 = reg.score(x[-n_val:], y_score[-n_val:])
    print(f"E(净分) 回归器验证 R²={r2:.3f}（n={n_val}）", file=sys.stderr)

    # 分歧 = 两个目标的秩差
    diverg = _pct(p_first) - _pct(e_score)
    print(f"\nP(首名) 与 E(净分) 的秩相关: {np.corrcoef(_pct(p_first), _pct(e_score))[0,1]:+.4f}")
    print(f"|divergence| 分布: p50={np.percentile(np.abs(diverg),50):.3f} "
          f"p90={np.percentile(np.abs(diverg),90):.3f} "
          f"p99={np.percentile(np.abs(diverg),99):.3f}")

    # 分歧最大的两端画像
    hi = np.percentile(diverg, 95)
    lo = np.percentile(diverg, 5)
    _profile("P(首名)≫E(净分)（前5%：搏杀型局面？）",
             diverg >= hi, extra, y_first, y_score, diverg)
    _profile("P(首名)≪E(净分)（后5%：守分型局面？）",
             diverg <= lo, extra, y_first, y_score, diverg)
    _profile("两目标一致（中段 50%）",
             np.abs(diverg) <= np.percentile(np.abs(diverg), 50),
             extra, y_first, y_score, diverg)

    # 按局况分桶看分歧均值——分歧是否集中在特定局况
    print("\n=== 分歧按局况分桶（divergence 均值）===")
    for col_i, col in enumerate(("rank", "rounds_left")):
        v = extra[:, col_i]
        if col == "rank":
            bins = [(0.5, 1.5, "rank1 领先"), (1.5, 2.5, "rank2"),
                    (2.5, 3.5, "rank3"), (3.5, 4.5, "rank4 落后")]
        else:
            bins = [(0.5, 2.5, "剩1-2局 末段"), (2.5, 4.5, "剩3-4局"),
                    (4.5, 8.5, "剩5+局 早段")]
        for lo_b, hi_b, label in bins:
            m = (v >= lo_b) & (v < hi_b)
            if m.sum() > 0:
                print(f"  {col} [{label}] n={int(m.sum()):>6}  "
                      f"divergence 均值 {diverg[m].mean():+.4f}  "
                      f"|d|均值 {np.abs(diverg[m]).mean():.4f}")

    print("\n=== 结论判读指引 ===")
    print("秩相关 >0.9 且两端画像无局况聚集 ⇒ E(净分) 已隐含首名目标，P(首名)线可判'不接入'")
    print("两端画像明显聚集（如 rank1 守/rank4 搏）且 |d| p90 >0.3 ⇒ 双目标修正有真实空间，建议接 A/B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
