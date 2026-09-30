"""P(首名) vs E(净分) 的特征重要性对照——首名目标到底吃哪些信号？

接续 analyze_pfirst_divergence.py：分歧分析回答「两目标在哪些局面分歧」，
本脚本回答「两目标各自依赖哪些特征」——若 P(首名) 的 top 特征全是局况
（rank/gap/rounds_left/score_share），说明它本质是个**局况调制器**，接入方式
应是「局况条件化打分」而非「并列第二个模型」；若手牌特征也吃重，则说明
「什么牌型利于争首名」有独立于「什么牌型利于得分」的结构。

输出：两模型各自 top-12 特征 + 局况 4 维合计占比对照。
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from majiang.strategy import features

EXTRA = ("rank", "gap_to_first", "rounds_left", "score_share")
NAMES = list(features.FEATURE_NAMES) + list(EXTRA)


def _top(names: list[str], imp: np.ndarray, k: int = 12) -> list[str]:
    order = np.argsort(imp)[::-1]
    return [f"{names[i]}({imp[i]:.3f})" for i in order[:k]]


def main() -> int:
    with np.load("data/pfirst_train.part000.npz") as d:
        x = d["x"]
        y_first = d["y_first"]
        y_score = d["y_score"]
    n_val = x.shape[0] // 5
    x_tr, x_va = x[:-n_val], x[-n_val:]
    print(f"样本 {x.shape[0]}（{x.shape[0]-n_val} 训 / {n_val} 验）", file=sys.stderr)

    clf = GradientBoostingClassifier(max_depth=6, n_estimators=200,
                                     learning_rate=0.05, random_state=20261001)
    clf.fit(x_tr, y_first[:-n_val])
    reg = GradientBoostingRegressor(max_depth=6, n_estimators=200,
                                    learning_rate=0.05, random_state=20261001)
    reg.fit(x_tr, y_score[:-n_val])

    situ_idx = [NAMES.index(nm) for nm in EXTRA]
    hand_imp_first = float(sum(clf.feature_importances_[i] for i in range(29)))
    situ_imp_first = float(sum(clf.feature_importances_[i] for i in situ_idx))
    hand_imp_score = float(sum(reg.feature_importances_[i] for i in range(29)))
    situ_imp_score = float(sum(reg.feature_importances_[i] for i in situ_idx))

    print("\n=== 局况 4 维合计重要性占比 ===")
    print(f"P(首名): 局况 {situ_imp_first:.1%} / 手牌 {hand_imp_first:.1%}")
    print(f"E(净分): 局况 {situ_imp_score:.1%} / 手牌 {hand_imp_score:.1%}")

    print("\n=== P(首名) top-12 ===")
    for s in _top(NAMES, clf.feature_importances_):
        print(" ", s)
    print("\n=== E(净分) top-12 ===")
    for s in _top(NAMES, reg.feature_importances_):
        print(" ", s)

    print("\n=== 判读指引 ===")
    print("P(首名) 局况占比 ≫ E(净分) ⇒ 首名目标主要靠局况调制 ⇒ 接入形态=局况条件化修正")
    print("两者局况占比接近 ⇒ P(首名) 并未学到独立的局况用法 ⇒ 接入价值存疑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
