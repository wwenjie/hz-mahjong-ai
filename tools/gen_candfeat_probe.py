"""candidate_features 信号验证探针：带候选向听特征的 P(首名) 数据生成 + AUC 对照。

回答一个问题：**`cand_keep/lower/raise_shanten` 三个特征在现有 33 维之上有没有
增量信号？** 没有则原型止步（避免 A 浪费评审）；有则把数据交给 A 决定是否接进
``features.py``。

方法：复用 ``gen_pfirst_data`` 的整场自对弈结构，每条样本在「29 手牌 + 4 局况」
尾部**追加 3 个候选特征**（对「打后」局面计算，与 features 同口径）。然后同一份
数据上对比 33 维 vs 36 维的 P(首名) AUC。

用法::

    uv run python tools/gen_candfeat_probe.py --matches 40
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from majiang.rules.action import DISCARD
from majiang.sim.batch import run_match
from majiang.strategy import candidate_features, features
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig


def _extra(situation) -> list[float]:
    """4 个局况特征（与 gen_pfirst_data._extra 同口径，内联避免 tools 包导入）。"""
    table = situation.table
    scores = table.scores
    me = scores[situation.seat]
    rank = sum(1 for v in scores if v > me) + 1
    gap = max(scores) - me
    left = table.rounds_total - table.round_no
    total_abs = sum(abs(v) for v in scores)
    share = me / total_abs if total_abs else 0.0
    return [float(rank), float(gap), float(left), float(share)]


class ProbeRecorder:
    """与 gen_pfirst_data.Recorder 同构，但每行追加 3 个候选向听特征。"""

    def __init__(self, inner: object, seat: int) -> None:
        self.inner = inner
        self.seat = seat
        self.rows: list[list[float]] = []

    def configure(self, tournament: object) -> None:
        configure = getattr(self.inner, "configure", None)
        if callable(configure):
            configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        if (
            choice is not None
            and choice.kind == DISCARD
            and choice.tile is not None
            and situation.drawn_tile is not None
            and situation.hand.counts[choice.tile] > 0
            and situation.table.scores
        ):
            after = replace(situation, hand=situation.hand.without_tile(choice.tile))
            cand = candidate_features.extract(after.hand.counts, after.hand.meld_count)
            self.rows.append(features.extract(after) + _extra(after) + cand)
        return choice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="candidate_features 信号探针")
    parser.add_argument("--matches", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--out", default="data/candfeat_probe.npz")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv)

    config = PolicyConfig.for_mode(Mode.QUALIFIER)
    rng = random.Random(args.seed)
    feat_buf: list[list[float]] = []
    first_buf: list[float] = []
    started = time.perf_counter()

    for m in range(args.matches):
        recorders = [ProbeRecorder(HeuristicDecider(config), seat) for seat in range(4)]
        result = run_match(recorders, rounds=args.rounds, seed=rng.randrange(1 << 30),
                           start_dealer=m % 4)
        finals = [s.total_score for s in result.seats]
        top = max(finals)
        for seat, recorder in enumerate(recorders):
            y = 1.0 if finals[seat] == top else 0.0
            for row in recorder.rows:
                feat_buf.append(row)
                first_buf.append(y)
        if (m + 1) % 10 == 0:
            print(f"  {m+1}/{args.matches} 场，样本 {len(feat_buf)}，"
                  f"用时 {time.perf_counter()-started:.0f}s", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=np.asarray(feat_buf, dtype=np.float32),
        y_first=np.asarray(first_buf, dtype=np.float32),
    )
    print(f"完成：{len(feat_buf)} 样本 × {len(feat_buf[0])} 维 → {out.resolve()}")

    # 同一份数据：33 维 vs 36 维的 AUC 对照
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    x = np.asarray(feat_buf, dtype=np.float32)
    y = np.asarray(first_buf, dtype=np.float32)
    n_val = x.shape[0] // 5
    for label, cols in (("33维(现状)", 33), ("36维(+候选向听)", 36)):
        clf = GradientBoostingClassifier(
            max_depth=6, n_estimators=200, learning_rate=0.05, random_state=20261001
        )
        clf.fit(x[:-n_val, :cols], y[:-n_val])
        auc = roc_auc_score(y[-n_val:], clf.predict_proba(x[-n_val:, :cols])[:, 1])
        print(f"[{label}] 验证 AUC = {auc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
