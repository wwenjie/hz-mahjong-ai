"""生成「终局首名概率」模型的整场自对弈数据（P(首名) 目标对齐线）。

与 ``gen_value_data.py``（单局净分回归）**刻意分开**：那条管线的 target 是单局
``result.scores[seat]``，没有跨局概念；本工具用 ``run_match``（整场、局况已接线，
见 2026-09-30 的 sim 局况接线修复），target 是**整场终局该座是否首名**（0/1）。

每条样本 = （决策时刻「打后」局面特征 + 4 个局况特征， 两个 target）：

- 局面特征：复用 ``majiang.strategy.features.extract``，与线上推理同一份代码，不改它。
- 局况特征（独立于 ``features.py``，不动 A 的地盘）：
  ``(当前名次 1-4, 与第一名分差, 剩余局数, 相对场分)``。
  ``相对场分`` = 我的累计净分 ÷ max(|累计| 之和, 1)，把分差归一化到 [0,1]。
- 双 target（采纳评审建议，避免单标签噪声）：
  ``y_first`` = 终局是否首名（并列第一记 1）；``y_score`` = 整场终局累计净分。

局况未知（空 scores）的决策点**跳过**——那是单局调用路径，不混入本数据集。

用法::

    uv run python tools/gen_pfirst_data.py --matches 400 --out data/pfirst_train.npz
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
from majiang.strategy import features
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

BATCH_FLUSH = 200_000
EXTRA_FEATURE_NAMES = ("rank", "gap_to_first", "rounds_left", "score_share")


def _extra(situation) -> list[float]:
    """4 个局况特征（仅当 scores 已知时调用）。"""
    table = situation.table
    scores = table.scores
    me = scores[situation.seat]
    rank = sum(1 for v in scores if v > me) + 1
    gap = max(scores) - me
    left = table.rounds_total - table.round_no
    total_abs = sum(abs(v) for v in scores)
    share = me / total_abs if total_abs else 0.0
    return [float(rank), float(gap), float(left), float(share)]


class Recorder:
    """包住策略，记录「打后」局面特征 + 局况特征；整场结束后由调用方补 target。

    记「打后」与 ``gen_value_data.Recorder`` 同因：推理时对每个候选查的正是
    「打这张之后的局面」，记打前会让所有候选特征相同（train-serve skew）。
    局况未知（空 scores）的决策点跳过，不污染数据集。
    """

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
            self.rows.append(features.extract(after) + _extra(after))
        return choice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 P(首名) 模型训练数据")
    parser.add_argument("--matches", type=int, default=400, help="整场数（每场 8 局）")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--out", default="data/pfirst_train.npz")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.QUALIFIER.value)
    args = parser.parse_args(argv)

    config = PolicyConfig.for_mode(Mode(args.mode))
    rng = random.Random(args.seed)
    feat_buf: list[list[float]] = []
    first_buf: list[float] = []
    score_buf: list[float] = []
    parts: list[Path] = []
    started = time.perf_counter()

    for m in range(args.matches):
        # 每场独立 seed；start_dealer 轮换，避免庄闲座次偏差（与 run_batch 一致）。
        recorders = [
            Recorder(HeuristicDecider(config), seat) for seat in range(4)
        ]
        result = run_match(
            recorders,
            rounds=args.rounds,
            seed=rng.randrange(1 << 30),
            start_dealer=m % 4,
        )
        # 「首名」按**整场累计净分**判定——平台排名的第一键就是总得分
        # （B' 用 C 的 6 房样本核实，见 080dbb9 commit message）。
        # 曾误改为 place_points（逐局名次分累计），那是模拟器内部指标而非平台口径。
        finals = [s.total_score for s in result.seats]
        top = max(finals)
        for seat, recorder in enumerate(recorders):
            y_first = 1.0 if finals[seat] == top else 0.0
            y_score = float(finals[seat])
            for row in recorder.rows:
                feat_buf.append(row)
                first_buf.append(y_first)
                score_buf.append(y_score)
        if len(feat_buf) >= BATCH_FLUSH or m + 1 == args.matches:
            parts.append(_flush(feat_buf, first_buf, score_buf, Path(args.out), len(parts)))
            elapsed = time.perf_counter() - started
            print(
                f"  已跑 {m + 1}/{args.matches} 场，累计样本 {len(feat_buf)}，"
                f"用时 {elapsed:.0f}s",
                file=sys.stderr,
            )
            feat_buf.clear()
            first_buf.clear()
            score_buf.clear()

    elapsed = time.perf_counter() - started
    print(f"完成：{args.matches} 场（每场 {args.rounds} 局），输出 {len(parts)} 个分片，用时 {elapsed:.0f}s")
    for part in parts:
        print(f"  {part.resolve()}")
    return 0


def _flush(
    feat_buf: list[list[float]],
    first_buf: list[float],
    score_buf: list[float],
    path: Path,
    part_index: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.stem}.part{part_index:03d}.npz")
    n_feat = features.FEATURE_COUNT + len(EXTRA_FEATURE_NAMES)
    assert all(len(r) == n_feat for r in feat_buf), "特征列数不一致"
    np.savez_compressed(
        part,
        x=np.asarray(feat_buf, dtype=np.float32),
        y_first=np.asarray(first_buf, dtype=np.float32),
        y_score=np.asarray(score_buf, dtype=np.float32),
    )
    return part


if __name__ == "__main__":
    sys.exit(main())
