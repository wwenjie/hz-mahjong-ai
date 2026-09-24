"""生成价值模型的自对弈数据（tasks.md 5.15）。

数据以**启发式自对弈**产生——必须用将来要部署的那个策略：用烂策略生成的数据会让模型
学到「好策略永远不会进入的状态」，造成分布失配。

每条样本 = （决策时刻的公开局面特征，本局本人最终得分）。目标变量由模拟器直接给出，
**精确、无需标注**。特征提取复用 `majiang.strategy.features`，与线上推理同一份代码。

用法::

    uv run python tools/gen_value_data.py --rounds 4000 --out data/value_train.npz
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
from majiang.sim.round import run_round
from majiang.strategy import features
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

BATCH_FLUSH = 200_000


class Recorder:
    """包住策略，记录**打出某张牌之后的局面**特征；回合结束后补上目标值。

    记录「打后」而非「打前」是刻意的：推理时对每个候选查询的正是「打这张之后的局面」，
    若记录打前状态，则所有候选的特征完全相同、模型毫无区分力（train-serve skew）。
    """

    def __init__(self, inner: object) -> None:
        self.inner = inner
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
        ):
            after = replace(situation, hand=situation.hand.without_tile(choice.tile))
            self.rows.append(features.extract(after))
        return choice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成价值模型训练数据")
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--out", default="data/value_train.npz")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--mode", choices=[mode.value for mode in Mode], default=Mode.QUALIFIER.value)
    args = parser.parse_args(argv)

    config = PolicyConfig.for_mode(Mode(args.mode))
    rng = random.Random(args.seed)
    features_buffer: list[list[float]] = []
    targets_buffer: list[float] = []
    parts: list[Path] = []
    started = time.perf_counter()
    rounds_done = 0

    for index in range(args.rounds):
        recorders = [Recorder(HeuristicDecider(config)) for _ in range(4)]
        result = run_round(recorders, dealer=index % 4, round_no=index + 1, rng=rng)
        for seat, recorder in enumerate(recorders):
            target = float(result.scores[seat])
            for row in recorder.rows:
                features_buffer.append(row)
                targets_buffer.append(target)
        rounds_done += 1
        if len(features_buffer) >= BATCH_FLUSH or index + 1 == args.rounds:
            parts.append(
                _flush(features_buffer, targets_buffer, Path(args.out), len(parts))
            )
            elapsed = time.perf_counter() - started
            print(
                f"  已跑 {rounds_done}/{args.rounds} 局，累计样本 {parts} 文件，"
                f"用时 {elapsed:.0f}s（{rounds_done / max(elapsed, 1e-9):.1f} 局/秒）",
                file=sys.stderr,
            )
            features_buffer.clear()
            targets_buffer.clear()

    elapsed = time.perf_counter() - started
    print(f"完成：{rounds_done} 局，输出 {len(parts)} 个分片，用时 {elapsed:.0f}s")
    for part in parts:
        print(f"  {part.resolve()}")
    return 0


def _flush(
    features_buffer: list[list[float]],
    targets_buffer: list[float],
    path: Path,
    part_index: int,
) -> Path:
    """把缓冲区写成独立分片（避免训练前把全部样本压在内存里）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.stem}.part{part_index:03d}.npz")
    np.savez_compressed(
        part,
        x=np.asarray(features_buffer, dtype=np.float32),
        y=np.asarray(targets_buffer, dtype=np.float32),
    )
    return part


if __name__ == "__main__":
    sys.exit(main())
