"""生成对手听牌模型的训练数据（tasks.md 6B.1–6B.3）。

标签 = **该对手当前是否听牌**——由模拟器的完整手牌给出，是**无噪声的事实**（这与价值
模型的目标形成对照：局分是随机变量的实现值，96% 的方差是运气）。

特征只由公开信息构成、与线上推理共用同一份 `opponent_features`，因此不存在
train-serve skew；标签取自完整手牌属**离线标注**，不进入推理输入（合规）。

用法::

    uv run python tools/gen_opponent_data.py --rounds 1500 --out data/opponent_train.npz
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from majiang.rules import shanten as shanten_module
from majiang.rules.action import DISCARD
from majiang.sim.round import run_round
from majiang.strategy import opponent_features
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

BATCH_FLUSH = 300_000


def is_ready(state: object, seat: int) -> bool:
    """该座位此刻是否听牌——用精确向听判定（与「听牌 ⟺ 0 向听」的既有验证一致）。"""
    seat_state = state.seats[seat]  # type: ignore[attr-defined]
    try:
        return shanten_module.shanten(seat_state.hand, len(seat_state.melds)) == 0
    except shanten_module.ShantenError:
        return False


class Recorder:
    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.state: object | None = None
        self.rows: list[list[float]] = []
        self.labels: list[float] = []

    def configure(self, tournament: object) -> None:
        configure = getattr(self.inner, "configure", None)
        if callable(configure):
            configure(tournament)

    def observe_state(self, state: object) -> None:
        self.state = state

    def choose(self, situation, actions, *, budget_ms: int = 0):
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        if self.state is not None and situation.drawn_tile is not None:
            for other in opponent_features.opponents_of(situation.seat):
                self.rows.append(opponent_features.extract(situation, other))
                self.labels.append(1.0 if is_ready(self.state, other) else 0.0)
        return choice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成对手听牌模型训练数据")
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--out", default="data/opponent_train.npz")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--mode", choices=[mode.value for mode in Mode], default=Mode.QUALIFIER.value)
    args = parser.parse_args(argv)

    config = PolicyConfig.for_mode(Mode(args.mode))
    rng = random.Random(args.seed)
    rows: list[list[float]] = []
    labels: list[float] = []
    parts: list[Path] = []
    started = time.perf_counter()

    for index in range(args.rounds):
        recorders = [Recorder(HeuristicDecider(config)) for _ in range(4)]
        run_round(recorders, dealer=index % 4, round_no=index + 1, rng=rng)
        for recorder in recorders:
            rows.extend(recorder.rows)
            labels.extend(recorder.labels)
        if len(rows) >= BATCH_FLUSH or index + 1 == args.rounds:
            parts.append(_flush(rows, labels, Path(args.out), len(parts)))
            elapsed = time.perf_counter() - started
            positive = sum(labels) / max(1, len(labels))
            print(
                f"  已跑 {index + 1}/{args.rounds} 局，本批样本 {len(rows)}，正例率 {positive:.1%}，"
                f"用时 {elapsed:.0f}s（{(index + 1) / max(elapsed, 1e-9):.1f} 局/秒）",
                file=sys.stderr,
            )
            rows.clear()
            labels.clear()

    print(f"完成：{args.rounds} 局，{len(parts)} 个分片，用时 {time.perf_counter() - started:.0f}s")
    for part in parts:
        print(f"  {part.resolve()}")
    return 0


def _flush(rows: list[list[float]], labels: list[float], path: Path, index: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.stem}.part{index:03d}.npz")
    np.savez_compressed(
        part,
        x=np.asarray(rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.float32),
    )
    return part


if __name__ == "__main__":
    sys.exit(main())
