#!/usr/bin/env python
"""生成 BC 数据集（教师 = 主仓冻结版本 v5）。

与 gen_bc_data.py 的唯一区别：教师从 HeuristicDecider 默认档
换成 versions.build("v5", Mode.QUALIFIER)。观测/记录逻辑完全一致。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.obs import situation_to_obs, obs_to_flat


class BCData:
    def __init__(self):
        self.obs_list: list[dict[str, np.ndarray]] = []
        self.y_list: list[int] = []

    def __len__(self):
        return len(self.obs_list)

    def add(self, obs, action_tile: int):
        self.obs_list.append(obs)
        self.y_list.append(action_tile)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        x_flat = np.stack([obs_to_flat(obs) for obs in self.obs_list])
        y = np.asarray(self.y_list, dtype=np.int64)
        fields = {}
        for key in self.obs_list[0].keys():
            fields[key] = np.stack([obs[key] for obs in self.obs_list])
        np.savez_compressed(path, x_flat=x_flat, y=y, **fields)


class RecordingV5:
    """包住主仓 v5，边对局边记录出牌决策。"""

    def __init__(self, acc: BCData, *, label: str = "v5[rec]"):
        from majiang.strategy.policy import Mode
        from majiang.strategy.versions import build
        self.inner = build("v5", Mode.QUALIFIER)
        self.acc = acc
        self.name = label
        self.last_reason = ""
        self.last_detail: dict[str, object] = {}

    def configure(self, tournament) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = getattr(self.inner, "last_reason", "")
        self.last_detail = dict(getattr(self.inner, "last_detail", {}) or {})
        if (
            choice is not None
            and getattr(choice, "kind", None) == "discard"
            and choice.tile is not None
            and situation.drawn_tile is not None
        ):
            obs = situation_to_obs(situation, situation.seat)
            self.acc.add(obs, choice.tile)
        return choice


def generate_bc_data(*, matches: int, rounds: int = 8, seed: int = 20261001) -> BCData:
    from majiang.sim.batch import run_match
    acc = BCData()
    for index in range(matches):
        deciders = [RecordingV5(acc) for _ in range(4)]
        run_match(
            deciders,
            rounds=rounds,
            base_score=1,
            seed=seed * 100003 + index,
            start_dealer=index % 4,
        )
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 BC 数据集（教师=v5）")
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    data = generate_bc_data(matches=args.matches, rounds=args.rounds, seed=args.seed)
    dt = time.perf_counter() - t0
    data.save(args.out)
    print(
        f"[v5] 样本 {len(data):6d}  局 {args.matches * args.rounds:4d}  "
        f"用时 {dt:6.1f}s  -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
