#!/usr/bin/env python
"""并行 BC 数据生成（多进程池）。

用法::

    PYTHONPATH=src uv run python scripts/gen_bc_data_par.py --train-matches 100 --workers 8

把 matches 分摊到多进程池，每个 worker 跑一部分 matches，最后合并。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.obs import situation_to_obs, obs_to_flat


def _worker_generate(args):
    """子进程：跑若干场，返回 (obs_list, y_list)。"""
    worker_id, matches, rounds, seed = args
    from majiang.sim.batch import run_match
    from majiang.strategy.policy import HeuristicDecider

    obs_list = []
    y_list = []

    class RecordingHeuristic:
        def __init__(self):
            self.inner = HeuristicDecider()
            self.name = "heuristic[rec]"
            self.last_reason = ""
            self.last_detail = {}

        def configure(self, tournament):
            pass

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
                obs_list.append(obs)
                y_list.append(choice.tile)
            return choice

    for index in range(matches):
        deciders = [RecordingHeuristic() for _ in range(4)]
        run_match(
            deciders,
            rounds=rounds,
            base_score=1,
            seed=seed * 100003 + index,
            start_dealer=index % 4,
        )

    return obs_list, y_list


def merge_results(results):
    """合并多个 worker 的结果。"""
    all_obs = []
    all_y = []
    for obs_list, y_list in results:
        all_obs.extend(obs_list)
        all_y.extend(y_list)
    return all_obs, all_y


def save_dataset(obs_list, y_list, path):
    """保存数据集。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 展平版本
    x_flat = np.stack([obs_to_flat(obs) for obs in obs_list])
    y = np.asarray(y_list, dtype=np.int64)

    # dict 版本
    fields = {}
    for key in obs_list[0].keys():
        fields[key] = np.stack([obs[key] for obs in obs_list])

    np.savez_compressed(path, x_flat=x_flat, y=y, **fields)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="并行 BC 数据生成")
    ap.add_argument("--train-matches", type=int, default=100)
    ap.add_argument("--valid-matches", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=20260928)
    ap.add_argument("--valid-seed", type=int, default=771014)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, matches, seed in (
        ("train", args.train_matches, args.train_seed),
        ("valid", args.valid_matches, args.valid_seed),
    ):
        t0 = time.perf_counter()

        # 把 matches 分摊到 workers
        per_worker = matches // args.workers
        remainder = matches % args.workers
        worker_args = []
        offset = 0
        for w in range(args.workers):
            m = per_worker + (1 if w < remainder else 0)
            if m > 0:
                # 每个 worker 用不同的 seed 基础，避免撞车
                worker_seed = seed + w * 1000000
                worker_args.append((w, m, args.rounds, worker_seed))
            offset += m

        # 多进程跑
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker_generate, worker_args)

        obs_list, y_list = merge_results(results)

        dt = time.perf_counter() - t0
        target = out / f"bc_{name}.npz"
        save_dataset(obs_list, y_list, target)
        per_round = matches * args.rounds
        print(
            f"[{name}] 样本 {len(y_list):6d}  局 {per_round:4d}  用时 {dt:6.1f}s "
            f"（{dt / max(per_round,1) * 1000:.0f} ms/局，{args.workers} workers）  -> {target}"
        )
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
