#!/usr/bin/env python3
"""T2 数据生成：价值模型的 **29 维基础特征 + 四家弃牌序列**。

动机（T1 结论）：29 维里 `discards_0..3` 只用了弃牌**张数**，弃牌的顺序与内容在特征层
就被丢掉了——任何模型（GBDT 或 NN）在这套表示下都学不到「出牌历史」里的信息。
本生成器把**有序弃牌序列**一并录下，供 T2 检验「序列表示是否有增益」。

口径与既有 `tools/gen_value_data.py` **逐项对齐**（便于与 T1 的结果直接比较）：
- 每条样本 =（决策时刻的公开局面，本局本人最终得分），只记**本人**决策、只记「打后」状态；
- 标签由模拟器给出，精确、无需标注；
- 特征与推理共用 `features.extract`，不另起一套（避免 train-serve skew）。

新增（本文件独有）：
- ``seq``：形状 ``(4, MAX_SEQ)``，按座位给出**时序**弃牌（牌索引 0..33），不足左侧补 ``PAD=34``。
  只含公开信息（场上弃牌），**不含任何对手暗牌**。

用法::

    nice -n 19 uv run python research/gen_value_seq_data.py --rounds 1600 --workers 4 \\
        --out data/value_seq.part000.npz

产物分片：``<stem>.w<worker>c<chunk>.npz``，**每 chunk 落盘一次**（默认 20 局）。
已存在的 chunk 直接跳过——长跑进程会被环境周期性回收（日志无 Traceback、dmesg 无 OOM），
所以「只在结尾写一次」等于白跑。被杀后**原样重跑**即可接上。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from majiang.rules.action import DISCARD
from majiang.sim.round import run_round
from majiang.strategy import features
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

MAX_SEQ = 24
PAD = features.tiles.TILE_KINDS  # 34
SEATS = 4


def _encode_sequences(discards) -> np.ndarray:
    """把四家弃牌序列编成 ``(4, MAX_SEQ)`` 的定长整数矩阵（左补 PAD）。"""
    out = np.full((SEATS, MAX_SEQ), PAD, dtype=np.int8)
    for seat in range(SEATS):
        group = list(discards[seat]) if seat < len(discards) else []
        tail = group[-MAX_SEQ:]
        if tail:
            out[seat, MAX_SEQ - len(tail):] = np.asarray(tail, dtype=np.int8)
    return out


class Recorder:
    """包住策略，记录「打后」局面的 29 维特征 + 四家弃牌序列。"""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.x: list[list[float]] = []
        self.seq: list[np.ndarray] = []

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
            self.x.append(features.extract(after))
            self.seq.append(_encode_sequences(situation.discards))
        return choice


def _write_chunk(path: Path, xs, seqs, ys) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 注意：np.savez_compressed 会在路径不以 .npz 结尾时自动补 .npz，
    # 所以临时名必须也以 .npz 结尾，否则 os.replace 找不到文件。
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        x=np.asarray(xs, dtype=np.float32),
        seq=np.asarray(seqs, dtype=np.int8),
        y=np.asarray(ys, dtype=np.float32),
    )
    os.replace(tmp, path)  # 原子替换：不会留下半个文件


def _run_shard(job: tuple[int, str, int, int, str]) -> list[str]:
    """按 chunk 跑并**逐块落盘**；已存在的 chunk 跳过（可续跑）。"""
    seed, out_pattern, rounds, chunk, mode = job
    config = PolicyConfig.for_mode(Mode(mode))
    stem = out_pattern[:-4] if out_pattern.endswith(".npz") else out_pattern
    written: list[str] = []
    for start in range(0, rounds, chunk):
        count = min(chunk, rounds - start)
        part = Path(f"{stem}.c{start:05d}.npz")
        if part.exists():
            written.append(str(part))
            continue
        # 每个 chunk 用独立种子，使「续跑」与「一次跑完」逐样本一致
        rng = random.Random(seed + start * 7919)
        xs: list[list[float]] = []
        seqs: list[np.ndarray] = []
        ys: list[float] = []
        for index in range(count):
            recorders = [Recorder(HeuristicDecider(config)) for _ in range(SEATS)]
            result = run_round(recorders, dealer=index % SEATS, round_no=start + index + 1, rng=rng)
            for seat, recorder in enumerate(recorders):
                target = float(result.scores[seat])
                xs.extend(recorder.x)
                seqs.extend(recorder.seq)
                ys.extend([target] * len(recorder.x))
        _write_chunk(part, xs, seqs, ys)
        written.append(str(part))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T2 序列特征数据生成")
    parser.add_argument("--rounds", type=int, default=1600, help="总局数（各 worker 的合计）")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--mode", choices=[mode.value for mode in Mode], default=Mode.QUALIFIER.value)
    parser.add_argument("--out", default="data/value_seq.npz")
    parser.add_argument("--chunk", type=int, default=20, help="每个 chunk 的局数（落盘粒度）")
    args = parser.parse_args()

    total = args.rounds
    workers = max(1, args.workers)
    base = total // workers
    extra = total % workers
    jobs = []
    cursor = args.seed
    for index in range(workers):
        count = base + (1 if index < extra else 0)
        if count == 0:
            continue
        cursor += 1013
        stem = args.out[:-4] if args.out.endswith(".npz") else args.out
        pattern = f"{stem}.w{index:03d}.npz"
        jobs.append((cursor, pattern, count, args.chunk, args.mode))

    print(f"生成 {total} 局，{len(jobs)} 个 worker，chunk={args.chunk}局（nice 由调用方保证）",
          file=sys.stderr)
    started = time.perf_counter()
    with Pool(processes=len(jobs)) as pool:
        paths = pool.map(_run_shard, jobs)
    flat = [p for group in paths for p in group]
    total_samples = 0
    for path in flat:
        with np.load(path) as doc:
            total_samples += doc["x"].shape[0]
    elapsed = time.perf_counter() - started
    print(f"完成 {total} 局 / {len(flat)} 分片 / {total_samples} 样本，用时 {elapsed:.0f}s")
    print(f"  glob: {args.out[:-4] if args.out.endswith('.npz') else args.out}.w*.c*.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
