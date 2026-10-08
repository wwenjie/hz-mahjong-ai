#!/usr/bin/env python
"""A/B 对拍·评测方案2：同牌四位轮换协议（duplicate-style）。

与 run_ab_v7.py（方案1）的差异：
- 方案1：模型固定坐 seat 0，仅初始庄家轮换 → 640 局不同牌摊牌运。
- 方案2（本脚本）：每个 match 用**相同 seed**（牌墙序列只由 seed 决定，与座位无关），
  模型分别坐 0/1/2/3 号位各打一遍 → 同一副牌模型从四个位置各经历一次。

决策器直接 import 自 run_ab_v7（与方案1 逐字一致，杜绝拷贝走样）。

协议：matches 场 × rounds 局 × 4 座位轮换，单 seed。
样本量：matches × 4 × rounds 局（默认 40×4×8 = 1280 局/模型）。

用法::

    PYTHONPATH=src python scripts/run_ab_v7_dup.py \
        --model runs/bc_v7_base.pt --seed 20261003 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # 让 from run_ab_v7 import 可用

import torch  # noqa: E402

from run_ab_v7 import BCV7Decider  # noqa: E402  ← 复用方案1 决策器，保证一致


def run_ab_v7_dup(*, model_path, matches: int, rounds: int, seed: int, device: str = "cuda"):
    """同牌四位轮换：模型轮坐 4 个位置，每位置跑 matches 场（同 seed ⇒ 同牌）。"""
    from majiang.sim.batch import run_batch
    from majiang.strategy.versions import build
    from majiang.strategy.policy import Mode

    def make_bc():
        return BCV7Decider(model_path, device)

    def make_v5():
        return build("v5", Mode.QUALIFIER)

    # 4 轮：模型分别坐 0/1/2/3 号位；同 seed → 每轮牌墙完全相同
    per_pos_stats = []
    for pos in range(4):
        factories = [make_v5, make_v5, make_v5, make_v5]
        factories[pos] = make_bc
        labels = ["v5", "v5", "v5", "v5"]
        labels[pos] = "bcv7"
        result = run_batch(
            factories, matches=matches, rounds=rounds, base_score=1, seed=seed,
            labels=labels, dealer_rotation=True,
        )
        per_pos_stats.append(result.seats[pos])  # 只取模型座位
        s = result.seats[pos]
        print(f"  pos={pos}: 总得分 {s.total_score:>6d}  名次分 {s.place_points:>5d}  "
              f"胡率 {s.win_rate:>5.1%}  均分 {s.average_score:>7.1f}", flush=True)

    # 汇总 4 个位置
    total_score = sum(s.total_score for s in per_pos_stats)
    place_points = sum(s.place_points for s in per_pos_stats)
    god_count = sum(s.god_count for s in per_pos_stats)
    wins = sum(s.wins for s in per_pos_stats)
    total_rounds = sum(s.rounds for s in per_pos_stats)
    return {
        "total_score": total_score,
        "place_points": place_points,
        "god_count": god_count,
        "wins": wins,
        "rounds": total_rounds,
        "win_rate": wins / total_rounds if total_rounds else 0.0,
        "avg_score": total_score / total_rounds if total_rounds else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测方案2：同牌四位轮换 vs v5 教师")
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    print(f"模型: {args.model}")
    print(f"方案2 同牌四位轮换: {args.matches} 场 × {args.rounds} 局 × 4 位置 vs v5，seed={args.seed}")
    print()

    t0 = time.perf_counter()
    result = run_ab_v7_dup(
        model_path=args.model, matches=args.matches, rounds=args.rounds,
        seed=args.seed, device=args.device,
    )
    dt = time.perf_counter() - t0

    print(f"\n用时: {dt:.1f}s")
    print(f"汇总（4 位置合计 {result['rounds']} 局）:")
    print(f"  总得分 {result['total_score']:>6d}  名次分 {result['place_points']:>5d}  "
          f"胡率 {result['win_rate']:>5.1%}  均分 {result['avg_score']:>7.1f}")

    out_dir = Path("records")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ab2_dup_{Path(args.model).stem}_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "protocol": "dup4", "matches": args.matches,
                   "rounds": args.rounds, "seed": args.seed, "elapsed_s": dt,
                   **result}, f, indent=2)
    print(f"记录已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
