"""自对弈**胡牌结构**普查：爆头率 / 胡率 / 白板 / 均番（解耦指标，用于筛财神路线臂）。

**为什么要它**：`tools/baotou_census.py` 在真机上量到**我方爆头占自己胡牌 14.4% vs 三家 23.8%**
（1.66×）、胡率 20.9% vs 25.7%（0.81×）——这是当日测到的最大结构性缺口。
而 `preserve_god` / `natural_route` 这两个旋钮在台账里**从未被 A/B 过**（四个 job 全 `skipped`）。
**先用解耦指标筛（本工具），再用名次分做 A/B**——这正是本项目的既定纪律：
爆头率/胡率是**机制指标**，不能当进度指标用，但**非常适合筛臂**。

同种子 ⇒ 同一副牌 ⇒ 臂间**逐位配对**（不是两个独立批次）。

用法::
    .venv/bin/python tools/winmix_census.py --arms v7,v7-preserve,v7-natural --matches 24
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.sim.batch import run_match  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

INTEREST = ("爆头", "财飘", "七对", "杠", "碰碰", "清一色", "豪华")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="自对弈胡牌结构普查（爆头率等）")
    ap.add_argument("--arms", default="v7,v7-preserve,v7-natural")
    ap.add_argument("--matches", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args(argv)

    print(f"自对弈（每座同一档）{args.matches} 场 × {args.rounds} 局，种子 {args.seed}（臂间同种子配对）")
    print(f"{'档位':<16}{'胡率':>8}{'爆头占胡':>10}{'爆头/局':>9}{'白板/局':>9}"
          f"{'均番':>8}{'名次分':>9}")
    rows = {}
    for arm in [name.strip() for name in args.arms.split(",") if name.strip()]:
        wins = 0
        rounds = 0
        baotou = 0
        gods = 0
        fans = 0
        place = 0
        details: collections.Counter = collections.Counter()
        for index in range(args.matches):
            result = run_match(
                [make_decider(arm, Mode.QUALIFIER) for _ in range(4)],
                rounds=args.rounds,
                base_score=1,
                seed=args.seed * 100003 + index,
                labels=[arm] * 4,
            )
            for stat in result.seats:
                wins += stat.wins
                gods += stat.god_count
                fans += stat.fan_total
                place += stat.game_place_points
                rounds += stat.rounds
            details.update(result.details)
        rounds = max(1, rounds)
        # **`details` 的键是复合串**（实测 `'平胡+爆头'` / `'七对+爆头'` / `'平胡+杠开'`），
        # 所以必须**子串求和**，不能 `details["爆头"]`（那样恒得 0，2026-10-10 14:00 踩过）。
        baotou = sum(v for k, v in details.items() if "爆头" in k)
        wins = max(1, wins)
        rows[arm] = (wins, rounds, baotou, gods, fans, place, details)
        print(f"{arm:<16}{wins / rounds:>8.1%}{baotou / wins:>10.1%}"
              f"{baotou / rounds:>9.3f}{gods / rounds / 4:>9.3f}"
              f"{fans / wins:>8.3f}{place / args.matches:>9.3f}")
    print("\n  各档 `details` 计数（引擎写出的番种明细）：")
    for arm, (_, _, _, _, _, _, details) in rows.items():
        picked = {k: v for k, v in details.items() if any(w in k for w in INTEREST)}
        print(f"    {arm:<16}{dict(sorted(picked.items(), key=lambda kv: -kv[1]))}")
    print("\n  真机参照（三家每家）：胡率 25.7%/局、爆头占胡 23.8%、均番 1.325")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
