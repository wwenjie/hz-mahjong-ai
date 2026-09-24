"""按路线标定「向听 → 胜率」表（tasks.md 5.6 的修复前提）。

共用一张路线无关的表是 5.6 首版失败的原因：七对不能吃碰，同向听的实际成牌率低于
一般形。本工具让四家座位**同时承诺同一条路线**打若干局，因此观测到的成牌率对「该
路线」是无混淆的——这正是路线估值需要的条件概率。

用法::

    uv run python tools/calibrate.py --commitment pair --rounds 1200
    uv run python tools/calibrate.py --commitment meld --rounds 1200

输出两张表：按七对向听、按一般形向听，各有样本数与最终自摸率。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict

from majiang.rules import shanten as shanten_module
from majiang.rules.action import DISCARD
from majiang.runtime.decider import GuardedDecider
from majiang.sim.round import run_round
from majiang.strategy.policy import Commitment, HeuristicDecider, PolicyConfig

MIN_SAMPLES = 30


class Probe:
    """记录每次本人可出牌时的两条路线向听，最终由调用方补上「是否自摸」。"""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.rows: list[tuple[int | None, int, int]] = []

    def choose(self, situation, actions, *, budget_ms):
        discards = [action for action in actions if action.kind == DISCARD]
        if situation.drawn_tile is not None and discards:
            counts = situation.hand.counts
            meld_count = situation.hand.meld_count
            pair = (
                shanten_module.seven_pairs_shanten(counts) if meld_count == 0 else None
            )
            try:
                meld = shanten_module.shanten_any(counts, meld_count)
            except shanten_module.ShantenError:
                meld = shanten_module.SHANTEN_MAX
            self.rows.append((pair, meld, situation.table.draws_left))
        return self.inner.choose(situation, actions, budget_ms=budget_ms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按路线标定向听→胜率表")
    parser.add_argument("--commitment", choices=[c.value for c in Commitment], default="meld")
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--base-score", type=int, default=1)
    args = parser.parse_args(argv)

    commitment = Commitment(args.commitment)
    rng = random.Random(args.seed)
    pair_table: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    meld_table: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    fan_counts: dict[int, int] = defaultdict(int)
    detail_counts: Counter = Counter()
    wins = flows = 0

    for index in range(args.rounds):
        probes = [
            Probe(
                GuardedDecider(
                    HeuristicDecider(PolicyConfig(commitment=commitment))
                )
            )
            for _ in range(4)
        ]
        result = run_round(
            probes, dealer=index % 4, round_no=index + 1, base_score=args.base_score, rng=rng
        )
        flows += result.is_flow
        wins += not result.is_flow
        if not result.is_flow:
            fan_counts[result.fan] += 1
            detail_counts["+".join(result.detail) or "?"] += 1
        for seat, probe in enumerate(probes):
            won = (not result.is_flow) and result.winner == seat
            for pair_shanten, meld_shanten, _draws in probe.rows:
                if pair_shanten is not None:
                    pair_table[pair_shanten][0] += 1
                    pair_table[pair_shanten][1] += won
                meld_table[meld_shanten][0] += 1
                meld_table[meld_shanten][1] += won

    print(f"承诺 {commitment.value}：{args.rounds} 局，其中胡 {wins}、流局 {flows}")
    if fan_counts:
        total_wins = sum(fan_counts.values())
        mean_fan = sum(fan * n for fan, n in fan_counts.items()) / max(1, total_wins)
        print(f"\n胡牌番分布（均番 {mean_fan:.2f}）:")
        for fan in sorted(fan_counts):
            print(f"  番 {fan:>2d}: {fan_counts[fan]:>5d} 次 ({fan_counts[fan] / max(1, total_wins):.0%})")
    if detail_counts:
        print("\n番型组合 top:")
        for label, count in detail_counts.most_common(6):
            print(f"  {label:28s} {count}")
    for label, table in (("七对向听", pair_table), ("一般形向听", meld_table)):
        print(f"\n{label} → 最终自摸率")
        print(f"{'向听':>4s} {'样本':>8s} {'胜率':>7s}")
        for shanten in sorted(table):
            count, won = table[shanten]
            if count >= MIN_SAMPLES:
                print(f"{shanten:>4d} {count:>8d} {won / count:>6.1%}")
    print("\n可直接粘贴进 routes.py 的元组（按向听 0..6，缺档用邻值兜底）:")
    for label, table in (("PAIR", pair_table), ("MELD", meld_table)):
        values = []
        for shanten in range(7):
            count, won = table.get(shanten, [0, 0])
            values.append(round(won / count, 3) if count >= MIN_SAMPLES else None)
        filled = [v for v in values if v is not None]
        fallback = filled[-1] if filled else 0.05
        print(f"  {label} = {tuple(v if v is not None else fallback for v in values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
