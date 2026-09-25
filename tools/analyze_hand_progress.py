"""手牌进度诊断：我们与对手在**每次摸牌前**处于几向听（tasks.md 5.2 / 5.4）。

**动机**：战力测量给出我们听牌率 21.3%、对手 31.5%，胜率 20.6% vs 26.3%。这是一个
「离胡牌还有多远」的差距，但「听牌率」只是一个二值切面——它无法区分两种完全不同的病因：

- **慢**：我们的向听分布整体右移（别人 1 向听时我们还在 3 向听）→ 问题在成牌速度
- **断**：分布形状相近，但我们更常把已成型的牌打散 → 问题在出牌选择

本工具直接输出**向听分布**与**按摸牌次序的进度曲线**，把这两种病因分开。

口径：``tile_drawn`` 事件发生**之前**的局面（即上一张打完之后、还没摸牌的 13 张手牌），
这与 ``tools/measure_strength.py`` 的听牌率口径一致（0 向听 ⟺ 听牌）。

两遍计算：
1. **精算**：``shanten_any``（精确），可限 ``--exact-limit`` 个文件
2. **快算**：``quick_shanten``（微秒级近似），跑全量

两遍的差异本身是信号：若 ``quick_shanten`` 在分布上明显偏乐观，那么任何以它为判据的
策略改动（例如 ``tiebreak="ukeire"``）都是在**噪声上做决策**——这能解释 5.4 附注里
「同向听改按进张排序反而变差」的结果。

用法::

    uv run python tools/analyze_hand_progress.py --manifest notes/manifest-20260926.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.rules.hand import Hand
from majiang.sim import replay

SEATS = 4
DRAWN = "tile_drawn"
BUCKETS = (0, 1, 2, 3, 4)
GROUP_US = "我们"
GROUP_THEM = "对手"


def bucket(shanten: int) -> int:
    return min(shanten, max(BUCKETS))


def scan(
    payload: dict,
    ours: str,
    exact: bool,
    dist: dict[str, Counter],
    by_index: dict[str, dict[int, Counter]],
    stats: Counter,
) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or ours not in ids:
        return
    mine = ids.index(ours)

    for state, events in replay.iter_rounds(payload):
        seen_draws: Counter = Counter()
        for event in events:
            if str(event.get("type")) == DRAWN:
                seat = event.get("seat")
                if isinstance(seat, int) and 0 <= seat < SEATS:
                    seat_state = state.seats[seat]
                    hand = Hand.from_counts(seat_state.hand, seat_state.melds)
                    if exact:
                        try:
                            value = shanten_module.shanten_any(hand.counts, hand.meld_count)
                        except Exception:  # noqa: BLE001
                            stats["向听计算失败"] += 1
                            replay.apply_event(state, event)
                            continue
                    else:
                        value = shanten_module.quick_shanten(hand.counts, hand.meld_count)
                    group = GROUP_US if seat == mine else GROUP_THEM
                    dist[group][bucket(value)] += 1
                    seen_draws[seat] += 1
                    if seen_draws[seat] <= 14:
                        by_index[group].setdefault(seen_draws[seat], Counter())[bucket(value)] += 1
                    stats[f"{group} 摸牌次数"] += 1
            replay.apply_event(state, event)


def render(
    dist: dict[str, Counter], by_index: dict[str, dict[int, Counter]], label: str
) -> None:
    print(f"=== {label} ===")
    total = {group: sum(counter.values()) for group, counter in dist.items()}
    if not total.get(GROUP_US):
        print("  没有样本")
        return
    print(f"{'向听':>6} {'我们':>10} {'对手':>10}")
    for value in BUCKETS:
        key = value if value < max(BUCKETS) else f"{max(BUCKETS)}+"
        ours = dist[GROUP_US][value] / total[GROUP_US]
        theirs = dist[GROUP_THEM][value] / total[GROUP_THEM]
        print(f"{str(key):>6} {ours:>9.1%} {theirs:>9.1%}")
    print()
    print("按「该座位第 n 次摸牌」的听牌率（0 向听占比）")
    print(f"{'n':>4} {'我们':>9} {'对手':>9} {'样本(我们)':>10}")
    for index in sorted(by_index[GROUP_US]):
        ours_counter = by_index[GROUP_US][index]
        theirs_counter = by_index[GROUP_THEM].get(index, Counter())
        ours_total = sum(ours_counter.values())
        theirs_total = sum(theirs_counter.values())
        if ours_total < 50 or theirs_total < 50:
            continue
        print(
            f"{index:>4} {ours_counter[0] / ours_total:>8.1%} "
            f"{theirs_counter[0] / theirs_total:>8.1%} {ours_total:>10}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="手牌进度诊断")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 个文件")
    parser.add_argument(
        "--exact-limit",
        type=int,
        default=150,
        help="精确向听只跑前 N 个文件（0 = 只跑快算）",
    )
    args = parser.parse_args(argv)

    lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
    paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("清单为空", file=sys.stderr)
        return 1

    def run(label: str, subset: list[str], exact: bool) -> None:
        dist: dict[str, Counter] = {GROUP_US: Counter(), GROUP_THEM: Counter()}
        by_index: dict[str, dict[int, Counter]] = {GROUP_US: {}, GROUP_THEM: {}}
        stats: Counter = Counter()
        for path in subset:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            try:
                scan(payload, args.ours, exact, dist, by_index, stats)
            except Exception as exc:  # noqa: BLE001
                stats[f"跳过:{type(exc).__name__}"] += 1
        render(dist, by_index, f"{label}（{len(subset)} 个文件）")
        print(f"  我们摸牌 {stats.get(f'{GROUP_US} 摸牌次数', 0)} 次，"
              f"对手摸牌 {stats.get(f'{GROUP_THEM} 摸牌次数', 0)} 次")
        for key in sorted(stats):
            if key.startswith("跳过") or key == "向听计算失败":
                print(f"  {key}: {stats[key]}")
        print()

    if args.exact_limit:
        run("精算 shanten_any", paths[: args.exact_limit], True)
    run("快算 quick_shanten", paths, False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
