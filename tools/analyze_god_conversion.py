"""财神转化率诊断：**同样持有财神，为什么我们赢得更少**。

**为什么做这个**：`tools/analyze_god_usage.py` 已经给出一个刺眼的形状——

| 第4摸时财神 | 我们 均向听 / 胡率 | 对手 均向听 / 胡率 |
|---|---|---|
| 0 张 | 2.13 / 13.17% | 2.03 / 15.45% |
| 1 张 | 1.59 / 28.90% | 1.45 / 36.98% |
| ≥2 张 | 1.08 / 50.24% | 0.94 / 60.51% |

三点读法：① 财神**获取**没问题（分布几乎逐位相同：60.4/32.8/6.7 vs 60.5/32.6/6.9）；
② 财神**保留**也没问题（对手打出财神 339/9600=3.5%，我们 44/3200=1.4%——我们更舍不得打）；
③ 但我们**转换**明显更差，而且**缺口随财神数放大**（−2.3pp → −8.1pp → −10.3pp）。

所以问题不是「有没有财神」，而是「拿着财神干了什么」。本工具做**两步分离**：

1. **控制向听**：(财神数 × 向听) 的交叉表。若同财神数下我们在各向听层里胡率都不低，
   那缺口只是「同一时点我们平均落后 0.1–0.15 向听」，属于前中期出牌问题；
   若向听也相等却仍差，问题在**临近听牌/听牌后的转化**。
2. **控制听口宽度**：我们与对手**听牌时**的可见张数分布，按财神数分层
   （复用 `analyze_wait_quality.py` 的 `winning_draws` + 可见张数口径）。
   若窄听口集中在「持有财神」的手里，那答案就很具体：**财神在手时我们选的听口更窄**，
   而财神的真正价值正是把听口变宽。

口径：两个时点都要写明。向听取**每个座位第 4 次摸牌**之后（避开「赢家提前结束」截断）；
听口取**出牌之后**（14 张扣掉打出的那张）。

用法::

    uv run python tools/analyze_god_conversion.py --limit 400
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win as win_module
from majiang.sim import replay

SEATS = 4
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"
US = "我们"
THEM = "对手"
MILESTONE = 4
GOD_BUCKETS = ("0张", "1张", "≥2张")
# 向听分层：0 是听牌，1/2 是 B 的交叉点所在，≥3 合成一格（样本少）
SHANTEN_BUCKETS = (0, 1, 2, 3)


def god_bucket(gods: int) -> str:
    return "0张" if gods == 0 else ("1张" if gods == 1 else "≥2张")


def shanten_bucket(value: int) -> int:
    return min(max(value, 0), 3)


def live_copies(state: replay.ReplayState, seat: int, waits: tuple[int, ...]) -> int:
    """听口里全场还剩几张可摸（4 − 已见：本方暗手 + 四家副露 + 四家弃牌）。

    ``melds``/``discards`` 必须是**按座位的二维序列**：``visible_counts`` 内部
    对每个元素再迭代一次，传扁平的一维会在 ``for tile in group`` 上炸
    "int object is not iterable"（同一坑踩过两次，第二次在这里）。
    """
    visible = shanten_module.visible_counts(
        list(state.seats[seat].hand),
        [meld.tiles for other in state.seats for meld in other.melds],
        [list(other.discards) for other in state.seats],
    )
    return sum(max(0, tiles.COPIES_PER_KIND - visible[tile]) for tile in waits)


def scan(payload: dict, ours: str, cells: dict, waits: dict, counters: Counter) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or ours not in ids:
        return
    mine = ids.index(ours)
    official = {
        int(entry.get("round_no", 0) or 0): entry for entry in (payload.get("rounds") or [])
    }

    for state, events in replay.iter_rounds(payload):
        milestones: dict[int, tuple[int, int]] = {}
        draws: Counter = Counter()
        for event in events:
            kind = str(event.get("type"))
            seat = event.get("seat")
            if kind == DRAWN and isinstance(seat, int) and 0 <= seat < SEATS:
                draws[seat] += 1
                if draws[seat] == MILESTONE:
                    seat_state = state.seats[seat]
                    try:
                        value = shanten_module.shanten_any(
                            seat_state.hand, len(seat_state.melds)
                        )
                    except Exception:  # noqa: BLE001
                        value = -1
                    milestones[seat] = (seat_state.hand[tiles.GOD], value)
            elif kind == DISCARDED and isinstance(seat, int) and 0 <= seat < SEATS:
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if tile is not None:
                    counts = list(state.seats[seat].hand)
                    counts[tile] -= 1
                    meld_count = len(state.seats[seat].melds)
                    try:
                        value = shanten_module.shanten_any(counts, meld_count)
                    except Exception:  # noqa: BLE001
                        value = -1
                    # **只在真的听牌时算听口**：`winning_draws` 约 0.1 s/次，
                    # 而听牌只占出牌的一部分，故这一步是可控的。
                    if value == 0:
                        waits_list = win_module.winning_draws(counts, meld_count)
                        if waits_list:
                            group = US if seat == mine else THEM
                            key = (group, god_bucket(counts[tiles.GOD]))
                            entry = waits[key]
                            entry[0] += 1
                            entry[1] += len(waits_list)
                            entry[2] += live_copies(state, seat, waits_list)
            replay.apply_event(state, event)

        entry = official.get(state.round_no) or {}
        winner = None if entry.get("is_draw") else entry.get("winner")
        for seat in range(SEATS):
            group = US if seat == mine else THEM
            counters[f"{group} 参与局数"] += 1
            if seat not in milestones:
                counters[f"{group} 未到第{MILESTONE}摸"] += 1
                continue
            gods, shanten_value = milestones[seat]
            if shanten_value < 0:
                counters[f"{group} 向听不可算"] += 1
                continue
            cell = cells[(group, god_bucket(gods), shanten_bucket(shanten_value))]
            cell[0] += 1
            if isinstance(winner, int) and winner == seat:
                cell[1] += 1


def report_cells(cells: dict) -> None:
    print("== 交叉表：第 4 摸时（财神数 × 向听）→ 本局最终胡牌率 ==")
    print(f"{'向听':>4} {'财神':>5} | {'我们 n':>7} {'我们胡率':>8} | "
          f"{'对手 n':>7} {'对手胡率':>8} | {'差':>7}")
    for shanten in SHANTEN_BUCKETS:
        for gods in GOD_BUCKETS:
            us_n, us_w = cells[(US, gods, shanten)]
            th_n, th_w = cells[(THEM, gods, shanten)]
            if us_n < 40 or th_n < 120:  # 格子样本下限：少于此不报数，避免把噪声当结论
                continue
            us_rate, th_rate = us_w / us_n, th_w / th_n
            mark = f"{th_rate - us_rate:+7.1%}"
            print(f"{shanten:>4} {gods:>5} | {us_n:>7} {us_rate:>8.2%} | "
                  f"{th_n:>7} {th_rate:>8.2%} | {mark}")
        print()

    print("== 按统一权重合成（对手的向听分布做基准，去掉「我们向听分布不同」的影响）==")
    for gods in GOD_BUCKETS:
        num = den = 0.0
        for shanten in SHANTEN_BUCKETS:
            us_n, us_w = cells[(US, gods, shanten)]
            th_n, th_w = cells[(THEM, gods, shanten)]
            if us_n < 40 or th_n < 120:
                continue
            weight = th_n
            num += weight * ((us_w / us_n) - (th_w / th_n))
            den += weight
        if den:
            print(f"  {gods:>5}: 标准化胡率差 {num / den:+6.2%}（按对手向听分布加权）")


def report_waits(waits: dict) -> None:
    print("\n== 听牌时的听口（出牌后判定，含可见张数）==")
    print(f"{'':<6}{'财神':>5}{'样本':>8}{'听口种数':>10}{'可见张数':>10}")
    for group in (US, THEM):
        for gods in GOD_BUCKETS:
            n, kinds, copies = waits[(group, gods)]
            if not n:
                continue
            print(f"{group:<6}{gods:>5}{n:>8}{kinds / n:>10.2f}{copies / n:>10.2f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="财神转化率诊断")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--events", default="")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args(argv)

    if args.events:
        paths = sorted(glob.glob(args.events))
    else:
        paths = [
            line.strip()
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有文件", file=sys.stderr)
        return 1

    cells: dict = defaultdict(lambda: [0, 0])
    waits: dict = defaultdict(lambda: [0, 0, 0])
    counters: Counter = Counter()
    for index, path in enumerate(paths):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 —— 带类型，不裸吞
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        scan(payload, args.ours, cells, waits, counters)
        if (index + 1) % 100 == 0:
            print(f"  ... {index + 1}/{len(paths)}", file=sys.stderr, flush=True)

    print(f"文件 {len(paths)} 个（向听取每座第 {MILESTONE} 次摸牌之后）。"
          f"原始计数：{' '.join(f'{k}={v}' for k, v in sorted(counters.items()))}\n")
    report_cells(cells)
    report_waits(waits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
