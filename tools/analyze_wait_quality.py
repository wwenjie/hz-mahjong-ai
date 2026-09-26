"""听口质量诊断：我们与对手「听牌时还剩多少张能胡」（tasks.md 5.2 的盲区）。

**为什么做这个**：我们此前只看过「是否听牌」这一个二值切面。但两支听牌率相同的策略，
一支可能平均听 2 张、另一支听 8 张——**胡牌率会差出好几个百分点**。这个维度我们
从未度量过，而它完全可以从真实事件流里算出来。

口径：在 ``tile_discarded`` 事件**之前**取局面（14 张），扣掉那次打出的牌得到 13 张，
若其向听为 0（听牌）则用 ``winning_draws`` 求听口，再用 ``visible_counts`` 求每个听口
**还剩几张可见**（4 − 已见）。

注意与其它工具的口径差异：`measure_strength` 的听牌率在 ``tile_discarded`` 上、
`analyze_hand_progress` 在 ``tile_drawn`` 上，本工具在**出牌后**判定听口。
引用时必须写明时点。

开销：``winning_draws`` 只在「打完真的听牌」时才调用（约 1/4 的出牌），
故需限 ``--limit`` 控制规模。

用法::

    uv run python tools/analyze_wait_quality.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win as win_module
from majiang.sim import replay

SEATS = 4
DISCARDED = "tile_discarded"
GROUP_US = "我们"
GROUP_THEM = "对手"


def live_copies(state: replay.ReplayState, seat: int, waits: tuple[int, ...]) -> int:
    """这批听口在全场还剩几张可摸（4 − 已见：本方暗手 + 四家副露 + 四家弃牌）。"""
    # 注意 ``visible_counts`` 的形状要求：``melds`` 与 ``discards`` 都是
    # **按座位的二维序列**（每一家的副露/弃牌各是一个序列）。传成一维会在
    # ``for tile in group`` 上炸成 "int object is not iterable"（栽过一次）。
    visible = shanten_module.visible_counts(
        list(state.seats[seat].hand),
        [meld.tiles for item in state.seats for meld in item.melds],
        [list(item.discards) for item in state.seats],
    )
    total = 0
    for tile in waits:
        total += max(0, tiles.COPIES_PER_KIND - visible[tile])
    return total


def scan(
    payload: dict,
    ours: str,
    stats_by_group: dict[str, list[tuple[int, int]]],
    waits_hist: dict[str, Counter],
    counters: Counter,
) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or ours not in ids:
        return
    mine = ids.index(ours)

    for state, events in replay.iter_rounds(payload):
        for event in events:
            if str(event.get("type")) != DISCARDED:
                replay.apply_event(state, event)
                continue
            seat = event.get("seat")
            tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
            if not (isinstance(seat, int) and 0 <= seat < SEATS) or tile is None:
                replay.apply_event(state, event)
                continue
            seat_state = state.seats[seat]
            counts = list(seat_state.hand)
            if counts[tile] <= 0:
                replay.apply_event(state, event)
                continue
            # 分母必须在**判定听牌之前**累加，否则「听牌占比」恒为 100%（踩过）
            counters[f"{GROUP_US if seat == mine else GROUP_THEM} 出牌次数"] += 1
            counts[tile] -= 1  # 打完这一张之后的暗手
            meld_count = len(seat_state.melds)
            try:
                if shanten_module.shanten_any(counts, meld_count) != 0:
                    replay.apply_event(state, event)
                    continue
                # **注意 winning_draws 在 win.py，不在 shanten.py**（导错一次，
                # 而当时用裸 except 吞掉了 AttributeError，392 条全被算成「听口计算失败」）。
                waits = win_module.winning_draws(counts, meld_count)
            except Exception as exc:  # noqa: BLE001 —— 单个局面失败不应中断整批
                # 计数器带异常类型：裸 except 会把「我导错了模块」伪装成「算不出来」
                counters[f"听口计算失败:{type(exc).__name__}"] += 1
                replay.apply_event(state, event)
                continue
            if not waits:
                counters["听牌但算不出听口"] += 1
                replay.apply_event(state, event)
                continue
            group = GROUP_US if seat == mine else GROUP_THEM
            copies = live_copies(state, seat, waits)
            stats_by_group[group].append((len(waits), copies))
            waits_hist[group][min(len(waits), 8)] += 1
            counters[f"{group} 出牌后听牌"] += 1
            replay.apply_event(state, event)


def render(
    stats_by_group: dict[str, list[tuple[int, int]]],
    waits_hist: dict[str, Counter],
    counters: Counter,
) -> None:
    print(f"{'':<10}{'出牌数':>10}{'听牌占比':>10}{'听口种数(均)':>13}{'剩余张数(均)':>14}{'中位':>6}")
    for group in (GROUP_US, GROUP_THEM):
        rows = stats_by_group[group]
        discards = counters.get(f"{group} 出牌次数", 0)
        if not rows or not discards:
            print(f"{group:<10}（无样本）")
            continue
        kinds = stats.mean(item[0] for item in rows)
        copies = stats.mean(item[1] for item in rows)
        median = stats.median(item[1] for item in rows)
        print(
            f"{group:<10}{discards:>10}{len(rows) / discards:>10.1%}"
            f"{kinds:>13.2f}{copies:>14.2f}{median:>6.0f}"
        )
    print()
    print(f"{'听口种数':>8}{'我们占比':>10}{'对手占比':>10}")
    total_us = sum(waits_hist[GROUP_US].values())
    total_them = sum(waits_hist[GROUP_THEM].values())
    for kinds in range(1, 9):
        label = f"{kinds}" if kinds < 8 else "8+"
        us = waits_hist[GROUP_US][kinds] / total_us if total_us else 0.0
        them = waits_hist[GROUP_THEM][kinds] / total_them if total_them else 0.0
        print(f"{label:>8}{us:>9.1%}{them:>10.1%}")
    for key in sorted(counters):
        if key.startswith("听") or key.endswith("失败"):
            print(f"  {key}: {counters[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="听口质量诊断")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args(argv)

    lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
    paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("清单为空", file=sys.stderr)
        return 1

    stats_by_group: dict[str, list[tuple[int, int]]] = defaultdict(list)
    waits_hist: dict[str, Counter] = defaultdict(Counter)
    counters: Counter = Counter()
    used = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        try:
            scan(payload, args.ours, stats_by_group, waits_hist, counters)
        except Exception as exc:  # noqa: BLE001
            counters[f"跳过:{type(exc).__name__}"] += 1
            continue
        used += 1

    print(f"文件 {used} 个，视角 {args.ours}，口径：**出牌后**判定听牌")
    print()
    render(stats_by_group, waits_hist, counters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
