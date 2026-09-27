"""财神使用诊断：我们与对手怎么用白板（tasks.md 2.5 / 2.7 的盲区）。

**为什么做**：白板（财神）在本平台权重异常高——
① **4 张白板 ×2 番**（番数连乘里的一环，最高可到 512）；
② **白板获取数是比赛的第三个排序键**（总得分 -> 名次分 -> 白板数）；
③ 财神是百搭，也是唯一能触发「抓打圈」的牌。
但我们对它的行为**一次也没量过**。

**方法（刻意避开因果截断）**：不在「结束时刻」统计持有量——那会被「赢家提前结束本局」
污染（赢得早 = 摸到的财神少）。改为在**每个座位第 N 次摸牌**这个**固定早期时点**上
记录持有数，再看**最终是否胡牌**。这样比较的是同一时点的同一状态。

输出四格：我们 / 对手 × (0 张 / 1 张 / ≥2 张) 的最终胡牌率，以及摸到财神的总量、
打出财神的总量（后者是「抓打圈」的来源，也直接损失灵活性）。

用法::

    uv run python tools/analyze_god_usage.py --events 'data/auto_sessions/*/events/*.json' --limit 400
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
from majiang.sim import replay

SEATS = 4
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"
US = "我们"
THEM = "对手"
MILESTONE = 4  # 在第几次摸牌上取状态


def bucket(gods: int) -> str:
    return "0张" if gods == 0 else ("1张" if gods == 1 else "≥2张")


def scan(payload: dict, ours: str, stats: dict, counters: Counter) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or ours not in ids:
        return
    mine = ids.index(ours)
    official = {
        int(entry.get("round_no", 0) or 0): entry for entry in (payload.get("rounds") or [])
    }

    for state, events in replay.iter_rounds(payload):
        milestones: dict[int, tuple[int, int]] = {}  # 座位 -> (财神数, 精确向听)
        draws: Counter = Counter()
        for event in events:
            kind = str(event.get("type"))
            if kind == DRAWN:
                seat = event.get("seat")
                if isinstance(seat, int) and 0 <= seat < SEATS:
                    draws[seat] += 1
                    if draws[seat] == MILESTONE:
                        seat_state = state.seats[seat]
                        # **同一时点同时记下向听**：这是把「财神用不好」与
                        # 「手牌本来就落后」分开的关键。若同财神数下我们的向听不大于对手、
                        # 却胡得更少，问题在转化；若向听也落后，问题在出牌本身。
                        try:
                            value = shanten_module.shanten_any(
                                seat_state.hand, len(seat_state.melds)
                            )
                        except Exception:  # noqa: BLE001
                            value = -1
                        milestones[seat] = (seat_state.hand[tiles.GOD], value)
            elif kind == DISCARDED:
                seat = event.get("seat")
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if tile == tiles.GOD and isinstance(seat, int) and 0 <= seat < SEATS:
                    counters[f"{US if seat == mine else THEM} 打出财神"] += 1
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
            key = (group, bucket(gods))
            stats[key][0] += 1
            if shanten_value >= 0:
                stats[key][2] += shanten_value
            if isinstance(winner, int) and winner == seat:
                stats[key][1] += 1


def report(stats: dict, counters: Counter) -> None:
    print(f"{'':<8}{'第4摸时财神':<12}{'样本':>8}{'均向听':>8}{'最终胡牌率':>12}{'占比':>9}")
    for group in (US, THEM):
        total = sum(stats[(group, b)][0] for b in ("0张", "1张", "≥2张"))
        for label in ("0张", "1张", "≥2张"):
            hands, wins, shanten_sum = stats[(group, label)]
            if not hands:
                continue
            print(f"{group:<8}{label:<12}{hands:>8}{shanten_sum / hands:>8.2f}"
                  f"{wins / hands:>12.2%}{hands / total:>9.1%}")
        print()
    print("原始计数：")
    for key in sorted(counters):
        print(f"  {key}: {counters[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="财神使用诊断")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args(argv)

    if args.manifest:
        lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
        paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    else:
        paths = sorted(glob.glob(args.events))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有文件", file=sys.stderr)
        return 1

    stats: dict = defaultdict(lambda: [0, 0, 0])
    counters: Counter = Counter()
    used = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        try:
            scan(payload, args.ours, stats, counters)
        except Exception as exc:  # noqa: BLE001 —— 带异常类型，避免被裸 except 吞掉
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        used += 1

    print(f"文件 {used} 个，状态取自每个座位的**第 {MILESTONE} 次摸牌**（避开「赢家提前结束」的截断）")
    print()
    report(stats, counters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
