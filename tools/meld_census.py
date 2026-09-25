"""副露普查：我们与对手每局持有/做出的副露数（tasks.md 5.5 的前提校验）。

**为什么单独做一个工具**：改吃碰闸门的整个出发点，是「我们副露 0.59 次/局，对手 1.09 次」。
这个前提如果错了，方向就错了。此前它是**一次性内联脚本**算出来的，无法复核、也无法比对子集，
所以这里固化成工具，并同时输出两个口径：

- **持有**：局末 ``len(melds)``（一次副露被补杠升级后仍算 1）
- **事件**：``peng`` / ``chi`` / ``gang`` 事件的计数（补杠会额外+1，明杠/暗杠新增副露）

两个口径不一致时，说明副露与事件的换算有问题，本身就是信号。

用法::

    uv run python tools/meld_census.py --manifest notes/manifest-20260926.txt
    uv run python tools/meld_census.py --events 'data/auto_sessions/*/events/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

from majiang.sim import replay

SEATS = 4
MELD_EVENTS = ("peng", "chi", "gang")


def measure(paths: list[str], ours: str) -> dict[str, object]:
    held: dict[str, list[float]] = defaultdict(list)
    held_chi: dict[str, list[float]] = defaultdict(list)
    events: dict[str, int] = defaultdict(int)
    rounds = 0
    files = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
        if len(ids) != SEATS or ours not in ids:
            continue
        files += 1
        mine = ids.index(ours)
        for block in payload.get("blocks") or ():
            for event in block.get("events") or ():
                kind = str(event.get("type"))
                if kind in MELD_EVENTS:
                    seat = event.get("seat")
                    if isinstance(seat, int) and 0 <= seat < SEATS and ours in ids:
                        events["我们" if seat == mine else "对手"] += 1
        for state, round_events in replay.iter_rounds(payload):
            rounds += 1
            for event in round_events:
                replay.apply_event(state, event)
            for index, uid in enumerate(ids):
                group = "我们" if uid == ours else "对手"
                melds = state.seats[index].melds
                held[group].append(float(len(melds)))
                held_chi[group].append(float(sum(1 for m in melds if m.kind == "chi")))
    return {
        "files": files,
        "rounds": rounds,
        "held": held,
        "held_chi": held_chi,
        "events": events,
    }


def report(result: dict[str, object], label: str) -> None:
    held = result["held"]
    held_chi = result["held_chi"]
    events = result["events"]
    rounds = int(result["rounds"])
    print(f"=== {label} ===")
    print(f"文件 {result['files']} 个，局数 {rounds}")
    if not rounds:
        return
    for group in ("我们", "对手"):
        values = held[group]
        chis = held_chi[group]
        if not values:
            continue
        per_round_events = events[group] / rounds
        # 对手是三家合计，折成「每家每局」才与我们可比
        divisor = 1 if group == "我们" else 3.0
        print(
            f"  {group}: 持有副露 {sum(values) / len(values):.3f}/局"
            f"（其中吃 {sum(chis) / len(chis):.3f}）"
            f"  副露事件 {per_round_events / divisor:.3f}/家·局"
            f"  样本 {len(values)}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="副露普查")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--events", default="", help="改用 glob（与 --manifest 二选一）")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--head", type=int, default=0, help="只取清单前 N 个文件（子集校验用）")
    parser.add_argument("--tail", type=int, default=0, help="只取清单后 N 个文件")
    args = parser.parse_args(argv)

    if args.events:
        paths = sorted(glob.glob(args.events))
        label = args.events
    else:
        lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
        paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        label = f"{args.manifest}（{len(paths)} 个）"
        if args.head:
            paths = paths[: args.head]
            label = f"{args.manifest} 前 {args.head} 个"
        elif args.tail:
            paths = paths[-args.tail :]
            label = f"{args.manifest} 后 {args.tail} 个"
    if not paths:
        print("没有文件", file=sys.stderr)
        return 1

    report(measure(paths, args.ours), label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
