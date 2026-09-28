"""行为指纹：给任意 uid 建一份「怎么打」的画像（只读事件流）。

**为什么做**：天梯显示前 25 名对手胡率 25–32%、每手 +1.0~+3.7 分，而我们冠军档约 20% / −1.0。
**+10pp 胡率**的缺口不是靠调开关能补的，必须先看清**他们怎么打**。
所以这里对指定 uid（默认取天梯前几名 + 我们）算同一批行为量，逐列对照：

- 副露率/局（吃+碰+杠）、打财神率/局；
- 到听率（按出牌序号分层，同机会比较）；
- 听牌时的听口可见张数；
- 平均起手到听所需摸牌数（首次到听的摸序）。

口径与限制：
- 只读**我们参与过的房**（对手也因此只在「对我们」的场里被采样）——这控制住了房间条件。
- 抓打圈强制的出牌不算自由决策点（与其它工具一致）。
- 事件流里**没有番**（番在 `rounds[].data.detail`），所以这里不报均番；要番得另走台账。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win as win_module
from majiang.sim import replay

SEATS = 4
OUR = "u_a7f7c67bb14a"


def rooms_for(uids: set[str]) -> list[Path]:
    """只保留**含这些 uid 中任意一个**的房（并按房取全文件）。"""
    by_room: dict[str, list[Path]] = {}
    for path in glob.glob("data/auto_sessions/*/events/*.json"):
        by_room.setdefault(str(Path(path).parts[-3]), []).append(Path(path))
    picked: list[Path] = []
    for room, files in by_room.items():
        try:
            doc = json.loads(files[0].read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = {str(s.get("user_id", "")) for s in (doc.get("seats") or [])}
        if ids & uids:
            picked.extend(sorted(files))
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对手行为指纹")
    parser.add_argument("--focus", nargs="+", required=True, help="要画像的 uid（可多个）")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args(argv)

    focus = set(args.focus) | {OUR}
    paths = rooms_for(focus)
    if args.max_files:
        paths = paths[: args.max_files]
    print(f"含目标 uid 的房文件 {len(paths)} 个\n")

    stats: dict[str, dict] = collections.defaultdict(
        lambda: {"rounds": 0, "melds": 0, "god_discards": 0, "discards": 0,
                 "widths": [], "draw_to_tenpai": [], "ordinal_tenpai": collections.Counter(),
                 "ordinal_all": collections.Counter()}
    )
    for index, path in enumerate(paths):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {path.name}: {type(exc).__name__}", file=sys.stderr)
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != SEATS:
            continue
        for state, events in replay.iter_rounds(doc):
            draws = [0] * SEATS
            ordinal = [0] * SEATS
            arrived: set[int] = set()
            for e in events:
                kind = str(e.get("type"))
                seat = e.get("seat")
                if kind == "tile_drawn" and isinstance(seat, int) and 0 <= seat < SEATS:
                    draws[seat] += 1
                elif kind in ("chi", "peng", "gang") and isinstance(seat, int) and 0 <= seat < SEATS:
                    stats[ids[seat]]["melds"] += 1
                elif kind == "tile_discarded" and isinstance(seat, int) and 0 <= seat < SEATS:
                    tile = replay._tile_of(e.get("tile"))
                    free = state.opened and not (e.get("data") or {}).get("catch_play")
                    if tile is not None and free:
                        who = stats[ids[seat]]
                        ordinal[seat] += 1
                        who["discards"] += 1
                        if tile == tiles.GOD:
                            who["god_discards"] += 1
                        counts = list(state.seats[seat].hand)
                        if counts[tile] > 0:
                            counts[tile] -= 1
                        melds = len(state.seats[seat].melds)
                        try:
                            value = shanten_module.shanten_any(counts, melds)
                        except Exception:  # noqa: BLE001
                            value = -1
                        if value >= 0:
                            bucket = min(ordinal[seat], 12)
                            who["ordinal_all"][bucket] += 1
                            if value == 0:
                                who["ordinal_tenpai"][bucket] += 1
                                if seat not in arrived:
                                    arrived.add(seat)
                                    who["draw_to_tenpai"].append(draws[seat])
                                    waits = win_module.winning_draws(counts, melds)
                                    if waits:
                                        vis = shanten_module.visible_counts(
                                            counts,
                                            [m.tiles for s in state.seats for m in s.melds],
                                            [list(s.discards) for s in state.seats],
                                        )
                                        who["widths"].append(
                                            sum(max(0, tiles.COPIES_PER_KIND - vis[t]) for t in waits)
                                        )
                replay.apply_event(state, e)
            for seat, uid in enumerate(ids):
                stats[uid]["rounds"] += 1
        if (index + 1) % 50 == 0:
            print(f"  ... {index + 1}/{len(paths)}", file=sys.stderr, flush=True)

    print(f"{'uid':>16s} {'局':>6s} {'副露/局':>8s} {'打财神‰':>8s} "
          f"{'首次到听摸序':>12s} {'听口宽':>7s}  到听率(序号5/8/10)")
    for uid in [u for u in args.focus if stats[u]["rounds"]] + ([OUR] if stats[OUR]["rounds"] else []):
        s = stats[uid]
        tag = "  ← 我们" if uid == OUR else ""
        rate = lambda k: (  # noqa: E731
            100 * s["ordinal_tenpai"][k] / s["ordinal_all"][k] if s["ordinal_all"][k] else float("nan")
        )
        print(f"{uid:>16s} {s['rounds']:>6d} {s['melds'] / max(1, s['rounds']):>8.2f} "
              f"{1000 * s['god_discards'] / max(1, s['discards']):>8.1f} "
              f"{(statistics.mean(s['draw_to_tenpai']) if s['draw_to_tenpai'] else float('nan')):>12.2f} "
              f"{(statistics.mean(s['widths']) if s['widths'] else float('nan')):>7.2f} "
              f"{rate(5):>6.1f}/{rate(8):>4.1f}/{rate(10):>4.1f}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
