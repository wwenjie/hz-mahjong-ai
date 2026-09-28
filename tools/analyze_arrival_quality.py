"""出牌质量 vs 副露速度：强 bot 的「进入 1/2 向听时进张数」vs 我们（只读）。

**要分解的问题**：对手侧指纹显示强 bot 相对我们有两个差——**副露 2 倍**、
**按序号的到听率高 21~26pp**。这两者哪一个是主因？
- 若他们在「进入 1 向听那一刻」的**精确进张**也明显高于我们 ⇒ 他们的**出牌质量**更好
  （那条路要靠 `shape_value` / `feed` 权重这类改动去追）；
- 若进张相近而只是到听更快 ⇒ 主因是**副露带来的推进**（那条路要靠副露闸门）。

**口径**：逐座重建；每局每座**首次进入 1/2 向听**时（出牌后判定）记一次精确进张
（`shanten.ukeire` 的牌种数与剩余张数之和）。与 agent B 的 `verify/arrival_shape.py` 同口径
（真机实测：进 2 向听 我们 9.90 / 对手 10.14；进 1 向听 我们 6.05 / 对手 6.70）。

**按房抽样**（不按文件——今晚刚踩过这个坑）。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"


def rooms_first(paths: list[str], rooms: int) -> list[str]:
    by_room: dict[str, list[str]] = {}
    for path in paths:
        by_room.setdefault(str(Path(path).parts[-3]), []).append(path)
    picked: list[str] = []
    for room in sorted(by_room)[:rooms]:
        picked.extend(sorted(by_room[room]))
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="出牌质量：进入各向听时的精确进张")
    parser.add_argument("--focus", nargs="*", default=[])
    parser.add_argument("--rooms", type=int, default=6, help="按房抽样的房数")
    args = parser.parse_args(argv)

    paths = rooms_first(sorted(glob.glob("data/auto_sessions/*/events/*.json")), args.rooms)
    per_uid: dict[str, dict[int, list[tuple[int, int]]]] = collections.defaultdict(
        lambda: {1: [], 2: []}
    )
    for path in paths:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4:
            continue
        for state, events in replay.iter_rounds(doc):
            arrived: set[tuple[int, int]] = set()  # (seat, level)
            for e in events:
                if (
                    e.get("type") == "tile_discarded"
                    and isinstance(e.get("seat"), int)
                    and state.opened
                    and not (e.get("data") or {}).get("catch_play")
                ):
                    seat = e["seat"]
                    tile = replay._tile_of(e.get("tile"))  # noqa: SLF001
                    counts = list(state.seats[seat].hand)
                    if tile is not None:
                        if counts[tile] > 0:
                            counts[tile] -= 1
                        melds = len(state.seats[seat].melds)
                        try:
                            value = sm.shanten_any(counts, melds)
                        except Exception:  # noqa: BLE001
                            value = -1
                        for level in (1, 2):
                            if value == level and (seat, level) not in arrived:
                                arrived.add((seat, level))
                                entries = sm.ukeire(counts, melds)
                                per_uid[ids[seat]][level].append(
                                    (len(entries), sum(c for _, c in entries))
                                )
                replay.apply_event(state, e)

    print(f"按房抽样 {args.rooms} 房（{len(paths)} 文件）\n")
    print(f"{'uid':>16s} {'进2向听 n':>9s} {'种':>6s} {'张':>6s} "
          f"{'进1向听 n':>9s} {'种':>6s} {'张':>6s}")
    order = [u for u in args.focus if per_uid[u][1]] + ([OUR] if per_uid[OUR][1] else [])
    for uid in order:
        data = per_uid[uid]
        tag = "  ← 我们" if uid == OUR else ""
        row = f"{uid:>16s}"
        for level in (2, 1):
            vals = data[level]
            if not vals:
                row += f" {'-':>9s} {'-':>6s} {'-':>6s}"
                continue
            kinds = statistics.mean(k for k, _ in vals)
            copies = statistics.mean(c for _, c in vals)
            row += f" {len(vals):>9d} {kinds:>6.2f} {copies:>6.1f}"
        print(row + tag)
    print("\n读法：若进张数明显低于对手 ⇒ 他们的**出牌质量**更好（要改 shape/权重）；"
          "\n若进张相近而到听更快 ⇒ 主因是**副露推进**（要改副露闸门）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
