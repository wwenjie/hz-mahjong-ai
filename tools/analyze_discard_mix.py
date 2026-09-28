"""弃牌牌种分布：我们 vs 对手（只读事件流，便宜）。

**为什么做**：我们的中段决策 97.7% 由 `feed` 决定，而 `feed` 的字面含义是
「中张(3-7) 对对手吸引力 1.0 > 边张 0.6 > 字牌 0.4」⇒ 我们的先验就是**留中张**。
所以：若**对手反而更多打中张**，说明我们「留中张」留过头了——与
`tools/calibrate_threat.py` 测出的「喂牌项被放大 2.6 倍」互为印证。

口径：按「打出牌的牌种」分三桶（字牌 / 边张 1,2,8,9 / 中张 3-7），
只看**自由出牌点**（剔抓打圈强制），按全局占比对比。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

from majiang.rules import tiles
from majiang.sim import replay

SEATS = 4
OUR = "u_a7f7c67bb14a"


def bucket(tile: int) -> str:
    if not tiles.is_number(tile):
        return "字牌"
    rank = tiles.rank(tile)
    return "中张(3-7)" if 3 <= rank <= 7 else "边张(1,2,8,9)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="弃牌牌种分布")
    parser.add_argument("--rooms", type=int, default=20, help="按房抽样的房数")
    args = parser.parse_args(argv)

    by_room: dict[str, list[str]] = {}
    for path in glob.glob("data/auto_sessions/*/events/*.json"):
        by_room.setdefault(str(Path(path).parts[-3]), []).append(path)
    picked: list[str] = []
    for room in sorted(by_room)[: args.rooms]:
        picked.extend(sorted(by_room[room]))

    counts: dict[str, collections.Counter] = {"我们": collections.Counter(),
                                              "对手": collections.Counter()}
    for path in picked:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != SEATS or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for state, events in replay.iter_rounds(doc):
            for e in events:
                if (
                    e.get("type") == "tile_discarded"
                    and isinstance(e.get("seat"), int)
                    and state.opened
                    and not (e.get("data") or {}).get("catch_play")
                ):
                    tile = replay._tile_of(e.get("tile"))  # noqa: SLF001
                    if tile is None:
                        continue
                    group = "我们" if e["seat"] == mine else "对手"
                    counts[group][bucket(tile)] += 1
                replay.apply_event(state, e)

    total = {g: sum(c.values()) for g, c in counts.items()}
    print(f"自由出牌点：我们 {total.get('我们', 0)}、对手 {total.get('对手', 0)}（{args.rooms} 房）\n")
    print(f"{'牌种':>12s} {'我们':>8s} {'对手':>8s} {'差':>8s}")
    for name in ("字牌", "边张(1,2,8,9)", "中张(3-7)"):
        pa = counts["我们"][name] / max(1, total.get("我们", 1))
        pb = counts["对手"][name] / max(1, total.get("对手", 1))
        print(f"{name:>12s} {pa:>8.1%} {pb:>8.1%} {pa - pb:>+8.1%}")
    print("\n读法：若对手更多打**中张** ⇒ 我们「留中张」留过头，与喂牌项被放大 2.6 倍互为印证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
