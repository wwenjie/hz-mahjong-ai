"""出牌层「破平层覆盖主分」普查（回答用户 #7/#9）。

`_choose_discard` 的次序是：先按主分 ``total = -10×向听 + 形质 - 3×喂牌 - 财神罚`` 排序，
再交给 `_break_ties_by_ukeire`——后者在**同向听**候选里**只按精确进张**重排并取最优。
于是：只要某张牌进张多 1 张，主分里的「喂牌/形质/字牌安全」全部被忽略。

本工具对每个我方弃牌点比较：
  * ``main``  = 主分 argmax（未覆盖）
  * ``final`` = v5 实选（含覆盖）
统计覆盖发生率；对被覆盖掉的主分赢家分类（**孤张字牌 / 孤张数牌 / 其他**），
因为用户两次报的（打「发」、打孤张）都落在「主分赢家是孤张、却被进张多 1 张的另一张挤掉」上。

用法::
    .venv/bin/python tools/trigger_census_override.py --rooms 200 --jobs 3 --out agent/out/trigger-points/override.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def _isolated(counts, tile: int) -> bool:
    """孤张：不是财神，且同花色 ±2 内没有任何己方牌。"""
    if tile == tiles.GOD:
        return False
    if tile >= 27:  # 字牌
        return counts[tile] == 1
    base = (tile // 9) * 9
    for d in (-2, -1, 1, 2):
        u = tile + d
        if base <= u < base + 9 and counts[u] > 0:
            return False
    return True


def scan_file(path: str) -> tuple[list[dict], collections.Counter]:
    dec_v5 = make_decider("v5", Mode.QUALIFIER)
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return [], collections.Counter()
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    if len(ids) != 4 or OUR not in ids:
        return [], collections.Counter()
    mine = ids.index(OUR)
    stats: collections.Counter = collections.Counter()
    points: list[dict] = []
    for block_no, (state, events) in enumerate(replay.iter_rounds(doc)):
        for ev_index, event in enumerate(events):
            if (event.get("type") != "tile_discarded" or event.get("seat") != mine
                    or (event.get("data") or {}).get("catch_play") or not state.opened):
                replay.apply_event(state, event)
                continue
            hand = list(state.seats[mine].hand)
            if sum(hand) % 3 != 2:
                replay.apply_event(state, event)
                continue
            try:
                sit = state.situation_for(mine, phase=PHASE_DRAW, drawn=None)
                actions = legal_actions(sit)
            except Exception:  # noqa: BLE001
                replay.apply_event(state, event)
                continue
            discards = [a for a in actions if a.kind == DISCARD]
            if len(discards) < 2:
                replay.apply_event(state, event)
                continue
            stats["我方弃牌点"] += 1
            scores = [dec_v5._score_discard(sit, a) for a in discards]  # noqa: SLF001
            main = max(scores, key=lambda s: s.total)
            final = dec_v5.choose(sit, actions, budget_ms=2000)
            if final is None or final.kind != DISCARD:
                replay.apply_event(state, event)
                continue
            if final.tile == main.tile:
                stats["主分=实选"] += 1
            else:
                stats["被覆盖"] += 1
                if _isolated(hand, main.tile):
                    if main.tile >= 27:
                        stats["覆盖|主分赢家=孤张字牌"] += 1
                    else:
                        stats["覆盖|主分赢家=孤张数牌"] += 1
                else:
                    stats["覆盖|主分赢家=有搭"] += 1
                points.append({
                    "room": doc.get("room_id"),
                    "file": path,
                    "round_no": state.round_no,
                    "block": block_no,
                    "ev_index": ev_index,
                    "hand_counts": hand,
                    "melds": [list(m.tiles) for m in state.seats[mine].melds],
                    "god_n": int(hand[tiles.GOD]),
                    "current_shanten": shanten_mod.shanten_any(hand, len(state.seats[mine].melds)),
                    "main_tile": main.tile,
                    "main_total": main.total,
                    "final_tile": final.tile,
                    "final_reason": dec_v5.last_reason,
                })
            replay.apply_event(state, event)
    return points, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="破平覆盖主分 普查")
    ap.add_argument("--rooms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="agent/out/trigger-points/override.jsonl")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    files = sorted(random.sample(files, min(args.rooms, len(files))))
    print(f"抽样 {len(files)} 房（共 {len(glob.glob(str(ROOT / 'data/auto_sessions/*/events/*.json')))} 房）", flush=True)

    total: collections.Counter = collections.Counter()
    points: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for pts, st in pool.map(scan_file, files, chunksize=2):
                points.extend(pts)
                total.update(st)
    else:
        for f in files:
            pts, st = scan_file(f)
            points.extend(pts)
            total.update(st)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in points:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    n = max(1, total["我方弃牌点"])
    print("=" * 72)
    print(f"我方弃牌点 {total['我方弃牌点']}")
    print(f"  主分=实选 {total['主分=实选']}（{total['主分=实选']/n*100:.1f}%）")
    print(f"  被覆盖   {total['被覆盖']}（{total['被覆盖']/n*100:.1f}%）")
    print(f"    覆盖|主分赢家=孤张字牌 {total['覆盖|主分赢家=孤张字牌']}"
          f"（{total['覆盖|主分赢家=孤张字牌']/n*100:.1f}%）")
    print(f"    覆盖|主分赢家=孤张数牌 {total['覆盖|主分赢家=孤张数牌']}"
          f"（{total['覆盖|主分赢家=孤张数牌']/n*100:.1f}%）")
    print(f"    覆盖|主分赢家=有搭   {total['覆盖|主分赢家=有搭']}"
          f"（{total['覆盖|主分赢家=有搭']/n*100:.1f}%）")
    print(f"逐点: {out}（{len(points)} 行）")
    for p in points[: args.examples]:
        hand = "".join(tiles.to_codes([t for t in range(tiles.TILE_KINDS) for _ in range(p["hand_counts"][t])]))
        print(f"    手={hand} 主分打={tiles.to_code(p['main_tile'])}({p['main_total']:+.2f}) "
              f"实选={tiles.to_code(p['final_tile'])} 向听={p['current_shanten']} 财神={p['god_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
