"""第二类触发点普查：**出牌层拆对子**（回答「该不该拆这对东」）。

触发定义：我方自己的弃牌时刻，且 `v5` 打出的那张在手里**恰好剩 2 张**（含即将打出的这张）
⇒ 打出去就把一个对子拆成单张。记录 `v5` 与处理臂各自选什么，供
`tools/trigger_counterfactual.py --mode discard` 做「保留对子 vs 拆对子」的条件对拍。

产出 JSONL（每行一个触发点）：
```
{room, file, round_no, block, ev_index, seq, hand_counts, melds, god_n, chain, piao,
 current_shanten, pair_tile, count_of_picked, v5_tile, arm_tile, klass}
```
`klass`：`拆对(臂保留)`（v5 拆、臂不拆）/ `拆对(臂也拆)` / `其他对子相关`。

用法::
    .venv/bin/python tools/trigger_census_discard.py --rooms 2000 --out agent/out/trigger-points/pairs.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="出牌层「拆对子」触发点普查")
    ap.add_argument("--rooms", type=int, default=0, help="0 = 全部")
    ap.add_argument("--shard", type=int, default=0, help="本进程处理 ci %% shards == shard 的文件")
    ap.add_argument("--shards", type=int, default=1, help="总分片数（各片写各自的 --out，最后合并）")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--limit-points", type=int, default=0, help=">0 时抽这么多点即停")
    ap.add_argument("--progress", type=int, default=25, help="每 N 房打一次进度（0=不打）")
    ap.add_argument("--out", default="agent/out/trigger-points/pairs.jsonl")
    ap.add_argument("--arm", default="v7-keeppairs")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)

    dec_arm = make_decider(args.arm, Mode.QUALIFIER)
    dec_v5 = make_decider("v5", Mode.QUALIFIER)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))
    if args.shards > 1:
        files = files[args.shard::args.shards]

    stats: collections.Counter = collections.Counter()
    points: list[dict] = []
    examples: dict[str, list[str]] = collections.defaultdict(list)
    scanned = 0

    for path in files:
        if args.limit_points and len(points) >= args.limit_points:
            break
        scanned += 1
        if args.progress and scanned % args.progress == 0:
            # **进度必须可见**：2026-10-08 23:35 一批后台分片静默消失、日志 0 行，
            # 无法区分「没起来」和「跑到一半被杀」。按房打点后一眼能看出死活。
            print(f"  [进度] {scanned}/{len(files)} 房，已收 {len(points)} 点，"
                  f"v5拆对 {stats['v5 拆对子']}", flush=True)
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        stats["命中房"] += 1
        for block_no, (state, events) in enumerate(replay.iter_rounds(doc)):
            if args.limit_points and len(points) >= args.limit_points:
                break
            stats["局"] += 1
            for ev_index, event in enumerate(events):
                if (event.get("type") != "tile_discarded" or event.get("seat") != mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                stats["我方弃牌点"] += 1
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
                # **必须现场算 v5**：历史出牌**不等于**当时的 v5——采集器跑过 14 小时 legacy 档
                # （`heuristic,meld-equal`），本仓大量采集局不是 v5 打的（实测 200 点里
                # 「历史拆对」与「v5 现场拆对」几乎不重叠）⇒ 用历史当基线会得到 0 产出的假触发点。
                pick_v5 = dec_v5.choose(sit, actions, budget_ms=2000)
                if pick_v5 is None or pick_v5.kind != DISCARD or hand[pick_v5.tile] < 2:
                    replay.apply_event(state, event)
                    continue
                tile = pick_v5.tile
                count = hand[tile]
                stats["v5 拆对子"] += 1
                try:
                    pick_arm = dec_arm.choose(sit, actions, budget_ms=2000)
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if pick_arm is None:
                    replay.apply_event(state, event)
                    continue
                arm_breaks = hand[pick_arm.tile] >= 2 if pick_arm.kind == DISCARD else False
                shanten = shanten_mod.shanten_any(hand, len(state.seats[mine].melds))
                # 分层：向听 0 的「拆对」其实是**选听口**（另一个问题），必须与「弃掉一个搭子对」分开。
                bucket = "听牌(选听口)" if shanten == 0 else f"向听{min(shanten, 4)}"
                klass = ("拆对(臂保留)" if not arm_breaks else "拆对(臂也拆)") + "|" + bucket
                stats[klass] += 1
                if not arm_breaks:
                    stats["臂改了出牌"] += 1
                    if len(examples[bucket]) < args.examples:
                        hand_str = "".join(
                            tiles.to_codes([t for t in range(tiles.TILE_KINDS) for _ in range(hand[t])])
                        )
                        examples[bucket].append(
                            f"    {Path(path).stem} 局{state.round_no} 手={hand_str} "
                            f"v5打={tiles.to_code(tile)}（该牌 {count} 张） 臂打={pick_arm.describe()} "
                            f"向听={shanten} 财神={hand[tiles.GOD]}"
                        )
                points.append({
                    "room": doc.get("room_id"),
                    "file": str(Path(path)),
                    "round_no": state.round_no,
                    "block": block_no,
                    "ev_index": ev_index,
                    "seq": event.get("seq"),
                    "offered": tile,  # 该模式用它记录历史（＝基线）出牌
                    "hand_counts": hand,
                    "melds": [list(m.tiles) for m in state.seats[mine].melds],
                    "god_n": int(hand[tiles.GOD]),
                    "chain": int(state.seats[mine].chain_count),
                    "piao": int(state.seats[mine].piao_count),
                    "current_shanten": shanten,
                    "pair_tile": tile,
                    "count_of_picked": count,
                    "v5_tile": tile,
                    "arm_tile": pick_arm.tile if pick_arm.kind == DISCARD else -1,
                    "klass": klass,
                })
                replay.apply_event(state, event)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for point in points:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")
    rounds = max(1, stats["局"])
    print(f"扫描 {len(files)} 房 → 命中我方 {stats['命中房']} 房 / {stats['局']} 局")
    print(f"  我方弃牌点 {stats['我方弃牌点']}（每局 {stats['我方弃牌点'] / rounds:.2f}）")
    print(f"  v5 拆对子 {stats['v5 拆对子']}（每局 {stats['v5 拆对子'] / rounds:.4f}"
          f"，每场(8局) {stats['v5 拆对子'] / rounds * 8:.3f}）")
    print(f"    其中臂保留（= 本类触发点）{stats['拆对(臂保留)']}（每场 "
          f"{stats['拆对(臂保留)'] / rounds * 8:.3f}）")
    print(f"    臂也拆 {stats['拆对(臂也拆)']}")
    for klass, lines in examples.items():
        print(f"  [{klass}] 例：")
        for line in lines:
            print(line)
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
