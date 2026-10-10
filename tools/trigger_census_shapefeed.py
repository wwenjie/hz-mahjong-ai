"""第五类触发点普查：**形质 × 喂牌冲突点**（seq21/23 那类「该打孤张却打了复合搭」）。

**触发定义（三条同时成立，全部可判定）**：我方出牌时刻，在**最小向听层**里取
`best_shape = argmax(blocks)`（`blocks` = `shape_value` 打完之后的手牌形质）与 `v7` 的**实选**：
1. `v7` 实选 ≠ `best_shape`；
2. `best_shape.feed_cost > v7实选.feed_cost`（即：形质更好的那张**更喂牌**）；
3. 两者 `god_penalty` 相等（排除「打财神」这个独立因素）。
⇒ 语义：**为了少喂牌而牺牲形质**。这正是用户反复报的那一类
（seq23 手 `2w2w 4w 7w7w8w9w 3b 4t5t5t 8t9t9t`：`best_shape` 是打孤张 3b（形质 5.12）、
`v7` 打 8t（形质 5.06，但 `visible_need` 说 3b 危险 1.67× ⇒ `3×Δfeed=1.46` 压过 0.06）。

**为什么要单独普查**：`feed_weight` 的**整体缩放**已双侧关闭（3.0→1.0 与 3.0→6.0 都更差），
但那是「缩放一个惩罚系数」；本轴问的是**结构规则**——「在喂牌与形质直接冲突的那些点上，
该不该听形质」。两者不是同一件事，前者是标定、后者是可判定的取舍规则。

产出 JSONL 供 `tools/trigger_counterfactual.py --mode discard --force-tile` 用：
`offered` = 记录 `v7` 实选，`arm_tile` = `best_shape.tile`（强制打形质更好的那张）。

用法::
    .venv/bin/python tools/trigger_census_shapefeed.py --rooms 1200 --jobs 8 \
        --out agent/out/trigger-points/shapefeed.jsonl
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
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def scan(files: list[str], dec, arm: str) -> tuple[list[dict], dict]:
    stats: collections.Counter = collections.Counter()
    gaps: collections.Counter = collections.Counter()
    points: list[dict] = []
    for path in files:
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
            stats["局"] += 1
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
                try:
                    scores = [dec._score_discard(sit, action) for action in discards]  # noqa: SLF001
                    pick = dec.choose(sit, actions, budget_ms=2000)
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if pick is None or pick.kind != DISCARD:
                    replay.apply_event(state, event)
                    continue
                top = min(item.shanten for item in scores)
                layer = [item for item in scores if item.shanten == top]
                mine_score = next((item for item in scores if item.tile == pick.tile), None)
                if mine_score is None or len(layer) < 2:
                    replay.apply_event(state, event)
                    continue
                best_shape = max(layer, key=lambda item: (item.blocks, item.total))
                stats["可比点"] += 1
                # 对照口径（不分因）：实选是否就是「形质最高的那张」
                if best_shape.tile == pick.tile:
                    stats["实选=形质最高"] += 1
                    replay.apply_event(state, event)
                    continue
                stats["实选≠形质最高"] += 1
                if best_shape.god_penalty != mine_score.god_penalty:
                    stats["  └因财神罚"] += 1
                elif best_shape.feed_cost > mine_score.feed_cost:
                    stats["  └因喂牌更高"] += 1
                elif best_shape.blocks == mine_score.blocks:
                    stats["  └形质并列(破平层/喂牌定)"] += 1
                else:
                    stats["  └其他"] += 1
                if (best_shape.tile == pick.tile
                        or best_shape.god_penalty != mine_score.god_penalty
                        or best_shape.feed_cost <= mine_score.feed_cost):
                    replay.apply_event(state, event)
                    continue
                # 触发：为了少喂牌牺牲形质。
                stats["触发(形质×喂牌冲突)"] += 1
                delta_total = mine_score.total - best_shape.total   # >0 ⇒ 实选主分更高
                buckets = ("<0.5" if delta_total < 0.5 else
                           "[0.5,1)" if delta_total < 1 else
                           "[1,2)" if delta_total < 2 else "≥2")
                stats[f"total差{buckets}"] += 1
                gaps[f"形质差{best_shape.blocks - mine_score.blocks:.2f}"] += 1
                klass = "形质×喂牌|向听" + str(min(top, 4))
                stats[klass] += 1
                points.append({
                    "room": doc.get("room_id"),
                    "file": str(path),
                    "round_no": state.round_no,
                    "block": block_no,
                    "ev_index": ev_index,
                    "seq": event.get("seq"),
                    "hand_counts": hand,
                    "melds": [list(m.tiles) for m in state.seats[mine].melds],
                    "god_n": int(hand[tiles.GOD]),
                    "chain": int(state.seats[mine].chain_count),
                    "piao": int(state.seats[mine].piao_count),
                    "current_shanten": top,
                    "offered": pick.tile,          # 基线（v7 实选）
                    "arm_tile": best_shape.tile,   # 处理臂（强制打形质更好的那张）
                    "delta_total": round(delta_total, 3),
                    "delta_blocks": round(best_shape.blocks - mine_score.blocks, 3),
                    "delta_feed": round(best_shape.feed_cost - mine_score.feed_cost, 3),
                    "klass": klass,
                })
                replay.apply_event(state, event)
    return points, dict(stats)


def _worker(payload):
    files, arm = payload
    from majiang.cli import make_decider as build
    from majiang.strategy.policy import Mode as M

    return scan(files, build(arm, M.QUALIFIER), arm)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="形质×喂牌冲突点普查")
    ap.add_argument("--rooms", type=int, default=1200)
    ap.add_argument("--arm", default="v7")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="agent/out/trigger-points/shapefeed.jsonl")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunks = [files[index::args.jobs] for index in range(args.jobs)]
        stats: collections.Counter = collections.Counter()
        points: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for part, part_stats in pool.map(_worker, [(chunk, args.arm) for chunk in chunks]):
                points.extend(part)
                stats.update(part_stats)
    else:
        points, raw = scan(files, make_decider(args.arm, Mode.QUALIFIER), args.arm)
        stats = collections.Counter(raw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for point in points:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")

    rounds = max(1, stats["局"])
    comparable = max(1, stats["可比点"])
    print(f"扫描 {len(files)} 房 → 命中我方 {stats['命中房']} 房 / {stats['局']} 局")
    print(f"  我方弃牌点 {stats['我方弃牌点']}  可比点 {stats['可比点']}")
    print(f"  **触发（为少喂牌牺牲形质）{stats['触发(形质×喂牌冲突)']}"
          f" = {stats['触发(形质×喂牌冲突)'] / comparable:.1%}**"
          f"  每局 {stats['触发(形质×喂牌冲突)'] / rounds:.3f}")
    for key in ("<0.5", "[0.5,1)", "[1,2)", "≥2"):
        if stats[f"total差{key}"]:
            print(f"    实选主分领先 {key:<8} {stats[f'total差{key}']}")
    print(f"  对照口径：实选=形质最高 {stats['实选=形质最高']}"
          f"（{stats['实选=形质最高'] / comparable:.1%}）；"
          f"实选≠形质最高 {stats['实选≠形质最高']}"
          f"（{stats['实选≠形质最高'] / comparable:.1%}）——穷因：")
    for key in ("  └因喂牌更高", "  └因财神罚", "  └形质并列(破平层/喂牌定)", "  └其他"):
        if stats[key]:
            print(f"    {key} {stats[key]}（{stats[key] / comparable:.2%}）")
    for key in sorted(k for k in stats if k.startswith("形质×喂牌|")):
        print(f"    {key} {stats[key]}")
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
