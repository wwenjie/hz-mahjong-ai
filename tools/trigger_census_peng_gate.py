"""题外普查：**碰的闸门**（与 `trigger_census.py` 的「吃」互补）。

**为什么单独做**（2026-10-09 15:10 A）：今晚量到副露是速度差距的核心机制
（我们 0.629 vs 强 bot 1.218），而**机会→副露 的转化率**：
**吃 22.4% vs 43.8%、碰 25.4% vs 64.0%**——**碰的差距更大**；且 `meld_tolerance=equal`
的「放宽闸门」我此前**只在「≥2 种吃法」的窗口上测过**（−0.790、整轴关闭），
**碰从未被测**。碰与吃结构不同（**碰不占「吃全局最多 2 副」的额度**）⇒ 很可能不同结论。

产出与 `trigger_census.py` 同 schema（`v5_pick`/`cand_pick`/`options`），
供 `tools/trigger_counterfactual.py --mode response --force-trigger` 直接对拍
（`resolve_responses` 先判碰后判吃，`--force-trigger` 会在该窗口用处理臂裁决）。

用法::
    .venv/bin/python tools/trigger_census_peng_gate.py --rooms 0 --shard 0 --shards 6 \\
        --arm v7m --out agent/out/trigger-points/penggate-shard0.jsonl
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
from majiang.rules.action import PENG, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_RESPONSE_PENG  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="碰窗口的闸门普查（v5 vs 处理臂）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--arm", default="v7m", help="处理臂（默认放宽闸门 v7m）")
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--out", default="agent/out/trigger-points/penggate.jsonl")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)

    dec_off = make_decider("v5", Mode.QUALIFIER)
    dec_on = make_decider(args.arm, Mode.QUALIFIER)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))
    if args.shards > 1:
        files = files[args.shard::args.shards]

    stats: collections.Counter = collections.Counter()
    points: list[dict] = []
    examples: list[str] = []

    for index, path in enumerate(files):
        if args.progress and (index + 1) % args.progress == 0:
            print(f"  [进度] {index + 1}/{len(files)} 房，点 {len(points)}", flush=True)
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for block_no, (state, events) in enumerate(replay.iter_rounds(doc)):
            for ev_index, event in enumerate(events):
                if (event.get("type") != "tile_discarded" or event.get("seat") == mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                offered = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if offered is None or offered == tiles.GOD:
                    replay.apply_event(state, event)
                    continue
                try:
                    sit = state.situation_for(
                        mine, phase=PHASE_RESPONSE_PENG, offered=offered, responding=(mine,)
                    )
                    acts = legal_actions(sit)
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if not any(a.kind == PENG for a in acts):
                    replay.apply_event(state, event)
                    continue
                stats["碰窗口"] += 1
                pick_off = dec_off.choose(sit, acts, budget_ms=2000)
                pick_on = dec_on.choose(sit, acts, budget_ms=2000)
                if pick_off is None or pick_on is None:
                    replay.apply_event(state, event)
                    continue
                if pick_off.kind != PENG and pick_on.kind == PENG:
                    stats["v5 pass 但处理臂碰"] += 1
                    hand = sit.hand
                    if len(examples) < args.examples:
                        hand_str = "".join(
                            tiles.to_codes(
                                [t for t in range(tiles.TILE_KINDS) for _ in range(hand.counts[t])]
                            )
                        )
                        examples.append(
                            f"    {Path(path).stem} 手={hand_str} 打={tiles.to_code(offered)} "
                            f"财神={int(hand.counts[tiles.GOD])} v5={pick_off.describe()} "
                            f"→ {args.arm}={pick_on.describe()} reason={dec_off.last_reason[:40]}"
                        )
                    points.append({
                        "room": doc.get("room_id"),
                        "file": str(Path(path)),
                        "round_no": state.round_no,
                        "block": block_no,
                        "ev_index": ev_index,
                        "seq": event.get("seq"),
                        "offered": offered,
                        "hand_counts": list(sit.hand.counts),
                        "melds": [list(m.tiles) for m in sit.all_melds],
                        "god_n": int(sit.hand.counts[tiles.GOD]),
                        "chain": int(sit.god.chain_count),
                        "piao": int(sit.god.piao_count),
                        "current_shanten": None,
                        "v5_pick": pick_off.describe(),
                        "cand_pick": pick_on.describe(),
                        "klass": "碰闸门",
                        "options": [],
                    })
                replay.apply_event(state, event)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for point in points:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")
    print(f"扫描 {len(files)} 房 → 碰窗口 {stats['碰窗口']}"
          f"，**v5 pass 但 {args.arm} 碰 = {stats['v5 pass 但处理臂碰']}**"
          f"（{stats['v5 pass 但处理臂碰'] / max(1, stats['碰窗口']):.1%} of 碰窗口）")
    for line in examples:
        print(line)
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
