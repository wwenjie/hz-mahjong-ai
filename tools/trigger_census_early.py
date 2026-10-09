"""触发点普查（第三类）：**早听 vs 好听**——用户 2026-10-09 10:11 提出的机制缺口。

背景：`v5` 的主键 `total = -10×向听 + 形质 - 3×喂牌 - 财神罚` 里**向听压倒一切**，
而进张次排序只在**同向听**候选之间比较 ⇒ **v5 按构造不能做「用晚一步换更宽听口」的取舍**。
实测可达面（`/tmp/dose_early_vs_wide.py`，1,603 个真机决策）：96% 存在「向听+1」候选，
**49% 里向听+1 的进张 ≥1.5× 当前最优**（持财神中位比 1.88×、无财神 2.23×）。

本工具把这类点抽出来（供 `tools/trigger_counterfactual.py --mode discard --force-tile` 对拍）：
- `v5_tile`  = v5 实际会打的（= 最小向听档里 v5 选中的那张）
- `arm_tile` = **向听+1 档里进张最多的那张**（反事实分支要强制打它）
- `klass`    = `早听窄|比{ratio:.2f}|财神{n}`（含进张比与财神数，便于分层）

用法::
    .venv/bin/python tools/trigger_census_early.py --rooms 600 --shard 0 --shards 14 \\
        --out agent/out/trigger-points/early-shard0.jsonl
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
from majiang.rules import shanten as sh  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="「早听 vs 好听」触发点普查")
    ap.add_argument("--rooms", type=int, default=0, help="0 = 全部")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--min-ratio", type=float, default=1.5, help="向听+1 候选的进张/当前最优 的阈值")
    ap.add_argument("--progress", type=int, default=25)
    ap.add_argument("--out", default="agent/out/trigger-points/early.jsonl")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)

    dec = make_decider("v5", Mode.QUALIFIER)
    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))
    if args.shards > 1:
        files = files[args.shard::args.shards]

    stats: collections.Counter = collections.Counter()
    points: list[dict] = []
    examples: list[str] = []
    scanned = 0

    for path in files:
        scanned += 1
        if args.progress and scanned % args.progress == 0:
            print(f"  [进度] {scanned}/{len(files)} 房，触发 {len(points)} 点", flush=True)
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
                if (event.get("type") != "tile_discarded" or event.get("seat") != mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                try:
                    situation = state.situation_for(mine)
                    actions = legal_actions(situation)
                    candidates = [a for a in actions if a.kind == DISCARD]
                    if len(candidates) < 2:
                        replay.apply_event(state, event)
                        continue
                    stats["决策"] += 1
                    scores = [dec._score_discard(situation, a) for a in candidates]  # noqa: SLF001
                    s0 = min(s.shanten for s in scores)
                    group0 = [s for s in scores if s.shanten == s0]
                    group1 = [s for s in scores if s.shanten == s0 + 1]
                    if not group1:
                        replay.apply_event(state, event)
                        continue
                    visible = sh.visible_counts(
                        situation.hand.counts,
                        [m.tiles for m in situation.all_melds],
                        situation.discards,
                    )
                    # v5 实际会打的（同一局面下现场算，不依赖历史）
                    pick = dec.choose(situation, actions, budget_ms=2000)
                    if pick is None or pick.kind != DISCARD or pick.tile not in {s.tile for s in group0}:
                        replay.apply_event(state, event)
                        continue

                    def copies(tile: int) -> int:
                        counts = list(situation.hand.counts)
                        counts[tile] -= 1
                        try:
                            return sum(c for _, c in sh.ukeire(
                                counts, situation.hand.meld_count, visible=visible
                            ))
                        except Exception:  # noqa: BLE001
                            return -1

                    current = copies(pick.tile)
                    if current <= 0:
                        replay.apply_event(state, event)
                        continue
                    best1 = max(group1, key=lambda s: s.blocks)
                    wide = copies(best1.tile)
                    if wide <= 0:
                        replay.apply_event(state, event)
                        continue
                    ratio = wide / current
                    if ratio < args.min_ratio:
                        replay.apply_event(state, event)
                        continue
                    god_n = situation.hand.counts[tiles.GOD]
                    klass = f"早听窄|比{ratio:.2f}|财神{god_n}"
                    stats["触发"] += 1
                    if god_n >= 1:
                        stats["触发(持财神)"] += 1
                    if len(examples) < args.examples:
                        hand = situation.hand.counts
                        hand_str = "".join(
                            tiles.to_codes([t for t in range(34) for _ in range(hand[t])])
                        )
                        examples.append(
                            f"    {Path(path).stem} 手={hand_str} 财神={god_n} | v5={tiles.to_code(pick.tile)}"
                            f"(向听{s0},{current}张) vs 宽={tiles.to_code(best1.tile)}"
                            f"(向听{s0 + 1},{wide}张, {ratio:.2f}×)"
                        )
                    points.append({
                        "room": doc.get("room_id"),
                        "file": str(Path(path)),
                        "round_no": state.round_no,
                        "block": block_no,
                        "ev_index": ev_index,
                        "seq": event.get("seq"),
                        "offered": pick.tile,
                        "hand_counts": list(situation.hand.counts),
                        "melds": [list(m.tiles) for m in situation.all_melds],
                        "god_n": int(god_n),
                        "chain": int(state.seats[mine].chain_count),
                        "piao": int(state.seats[mine].piao_count),
                        "current_shanten": s0,
                        "pair_tile": pick.tile,
                        "count_of_picked": int(situation.hand.counts[pick.tile]),
                        "v5_tile": pick.tile,
                        "arm_tile": best1.tile,
                        "wide_copies": wide,
                        "narrow_copies": current,
                        "ratio": round(ratio, 3),
                        "klass": klass,
                    })
                except Exception:  # noqa: BLE001
                    pass
                replay.apply_event(state, event)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for point in points:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")
    print(f"扫描 {len(files)} 房 → 决策 {stats['决策']}，**触发 {stats['触发']}**"
          f"（{stats['触发'] / max(1, stats['决策']):.2%} of 决策；持财神 {stats['触发(持财神)']}）")
    print(f"  判据：向听+1 的进张 ≥{args.min_ratio}× 当前最优（v5 选的是最小向听档）")
    for line in examples:
        print(line)
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
