"""第三类触发点普查：**破平层覆盖主分**（B' 2026-10-09 `cc5338d3` 立案，A 欠立据）。

**缺陷定义（可判定，不需要 A/B）**：`_break_ties_by_ukeire` 把「同向听」的**全部**候选
（含 `total` 明显更低的）一起丢进精确进张比较，胜者**整张替换** `scores[0]`
⇒ 主分（`total = -10×向听 + 形质 − 3×喂牌 − 财神罚`）里除向听以外的**全部信息被覆盖**。
一个名字叫 tie-break 的层，实际作用面是「同向听」而不是「同分」。

**两侧构造（零 `src/` 改动）**：
- 处理臂「主分优先」= v5 的六个旋钮 + `tiebreak="blocks"`（该值不在
  `("ukeire","exact-ukeire","tenpai-only")` 里 ⇒ 跳过破平层 ⇒ 就是 `scores[0]`）；
- 基线 = `v5`（`tiebreak="exact-ukeire"`）。
两者**只差破平层这一处**，故差分可归因。同时校验「显式旋钮构造的 exact-ukeire」与
`make_decider("v5")` 逐点同选（构造等价性自检）。

产出 JSONL（每行一个触发点，供 `tools/trigger_counterfactual.py --mode discard` 用）：
```
{room, file, round_no, block, ev_index, seq, hand_counts, melds, god_n, chain, piao,
 current_shanten, v5_tile, maxtotal_tile, gap, klass}
```
`gap` = `scores[0].total − 破平层选中者.total`（>0 即「已被覆盖」）。

用法::
    .venv/bin/python tools/trigger_census_tiebreak.py --rooms 2000 --progress 50 \
        --out agent/out/trigger-points/tiebreak.jsonl
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
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig  # noqa: E402

OUR = "u_a7f7c67bb14a"

# v5 的六个旋钮（`strategy/versions.py` 的 v5 条目逐字抄来）。
V5_KNOBS = {
    "wait_aware_tenpai": True,
    "shape_value": True,
    "ukeire_order": "blocks",
    "ukeire_max_shanten": 3,
    "ukeire_candidates": 3,
}


def build_arms():
    """返回 (v5, 显式旋钮版 v5, 主分优先版)。三角自检见 ``main``。"""
    v5 = make_decider("v5", Mode.QUALIFIER)
    v5_explicit = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER, **V5_KNOBS))
    # `tiebreak="blocks"` ⇒ `_choose_discard` 不调破平层 ⇒ 选中者 = 主分最高者。
    maxtotal = HeuristicDecider(
        PolicyConfig.for_mode(Mode.QUALIFIER, **V5_KNOBS, tiebreak="blocks")
    )
    return v5, v5_explicit, maxtotal


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="破平层覆盖主分：触发点普查")
    ap.add_argument("--rooms", type=int, default=0, help="0 = 全部")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--limit-points", type=int, default=0)
    ap.add_argument("--progress", type=int, default=25)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", default="agent/out/trigger-points/tiebreak.jsonl")
    args = ap.parse_args(argv)

    dec_v5, dec_v5x, dec_max = build_arms()

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))
    if args.shards > 1:
        files = files[args.shard::args.shards]

    stats: collections.Counter = collections.Counter()
    gaps: collections.Counter = collections.Counter()
    points: list[dict] = []
    examples: list[str] = []
    mismatch: list[str] = []
    scanned = 0

    for path in files:
        if args.limit_points and len(points) >= args.limit_points:
            break
        scanned += 1
        if args.progress and scanned % args.progress == 0:
            print(f"  [进度] {scanned}/{len(files)} 房，触发点 {len(points)}，"
                  f"覆盖率 {_rate(stats)}", flush=True)
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
                if len([a for a in actions if a.kind == DISCARD]) < 2:
                    replay.apply_event(state, event)
                    continue
                try:
                    pick_v5 = dec_v5.choose(sit, actions, budget_ms=2000)
                    pick_v5x = dec_v5x.choose(sit, actions, budget_ms=2000)
                    pick_max = dec_max.choose(sit, actions, budget_ms=2000)
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if pick_v5 is None or pick_max is None or pick_v5.kind != DISCARD:
                    replay.apply_event(state, event)
                    continue
                # 构造等价性自检：显式旋钮版必须与版本库版逐点同选，否则两侧不可比。
                if pick_v5x is None or (pick_v5x.kind, pick_v5x.tile) != (pick_v5.kind, pick_v5.tile):
                    if len(mismatch) < 5:
                        mismatch.append(f"{Path(path).stem} 局{state.round_no} ev{ev_index} "
                                        f"v5={pick_v5.describe()} 显式={pick_v5x.describe()}")
                    stats["构造不一致"] += 1
                if pick_max.kind != DISCARD:
                    replay.apply_event(state, event)
                    continue
                stats["可比点"] += 1
                if pick_max.tile == pick_v5.tile:
                    replay.apply_event(state, event)
                    continue
                # 破平层把主分最高者换掉了 ⇒ 触发点。
                scores = sorted(
                    (dec_v5x._score_discard(sit, a) for a in actions if a.kind == DISCARD),  # noqa: SLF001
                    key=lambda item: item.total, reverse=True,
                )
                best = scores[0]
                chosen = next((s for s in scores if s.tile == pick_v5.tile), None)
                gap = 0.0 if chosen is None else best.total - chosen.total
                shanten = best.shanten
                bucket = f"向听{min(shanten, 4)}"
                stats["触发点"] += 1
                stats[bucket] += 1
                gaps[_gap_bucket(gap)] += 1
                klass = f"覆盖主分|{bucket}"
                if len(examples) < args.examples:
                    hand_str = "".join(
                        tiles.to_codes([t for t in range(tiles.TILE_KINDS) for _ in range(hand[t])])
                    )
                    examples.append(
                        f"    {Path(path).stem} 局{state.round_no} 手={hand_str} "
                        f"主分选={tiles.to_code(best.tile)}({best.total:.2f}) "
                        f"v5选={tiles.to_code(pick_v5.tile)}({gap:+.2f}) 向听={shanten} "
                        f"财神={hand[tiles.GOD]}"
                    )
                points.append({
                    "room": doc.get("room_id"),
                    "file": str(Path(path)),
                    "round_no": state.round_no,
                    "block": block_no,
                    "ev_index": ev_index,
                    "seq": event.get("seq"),
                    "offered": pick_max.tile,
                    "hand_counts": hand,
                    "melds": [list(m.tiles) for m in state.seats[mine].melds],
                    "god_n": int(hand[tiles.GOD]),
                    "chain": int(state.seats[mine].chain_count),
                    "piao": int(state.seats[mine].piao_count),
                    "current_shanten": shanten,
                    "v5_tile": pick_v5.tile,
                    "maxtotal_tile": pick_max.tile,
                    "gap": round(gap, 3),
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
    print(f"  我方弃牌点 {stats['我方弃牌点']}  可比点 {stats['可比点']}")
    print(f"  **触发点（破平层换掉了主分最高者）{stats['触发点']}**"
          f"  ⇒ 覆盖率 {_rate(stats)}（空干预门要求 ≥5%）")
    for key in sorted(k for k in stats if k.startswith("向听")):
        print(f"    {key} {stats[key]}（{stats[key] / rounds:.4f}/局）")
    print("  覆盖幅度 `scores[0].total − 选中者.total` 分布：")
    for key in sorted(gaps):
        print(f"    {key:<12} {gaps[key]:>7}  ({gaps[key] / max(1, stats['触发点']):.1%})")
    if mismatch:
        print(f"  ⚠ 构造不一致 {stats['构造不一致']}：")
        for line in mismatch:
            print(f"    {line}")
    else:
        print("  ✓ 构造等价性自检通过（显式旋钮版 v5 ≡ 版本库 v5）")
    for line in examples:
        print(f"  [例]{line}")
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


def _rate(stats: collections.Counter) -> str:
    base = stats["可比点"]
    if not base:
        return "n/a"
    return f"{stats['触发点'] / base:.1%}"


def _gap_bucket(gap: float) -> str:
    if gap <= 0:
        return "0（同分，非覆盖）"
    if gap < 0.5:
        return "(0, 0.5)"
    if gap < 1:
        return "[0.5, 1)"
    if gap < 2:
        return "[1, 2)"
    if gap < 5:
        return "[2, 5)"
    return "≥5"


if __name__ == "__main__":
    raise SystemExit(main())
