"""第三类触发点普查：**响应窗口「碰掉七对」**（回答用户第③问：该不该碰东）。

触发定义（真机 replay）：
1. 他家打出某张牌、轮到我方响应（`PHASE_RESPONSE_PENG`），且**我方手里持有 ≥2 张**（可碰）；
2. 碰之前，我方是一手**七对牌**（`seven_pairs_shanten <= --sp-max`，默认 2，即七对已 ≤2 向听）；
3. 碰会让 `meld_count` +1 ⇒ **七对路线永久作废**（杭州规则：七对禁任何副露）。

据此分类：
* `碰毁七对(v5会碰)`：碰能把向听**严格降低**（`after < current`），且未触发 v5 的七对保护
  （`_is_pair_route` 要求 ≥5 对）⇒ **v5 会碰、但毁掉七对** ← **疑似缺口**（①②之外的第三类）。
* `碰毁七对(v5不碰)`：碰后向听不降（v5 strict 闸门 PASS）⇒ 与用户第③问同型（本例即此类）。
* `对子少(非七对)`：`sp_before > sp_max`，不是七对路线，不属本类。

用法::
    .venv/bin/python tools/trigger_census_peng.py --rooms 0            # 0 = 全部
    .venv/bin/python tools/trigger_census_peng.py --rooms 2000 --out agent/out/trigger-points/peng.jsonl
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
from majiang.rules.action import PENG, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_RESPONSE_PENG  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真机触发点普查（响应窗口·碰掉七对）")
    ap.add_argument("--rooms", type=int, default=0, help="抽样房数，0 = 全部")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--sp-max", type=int, default=2, help="七对向听 ≤ 此值才算「七对牌」，默认 2")
    ap.add_argument("--out", default="agent/out/trigger-points/peng.jsonl")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--limit-points", type=int, default=0)
    args = ap.parse_args(argv)

    dec = make_decider("v5", Mode.QUALIFIER)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))

    stats = collections.Counter()
    points: list[dict] = []
    examples: dict[str, list[str]] = collections.defaultdict(list)

    for path in files:
        if args.limit_points and len(points) >= args.limit_points:
            break
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
                if (event.get("type") != "tile_discarded" or event.get("seat") == mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                offered = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if offered is None:
                    replay.apply_event(state, event)
                    continue
                # 必须先判「我是否在响应位且能碰」——只有下家能吃，碰则是三家皆可。
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
                stats["可碰窗口"] += 1
                hand = sit.hand
                counts = list(hand.counts)
                sp_before = shanten_mod.seven_pairs_shanten(counts)
                if sp_before > args.sp_max:
                    replay.apply_event(state, event)
                    continue
                stats["碰前是七对"] += 1
                current = shanten_mod.shanten_any(counts, hand.meld_count)
                try:
                    after = dec._meld_after_shanten(sit, offered, next(a for a in acts if a.kind == PENG))  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    after = None
                # v5 的七对保护门槛（_is_pair_route：≥5 对）
                pairs = sum(c // 2 for c in counts)
                protected = hand.meld_count == 0 and pairs >= dec.config.pair_route_pairs
                try:
                    pick = dec.choose(sit, acts, budget_ms=2000)
                except Exception:  # noqa: BLE001
                    pick = None
                pick_kind = getattr(pick, "kind", None)
                will_peng = (after is not None and current is not None and after < current
                             and not protected)
                if will_peng:
                    klass = "碰毁七对(v5会碰)"
                elif after is not None and current is not None and after < current and protected:
                    klass = "碰降向听但被七对保护拦下"
                else:
                    klass = "碰毁七对(v5不碰)"
                stats[klass] += 1
                if klass == "碰毁七对(v5会碰)":
                    stats["★疑似缺口"] += 1
                hand_str = "".join(
                    tiles.to_codes([t for t in range(tiles.TILE_KINDS) for _ in range(counts[t])])
                )
                if len(examples[klass]) < args.examples:
                    examples[klass].append(
                        f"    {Path(path).stem} 局{state.round_no} 手={hand_str} 被出={tiles.to_code(offered)} "
                        f"七对={sp_before} 对数={pairs} 当前向听={current} 碰后向听={after} "
                        f"保护={protected} v5={pick_kind}"
                    )
                points.append({
                    "room": doc.get("room_id"),
                    "file": str(Path(path)),
                    "round_no": state.round_no,
                    "block": block_no,
                    "ev_index": ev_index,
                    "seq": event.get("seq"),
                    "discarder": int(event.get("seat")),
                    "offered": offered,
                    "hand_counts": counts,
                    "melds": [list(m.tiles) for m in state.seats[mine].melds],
                    "god_n": int(counts[tiles.GOD]),
                    "chain": int(state.seats[mine].chain_count),
                    "piao": int(state.seats[mine].piao_count),
                    "current_shanten": current,
                    "peng_after_shanten": after,
                    "seven_pairs_before": sp_before,
                    "pairs": pairs,
                    "protected": bool(protected),
                    "v5_pick": pick_kind,
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
    for key in ("可碰窗口", "碰前是七对", "碰毁七对(v5会碰)", "碰降向听但被七对保护拦下",
                "碰毁七对(v5不碰)"):
        v = stats[key]
        print(f"  {key:<24} {v:>7}   每局 {v / rounds:.5f}   每场(8局) {v / rounds * 8:.4f}")
    print(f"  ★疑似缺口（碰毁七对且 v5 会碰）：{stats['★疑似缺口']}")
    for klass, lines in examples.items():
        if lines:
            print(f"  [{klass}] 例：")
            for line in lines:
                print(line)
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
