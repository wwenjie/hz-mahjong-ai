"""触发点普查（P1 底座）：把**真机**里「≥2 种吃法」的响应窗口抽成数据集 + 定义性校验。

产出（每行一个触发点，JSONL）：
```
{room, file, round_no, block, ev_index, seq, offered,
 hand_counts, melds, god_n, chain, piao, catch_play, god_discarder,
 current_shanten, v5_pick, cand_pick, klass,
 options: [{tiles, after_shanten, old_value, ukeire_copies, ukeire_kinds}]}
```
`klass`：`漏吃`（v5 判无改善而实际有吃法降向听）/ `吃法错` / `次排序`（各吃法降幅相同）。

**定义性校验（P1 的必过门）**：`cand_pick` 必须等于 `argmin after_shanten` 的吃法——
这是「度量是否与定义一致」的可判定检验，不需要 A/B（用户 22:17 裁定采纳门的一部分）。

用法::
    .venv/bin/python tools/trigger_census.py --rooms 0            # 0 = 全部
    .venv/bin/python tools/trigger_census.py --rooms 200 --out agent/out/trigger-points/200.jsonl
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
from majiang.rules.action import CHI, PASS, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_RESPONSE_CHI  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真机触发点普查（≥2 种吃法的响应窗口）")
    ap.add_argument("--rooms", type=int, default=0, help="抽样房数，0 = 全部")
    ap.add_argument("--arm", default="v7-keepchi", help="处理臂（默认 ①+②；测放宽副露闸门给 v7m）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out", default="agent/out/trigger-points/all.jsonl")
    ap.add_argument("--examples", type=int, default=2, help="每类打印几个例子")
    args = ap.parse_args(argv)

    dec_off = make_decider("v5", Mode.QUALIFIER)  # 冠军口径：各吃法共用 combos[0]
    dec_on = make_decider(args.arm, Mode.QUALIFIER)  # 处理臂（默认 ①+②；测放宽闸门时给 v7m）

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))
    if args.shards > 1:
        files = files[args.shard::args.shards]

    stats = collections.Counter()
    points: list[dict] = []
    bad: list[str] = []
    examples: dict[str, list[str]] = collections.defaultdict(list)

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
                if (event.get("type") != "tile_discarded" or event.get("seat") == mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                offered = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if offered is None:
                    replay.apply_event(state, event)
                    continue
                try:
                    sit = state.situation_for(
                        mine, phase=PHASE_RESPONSE_CHI, offered=offered, responding=(mine,)
                    )
                    acts = [a for a in legal_actions(sit) if a.kind == CHI]
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if not acts:
                    replay.apply_event(state, event)
                    continue
                # **吃牌窗口只给下家**（引擎 `resolve_responses` 的 `chi_seat=(discarder+1)%4`）。
                # 不筛这个，CF 阶段会有 2/3 的点被判「非下家」丢掉（2026-10-09 13:35 实测：
                # 5,000 点里 3,282 个被排除）⇒ 在普查侧就筛掉，省算力也省误读。
                discarder_seat = event.get("seat")
                if not isinstance(discarder_seat, int) or (discarder_seat + 1) % 4 != mine:
                    stats["非下家跳过"] += 1
                    replay.apply_event(state, event)
                    continue
                stats["可吃窗口"] += 1
                if len(acts) < 2:
                    replay.apply_event(state, event)
                    continue
                stats["≥2吃法"] += 1
                hand = sit.hand
                current = shanten_mod.shanten_any(hand.counts, hand.meld_count)
                try:
                    old_vals = [dec_off._meld_after_shanten(sit, offered, a) for a in acts]  # noqa: SLF001
                    new_vals = [dec_on._meld_after_shanten(sit, offered, a) for a in acts]  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                pick_off = dec_off.choose(sit, [*acts, next(
                    (a for a in legal_actions(sit) if a.kind == PASS), acts[0])], budget_ms=2000)
                pick_on = dec_on.choose(sit, [*acts, next(
                    (a for a in legal_actions(sit) if a.kind == PASS), acts[0])], budget_ms=2000)
                options = []
                for act, old_v, new_v in zip(acts, old_vals, new_vals):
                    copies = 0
                    try:
                        copies = dec_on._meld_ukeire_copies(sit, offered, act, new_v)  # noqa: SLF001
                    except Exception:  # noqa: BLE001
                        pass
                    options.append({
                        "tiles": list(act.tiles),
                        "after_shanten": new_v,
                        "old_value": old_v,
                        "ukeire_copies": copies,
                    })
                best_new = min(new_vals)
                best_old = min(old_vals)
                if best_old >= current and best_new < current:
                    klass = "漏吃"
                elif best_old < current and best_new < best_old:
                    klass = "吃法错"
                elif len(set(new_vals)) == 1:
                    klass = "次排序"
                else:
                    klass = "其他"
                stats[klass] += 1
                # 定义性校验：该吃却没吃（pass）时，须区分「度量错」与「七对门槛/路线否决」；
                # 吃了就必须吃到 after_shanten 最小的那个吃法——这是可判定的正确性，不需要 A/B。
                if best_new < current:
                    stats["定义性-可判定点"] += 1
                    argmin_tiles = [o["tiles"] for o in options if o["after_shanten"] == best_new]
                    if pick_on.kind == PASS:
                        stats["定义性-被路线否决(pass)"] += 1
                    elif list(pick_on.tiles) not in argmin_tiles:
                        stats["定义性-不一致"] += 1
                        if len(bad) < 5:
                            bad.append(
                                f"{Path(path).stem} block{block_no} ev{ev_index} "
                                f"pick={pick_on.describe()} 最小={argmin_tiles} 新={new_vals}"
                            )
                if klass in examples and len(examples[klass]) < args.examples:
                    examples[klass].append(
                        f"    {Path(path).stem} 当前={current} 旧={old_vals} 新={new_vals} "
                        f"v5={pick_off.describe()} 候选={pick_on.describe()} "
                        f"我的牌={''.join(tiles.to_codes([t for t in range(tiles.TILE_KINDS) for _ in range(hand.counts[t])]))}"
                    )
                points.append({
                    "room": doc.get("room_id"),
                    "file": str(Path(path)),
                    "round_no": state.round_no,
                    "block": block_no,
                    "ev_index": ev_index,
                    "seq": event.get("seq"),
                    "offered": offered,
                    "hand_counts": list(hand.counts),
                    "melds": [list(m.tiles) for m in sit.all_melds],
                    "god_n": int(hand.counts[tiles.GOD]),
                    "chain": int(sit.god.chain_count),
                    "piao": int(sit.god.piao_count),
                    "catch_play": bool(state.catch_play),
                    "god_discarder": int(state.god_discarder),
                    "current_shanten": current,
                    "v5_pick": pick_off.describe(),
                    "cand_pick": pick_on.describe(),
                    "klass": klass,
                    "options": options,
                })
                replay.apply_event(state, event)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for point in points:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")

    rounds = max(1, stats["局"])
    print(f"扫描 {len(files)} 房 → 命中我方 {stats['命中房']} 房 / {stats['局']} 局")
    for key in ("可吃窗口", "≥2吃法", "漏吃", "吃法错", "次排序", "其他"):
        value = stats[key]
        print(f"  {key:<12} {value:>7}   每局 {value / rounds:.4f}   每场(8局) {value / rounds * 8:.3f}")
    print(f"  定义性校验：可判定点 {stats['定义性-可判定点']}，不一致 {stats['定义性-不一致']}"
          f"，被路线否决(pass) {stats['定义性-被路线否决(pass)']}"
          f" ⇒ {'**通过**' if stats['定义性-不一致'] == 0 else '**未通过（见下）**'}")
    for line in bad:
        print(f"    不一致: {line}")
    for klass, lines in examples.items():
        if lines:
            print(f"  [{klass}] 例：")
            for line in lines:
                print(line)
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
