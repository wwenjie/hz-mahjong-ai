#!/usr/bin/env python
"""局末达听比例·无条件口径（A 2026-10-06 17:28② [待C] 交办）。

**目的**：排除 17:09「终局向听分布（未胡者 −14pp）」的内生性——
「未胡者」是内生子集（bot 胡得多 ⇒ 它的未胡者更少、构成不同），两边未胡者不是同一把尺。

**口径（照抄 A 17:28②）**：`局末（他人胡/流局那一刻）我方达到听牌的比例`，**全座·局口径（不筛未胡者）**。
- 我方：每一局（去重）都有一个记录——局末时刻我方是否 shanten==0（听牌/已胡都算达到）。
- bot：同一局里每个目标 bot 座位一个记录。
- 分桶：起手财神数 × 终局副露 × 终局巡目段（与 17:09 探针同口径，便于对照）。

**判据（照抄 A 17:28②）**：**无条件口径也显示我方低 ~14pp ⇒ 分支① 稳**；**若几乎无差 ⇒ 未胡者口径降级为方向**。

产物：`agent/out/final-tenpai-rate.txt`。
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent" / "verify"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay as R  # noqa: E402

import stage_a_dataset_export as EXP  # noqa: E402

OUR = "u_a7f7c67bb14a"
GOD_ID = tiles.GOD


def build_targets(files):
    name_to_uid = {}
    for fpath in files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception:  # noqa: BLE001
            pass
    return {name_to_uid[n]: n for n in EXP.TOP_BOTS if n in name_to_uid}


def god_bucket(n): return "god=0" if n == 0 else ("god=1" if n == 1 else "god≥2")
def meld_bucket(m): return "meld=0" if m == 0 else "meld≥1"
def turn_seg(t): return "早(≤5)" if t <= 5 else ("中(6-10)" if t <= 10 else "晚(≥11)")


def process_room(path, targets, stats, recs, seen_rounds):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stats["bad_json"] += 1
        return
    room_id = doc.get("room_id") or Path(path).stem
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    if len(ids) != 4 or OUR not in ids:
        stats["no_our_seat"] += 1
        return
    my_seat = ids.index(OUR)
    bot_seats = [i for i, u in enumerate(ids) if u in targets]
    if not bot_seats:
        stats["no_target_bot"] += 1
        return
    try:
        rounds = list(R.iter_rounds(doc))
    except Exception:  # noqa: BLE001
        stats["iter_rounds"] += 1
        return

    try:
        rounds_meta = {r.get("round_no"): r for r in (doc.get("rounds") or [])}
    except Exception:  # noqa: BLE001
        rounds_meta = {}

    for state, events in rounds:
        rno = getattr(state, "round_no", None)
        rkey = (room_id, rno)
        if rkey in seen_rounds:
            stats["dup_round"] += 1
            continue
        seen_rounds.add(rkey)

        start_god = {s: state.seats[s].hand[GOD_ID] for s in range(4)}
        turn_cnt: collections.Counter = collections.Counter()
        memo: dict = {}
        for ev in events:
            seat = ev.get("seat")
            if seat is None:
                continue
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                stats["apply_event"] += 1
                break
            if ev.get("type") in ("tile_discarded", "peng", "chi", "gang", "ming_gang", "an_gang", "bu_gang"):
                turn_cnt[seat] += 1

        end_turn = max(turn_cnt.values()) if turn_cnt else 0
        # winner 从 doc 级 rounds 元数据取（事件流无 hu/win 事件；17:34 核实）
        rmeta = rounds_meta.get(rno) or {}
        winner = rmeta.get("winner")  # 座位索引；is_draw=1 时无 winner
        # 全座·局口径：每座位都记录（含胡牌者）
        for s in range(4):
            if s != my_seat and s not in bot_seats:
                continue  # 只留我方与目标 bot
            ss = state.seats[s]
            try:
                sh = shanten_mod.shanten_any(ss.hand, len(ss.melds), memo=memo)
            except Exception:  # noqa: BLE001
                stats["shanten_err"] += 1
                continue
            recs.append({
                "seat": s,
                "is_bot": s in bot_seats,
                "is_me": s == my_seat,
                "is_winner": (winner == s),
                "is_draw": bool(rmeta.get("is_draw")),
                "tenpai": sh <= 0,  # 局末是否达听（0 或更小=听/胡）
                "god": start_god[s],
                "meld": len(ss.melds),
                "end_turn": end_turn,
            })


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    print(f"事件文档 {len(files)} 个", flush=True)
    targets = build_targets(files)
    print(f"目标 bot: {len(targets)}/{len(EXP.TOP_BOTS)}", flush=True)

    stats = collections.Counter()
    recs: list[dict] = []
    seen_rounds: set = set()
    for f in files:
        process_room(f, targets, stats, recs, seen_rounds)

    print(f"去重局数 = {len(seen_rounds)}（dup {stats['dup_round']}）", flush=True)
    print(f"全座·局记录 = {len(recs)}", flush=True)

    my = [r for r in recs if r["is_me"]]
    bot = [r for r in recs if r["is_bot"]]
    print(f"我方 {len(my)} 局 / bot {len(bot)} 座位局", flush=True)

    def rate(rows):
        return sum(r["tenpai"] for r in rows) / len(rows) if rows else 0.0

    out = []
    out.append(f"局末达听比例·无条件口径（{len(seen_rounds)} 去重局；我方 {len(my)} 局 / bot {len(bot)} 座位局）")
    out.append("「达听」= 局末 shanten≤0（听牌或已胡都算达到）")
    out.append("")
    out.append("== 总览（全座·局口径，不筛未胡者）==")
    rm, rb = rate(my), rate(bot)
    out.append(f"  bot 局末达听率 = {rb:.1%}")
    out.append(f"  我方 局末达听率 = {rm:.1%}")
    out.append(f"  差（我−bot）= {rm-rb:+.1%}")
    out.append(f"  （对照：17:09 未胡者口径的 0 向听占比差 = −14.0pp；判据：无条件口径也低 ~14pp ⇒ 分支① 稳）")

    out.append("")
    out.append("== 分桶（起手财神 × 终局副露 × 终局巡目段）==")
    out.append("分桶 | bot n | bot 达听率 | 我方 n | 我方达听率 | 差(我−bot)")
    out.append("-----|-------|-----------|--------|-----------|----------")
    buckets: dict[tuple, dict] = collections.defaultdict(lambda: {"bot": [], "me": []})
    for r in recs:
        key = (god_bucket(r["god"]), meld_bucket(r["meld"]), turn_seg(r["end_turn"]))
        if r["is_bot"]:
            buckets[key]["bot"].append(r)
        elif r["is_me"]:
            buckets[key]["me"].append(r)
    for key in sorted(buckets.keys()):
        v = buckets[key]
        if not v["bot"] and not v["me"]:
            continue
        rbx = rate(v["bot"]); rmx = rate(v["me"])
        out.append(f"{key[0]:>6} {key[1]:>6} {key[2]:>7} | {len(v['bot']):>5} | {rbx:>9.1%} | {len(v['me']):>6} | {rmx:>9.1%} | {rmx-rbx:>+8.1%}")

    text = "\n".join(out)
    print(text)
    (ROOT / "agent/out/final-tenpai-rate.txt").write_text(text, encoding="utf-8")
    print(f"\n-> agent/out/final-tenpai-rate.txt", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
