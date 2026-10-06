#!/usr/bin/env python
"""终局向听分布（A 2026-10-06 16:59④ [待C] 交办）。

**量**：「没到听的那些局」里我们卡在哪——**终局时刻**（他人胡牌或流局那一刻）我方的向听分布
（0/1/2/3+ 占比），与 bot 同口径对照；分桶 `起手财神数 × 终局副露数 × 终局巡目段`。

**预登记决策树**（A 16:59④，出数后按此执行，不重议口径）：
| 结果 | 判读 | 动作 |
|---|---|---|
| 我们终局向听集中在 1（差一步），bot 在 0 | 瓶颈在最后一步 | 量「自摸口径下的最后一步」（听口可达性） |
| 我们终局向听集中在 2~3 | 瓶颈在中段形质积累 | 回到留牌/形质（财神留手、形质项） |
| 与他人胡的时机相关（bot 更早在别人胡前到听） | 速度×时机 | 按剩余巡数分层的巡目对照 |

**口径细节（C 自拟，报出供裁决）**：
- 「终局」= 每局最后一个事件处理完后的状态（无论谁胡或流局）；
- 「终局巡目段」= 终局时的出手序数分段：早（≤5）/ 中（6-10）/ 晚（≥11）；
- 终局向听用 `shanten_any`（兼容 13/14 张，免张数不符抛错）；
- 胡牌者本人终局向听必为 -1/0（已胡），仍计入分布（作为参照）；分析重点在「未胡者」。
- 同一 (room_id, round_no) 跨文档去重（沿用到听速度探针的去重键）。

产物：`agent/out/final-shanten-dist.txt`。
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


def god_bucket(n: int) -> str:
    return "god=0" if n == 0 else ("god=1" if n == 1 else "god≥2")


def meld_bucket(m: int) -> str:
    return "meld=0" if m == 0 else "meld≥1"


def turn_seg(t: int) -> str:
    return "早(≤5)" if t <= 5 else ("中(6-10)" if t <= 10 else "晚(≥11)")


def sh_bucket(sh: int) -> str:
    if sh <= 0:
        return "0(听/胡)"
    if sh == 1:
        return "1"
    if sh == 2:
        return "2"
    return "3+"


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
        last_ev = None
        for ev in events:
            seat = ev.get("seat")
            if seat is None:
                continue
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                stats["apply_event"] += 1
                break
            last_ev = ev
            if ev.get("type") in ("tile_discarded", "peng", "chi", "gang", "ming_gang", "an_gang", "bu_gang"):
                turn_cnt[seat] += 1

        # 终局：记录每座位终局向听
        winner = None
        if last_ev is not None and last_ev.get("type") in ("hu", "win", "hu_self", "hu_discard"):
            winner = last_ev.get("seat")
        end_turn = max(turn_cnt.values()) if turn_cnt else 0
        for s in range(4):
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
                "is_winner": winner == s,
                "sh": sh,
                "god": start_god[s],
                "meld": len(ss.melds),
                "turn": turn_cnt[s],
                "end_turn": end_turn,
                "room": room_id,
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
    print(f"终局向听记录 = {len(recs)} 座位局", flush=True)

    # 只分析「未胡者」的终局向听分布
    non_win = [r for r in recs if not r["is_winner"]]
    my_nw = [r for r in non_win if r["is_me"]]
    bot_nw = [r for r in non_win if r["is_bot"]]
    print(f"未胡者：我方 {len(my_nw)} 座位局 / bot {len(bot_nw)} 座位局", flush=True)

    # 总分布（未胡者，我方 vs bot）
    def dist(rows):
        c = collections.Counter(sh_bucket(r["sh"]) for r in rows)
        n = sum(c.values())
        return {k: c.get(k, 0) / n for k in ("0(听/胡)", "1", "2", "3+")}, n

    out = []
    out.append(f"终局向听分布（{len(seen_rounds)} 去重局；未胡者：我方 {len(my_nw)} / bot {len(bot_nw)} 座位局）")
    out.append("")
    # 总览
    dm, nm = dist(my_nw)
    db, nb = dist(bot_nw)
    out.append("== 总览（未胡者终局向听分布）==")
    out.append("向听 | bot 占比 | 我方占比 | 差(我−bot)")
    out.append("-----|---------|---------|----------")
    for k in ("0(听/胡)", "1", "2", "3+"):
        out.append(f"{k:>4} | {db[k]:>7.1%} | {dm[k]:>7.1%} | {dm[k]-db[k]:>+8.1%}")

    # 分桶（财神×副露×巡目段）
    out.append("")
    out.append("== 分桶（起手财神 × 终局副露 × 终局巡目段；未胡者）==")
    out.append("分桶 | bot n | bot 分布 0/1/2/3+ | 我方 n | 我方分布 0/1/2/3+ | 关键差")
    out.append("-----|-------|-------------------|--------|-------------------|-------")
    buckets: dict[tuple, dict] = collections.defaultdict(lambda: {"bot": [], "me": []})
    for r in non_win:
        key = (god_bucket(r["god"]), meld_bucket(r["meld"]), turn_seg(r["end_turn"]))
        if r["is_bot"]:
            buckets[key]["bot"].append(r)
        elif r["is_me"]:
            buckets[key]["me"].append(r)
    for key in sorted(buckets.keys()):
        v = buckets[key]
        if not v["bot"] and not v["me"]:
            continue
        dbx, nbx = dist(v["bot"])
        dmx, nmx = dist(v["me"])
        fb = "/".join(f"{dbx[k]:.0%}" for k in ("0(听/胡)", "1", "2", "3+"))
        fm = "/".join(f"{dmx[k]:.0%}" for k in ("0(听/胡)", "1", "2", "3+"))
        # 关键差 = 我方 1 向听占比 − bot 1 向听占比
        key_diff = dmx["1"] - dbx["1"]
        out.append(f"{key[0]:>6} {key[1]:>6} {key[2]:>7} | {nbx:>5} | {fb:>17} | {nmx:>6} | {fm:>17} | Δ1向听 {key_diff:+.1%}")

    text = "\n".join(out)
    print(text)
    (ROOT / "agent/out/final-shanten-dist.txt").write_text(text, encoding="utf-8")
    print(f"\n-> agent/out/final-shanten-dist.txt", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
