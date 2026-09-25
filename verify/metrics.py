#!/usr/bin/env python3
"""指标复算（独立实现，纯标准库，不 import 任何项目代码）。

输入：notes/manifest-20260926.txt 冻结清单（默认）或全量 data/auto_sessions。
口径（全部来自事件流 round_ended / 事件计数，不用顶层 rounds[] 摘要——
摘要的流局收录是有损的，见 verify/invariants.py 的 summary_rule 统计）：
  - 分母一律用 finished 文件的 8 局/场；abandoned 房整房剔除
  - 有胡率 = 非流局数 / 总局数；公平份额 = 有胡率 / 4
  - 名次：每场按四家总分排序（总分=全场 8 局 scores 累加，流局 0 分不影响），
    同分并列取名次分均值；另报「完整房」（房内场次全部 finished）的房级名次
  - 均番 = 非流局 round_ended.data.fan 按赢家归属分组均值
  - 副露 = chi+peng+gang 事件计数，按 seat 归属，除以该方参与局数
  - 胡牌时已出牌 = 非流局中赢家在该局的 tile_discarded 次数

用法：uv run python verify/metrics.py [--manifest notes/manifest-20260926.txt]
"""
import argparse
import collections
import glob
import json
import os
import sys

OUR_UID = "u_a7f7c67bb14a"


def split_rounds(doc):
    rounds = []
    cur = None
    for b in doc["blocks"]:
        rn = b.get("round_no")
        if cur is None or rn != cur["round_no"]:
            cur = {"round_no": rn, "events": []}
            rounds.append(cur)
        cur["events"].extend(b.get("events") or [])
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="notes/manifest-20260926.txt")
    args = ap.parse_args()

    if args.manifest and os.path.exists(args.manifest):
        files = [l.strip() for l in open(args.manifest, encoding="utf-8")
                 if l.strip() and not l.startswith("#")]
    else:
        files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))

    # 聚合器
    rounds_total = 0          # finished 文件的总局数
    rounds_decisive = 0
    wins = collections.Counter()          # 'us' / 'opp'
    fan_sum = collections.Counter()
    fan_n = collections.Counter()
    meld_count = collections.Counter()    # 副露次数
    meld_rounds = collections.Counter()   # 参与局数（分母）
    discards_at_win = collections.defaultdict(list)  # 'us'/'opp' -> [胡时已出次数]
    match_placements = collections.Counter()  # 我们的名次 1..4（每场）
    match_pts = 0.0
    n_matches = 0
    room_scores = collections.defaultdict(lambda: collections.Counter())  # room -> uid -> score
    room_files = collections.defaultdict(lambda: [0, 0])  # room -> [finished, total]
    seats_missing_us = 0
    files_skipped = 0

    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        room = doc.get("room_id") or p.split("/")[-3]
        status = doc.get("status")
        room_files[room][1] += 1
        if status != "finished":
            files_skipped += 1
            continue
        room_files[room][0] += 1

        seats = doc.get("seats") or []
        our_seat = next((i for i, s in enumerate(seats)
                         if s.get("user_id") == OUR_UID), None)
        if our_seat is None:
            seats_missing_us += 1
            files_skipped += 1
            continue

        match_score = collections.Counter()
        for rnd in split_rounds(doc):
            ended = [e for e in rnd["events"] if e["type"] == "round_ended"]
            if len(ended) != 1:
                continue  # finished 文件不应出现；有则跳过并在末尾报告
            e = ended[0]
            d = e["data"] or {}
            rounds_total += 1
            meld_rounds["us"] += 1
            meld_rounds["opp"] += 3

            melds = collections.Counter()
            discards = collections.Counter()
            for ev in rnd["events"]:
                t = ev["type"]
                s = ev.get("seat")
                if s not in (0, 1, 2, 3):
                    continue
                if t in ("chi", "peng", "gang"):
                    melds[s] += 1
                elif t == "tile_discarded":
                    discards[s] += 1
            for s in range(4):
                meld_count["us" if s == our_seat else "opp"] += melds[s]

            scores = d.get("scores") or [0, 0, 0, 0]
            for s in range(4):
                match_score[s] += scores[s]
                uid = seats[s].get("user_id") if s < len(seats) else f"seat{s}"
                room_scores[room][uid] += scores[s]

            if d.get("draw"):
                continue
            rounds_decisive += 1
            w = e.get("seat")
            who = "us" if w == our_seat else "opp"
            wins[who] += 1
            fan = d.get("fan")
            if isinstance(fan, (int, float)):
                fan_sum[who] += fan
                fan_n[who] += 1
            discards_at_win[who].append(discards[w])

        # 场次名次（总分排序，同分并列取均值名次分）
        n_matches += 1
        order = sorted(range(4), key=lambda s: -match_score[s])
        rank_of = {}
        i = 0
        pts = [3.0, 1.0, -1.0, -3.0]
        while i < 4:
            j = i
            while j + 1 < 4 and match_score[order[j + 1]] == match_score[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            avg_pts = sum(pts[i:j + 1]) / (j - i + 1)
            for k in range(i, j + 1):
                rank_of[order[k]] = (avg_rank, avg_pts)
            i = j + 1
        r, pt = rank_of[our_seat]
        match_placements[r] += 1
        match_pts += pt

    # 房级名次（只算房内场次全 finished 的完整房）
    complete_rooms = {r for r, (f, t) in room_files.items() if f == t}
    room_placements = collections.Counter()
    room_pts = 0.0
    for room in sorted(complete_rooms):
        sc = room_scores[room]
        if OUR_UID not in sc or len(sc) < 4:
            continue
        uids = sorted(sc, key=lambda u: -sc[u])
        rank_of = {}
        i = 0
        pts = [3.0, 1.0, -1.0, -3.0]
        while i < len(uids):
            j = i
            while j + 1 < len(uids) and sc[uids[j + 1]] == sc[uids[i]]:
                j += 1
            for k in range(i, j + 1):
                rank_of[uids[k]] = ((i + j) / 2 + 1, sum(pts[i:j + 1]) / (j - i + 1))
            i = j + 1
        r, pt = rank_of[OUR_UID]
        room_placements[r] += 1
        room_pts += pt

    n_opp_rounds = meld_rounds["opp"]
    out = {
        "snapshot": {"manifest": args.manifest, "files": len(files),
                     "files_skipped_not_finished_or_no_us": files_skipped},
        "rounds": {"total": rounds_total, "decisive": rounds_decisive,
                   "draws": rounds_total - rounds_decisive},
        "us": {
            "hands": rounds_total, "wins": wins["us"],
            "win_rate": wins["us"] / rounds_total if rounds_total else 0,
            "win_rate_decisive_only": wins["us"] / rounds_decisive if rounds_decisive else 0,
            "fan_avg": fan_sum["us"] / fan_n["us"] if fan_n["us"] else 0,
            "melds_per_round": meld_count["us"] / meld_rounds["us"] if meld_rounds["us"] else 0,
            "discards_at_win_avg": (sum(discards_at_win["us"]) / len(discards_at_win["us"])
                                    if discards_at_win["us"] else 0),
        },
        "opp": {
            "hands": n_opp_rounds, "wins": wins["opp"],
            "win_rate": wins["opp"] / n_opp_rounds if n_opp_rounds else 0,
            "win_rate_decisive_only": wins["opp"] / rounds_decisive / 3 if rounds_decisive else 0,
            "fan_avg": fan_sum["opp"] / fan_n["opp"] if fan_n["opp"] else 0,
            "melds_per_round": meld_count["opp"] / n_opp_rounds if n_opp_rounds else 0,
            "discards_at_win_avg": (sum(discards_at_win["opp"]) / len(discards_at_win["opp"])
                                    if discards_at_win["opp"] else 0),
        },
        "has_hu_rate": rounds_decisive / rounds_total if rounds_total else 0,
        "fair_share": rounds_decisive / rounds_total / 4 if rounds_total else 0,
        "placement_per_match": {
            "n": n_matches,
            "dist": {str(k): match_placements[k] for k in sorted(match_placements)},
            "pts_avg": match_pts / n_matches if n_matches else 0,
        },
        "placement_per_complete_room": {
            "rooms": len(complete_rooms),
            "dist": {str(k): room_placements[k] for k in sorted(room_placements)},
            "pts_avg": (room_pts / sum(room_placements.values())
                        if sum(room_placements.values()) else 0),
        },
    }
    os.makedirs("verify/out", exist_ok=True)
    with open("verify/out/metrics-latest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
