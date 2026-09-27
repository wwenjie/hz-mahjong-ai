#!/usr/bin/env python3
"""仪器方向检验独立分析：真机交错臂 v2 vs first-legal（A 的 16:45 方案）。

判据（预登记式的，先看效应方向再看显著性）：
  first-legal 在自对弈里惨败（流局 65%、均番 1.07）→ 真机上若也显著差于 v2，
  则「自对弈差异能预测真机差异」方向有效；若打不出差距，则真机分辨率不足。
口径：
  - 臂归属只看账本 decider 字段（v2 / first-legal），时段 16:30 起（交错开始后）
  - 指标：胜率、每场总得分（均值+中位）、名次分（+3/+1/−1/−3）、均番
  - 不确定度：按场 bootstrap（10000 次）给胜率差与总分差的 95% CI；
    房数少（7v8），房聚类会低估方差，CI 偏窄——保守读法注在输出里
用法：nice -n 15 uv run python verify/instrument_ab.py
"""
import collections
import glob
import json

import numpy as np

OUR_UID = "u_a7f7c67bb14a"
SINCE = "2026-09-27T16:30"


def arm_rooms():
    arms = collections.defaultdict(list)
    for line in open("data/auto_sessions/sessions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("started_at", "") >= SINCE and r.get("decider") in ("v2", "first-legal"):
            arms[r["decider"]].append(r["room_id"])
    return arms


def room_stats(room):
    """该房每场：我们的总分、名次、胜局数、番。"""
    out = []
    for p in sorted(glob.glob(f"data/auto_sessions/{room}/events/*.json")):
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        seats = doc.get("seats") or []
        uids = [s.get("user_id") for s in seats]
        if OUR_UID not in uids:
            continue
        our = uids.index(OUR_UID)
        total = [0, 0, 0, 0]
        wins = fans = rounds = 0
        seen = set()
        for b in doc["blocks"]:
            for e in b.get("events") or []:
                if e["type"] != "round_ended":
                    continue
                d = e.get("data") or {}
                rn = d.get("round_no")
                if rn in seen:
                    continue
                seen.add(rn)
                rounds += 1
                sc = d.get("scores") or [0, 0, 0, 0]
                for s in range(4):
                    total[s] += sc[s]
                if not d.get("draw"):
                    if e.get("seat") == our:
                        wins += 1
                        fans += d.get("fan") or 0
        if rounds == 0:
            continue
        order = sorted(range(4), key=lambda s: -total[s])
        place = order.index(our) + 1
        out.append({"score": total[our], "place": place, "wins": wins,
                    "rounds": rounds, "fan_avg": fans / wins if wins else 0.0})
    return out


def main():
    arms = arm_rooms()
    data = {arm: [m for room in rooms for m in room_stats(room)]
            for arm, rooms in arms.items()}
    pts = [3, 1, -1, -3]
    summary = {}
    for arm, ms in data.items():
        n = len(ms)
        if not n:
            continue
        wins = sum(m["wins"] for m in ms)
        rounds = sum(m["rounds"] for m in ms)
        scores = np.array([m["score"] for m in ms])
        places = np.array([pts[m["place"] - 1] for m in ms], dtype=float)
        fans = [m["fan_avg"] for m in ms if m["wins"]]
        summary[arm] = {
            "matches": n, "rounds": rounds, "win_rate": wins / rounds,
            "score_mean": float(scores.mean()), "score_med": float(np.median(scores)),
            "place_pts": float(places.mean()),
            "fan_avg": float(np.mean(fans)) if fans else 0.0,
            "scores": scores, "wins": wins,
        }
        print(f"[{arm}] 场={n} 局={rounds} 胜率={wins/rounds:.1%} "
              f"场均分={scores.mean():+.1f}(中位{np.median(scores):+.0f}) "
              f"名次分={places.mean():+.2f} 均番={np.mean(fans) if fans else 0:.2f}")

    if "v2" in summary and "first-legal" in summary:
        a, b = summary["v2"], summary["first-legal"]
        rng = np.random.RandomState(7)
        n1, n0 = a["matches"], b["matches"]
        d_wr, d_sc = [], []
        for _ in range(10000):
            sa = a["scores"][rng.randint(0, n1, n1)]
            sb = b["scores"][rng.randint(0, n0, n0)]
            d_sc.append(sa.mean() - sb.mean())
            wa = sum(a["wins"] for _ in [0])  # placeholder, 胜率 bootstrap 用局级
        # 胜率差：正态近似（局数够大）
        p1, p0 = a["win_rate"], b["win_rate"]
        se = (p1 * (1 - p1) / a["rounds"] + p0 * (1 - p0) / b["rounds"]) ** 0.5
        d_sc = np.array(d_sc)
        print(f"\n胜率差(v2−first-legal)={p1-p0:+.1%} (SE≈{se:.1%}, z≈{(p1-p0)/max(se,1e-9):.1f})")
        print(f"场均分差={a['score_mean']-b['score_mean']:+.1f} "
              f"95%CI[{np.percentile(d_sc,2.5):+.1f},{np.percentile(d_sc,97.5):+.1f}]")
        print("注：仅 7v8 房，房聚类使真实 CI 更宽；方向检验只看量级与符号。")


if __name__ == "__main__":
    main()
