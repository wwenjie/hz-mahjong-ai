#!/usr/bin/env python3
"""独立复核「真机噪声基底」（A 的参考值：按房 bootstrap 胜率差 SD 1.50%，
是二项预期 0.61% 的 6.0 倍；检出 1.25pp 需 ~350 小时）。不看 A 的实现。

我的口径（从第一性原理）：
- 估计量：我们的总胜率 p̂ = 胜局数 / 总局数
- 朴素基准：二项 SD = sqrt(p(1-p)/N)（假设局间独立）
- 聚类基准：按**房**做 cluster bootstrap（同一房的四家固定 → 局间不独立），
  B=10000 次重抽样，取 p̂ 的经验 SD
- 方差膨胀倍数 = 聚类 SD / 二项 SD
- 两臂 A/B：每次 replicate 把房随机对半分，两臂各自算 p̂ 取差 → 胜率差 SD
- 检出时长：SD_diff(n) = SD_diff(1房对) × sqrt(总房数/n)；
  每场次耗时取事件中位时长，串行跑房换算小时数

用法：uv run python verify/noise_floor.py [--boot 10000]
"""
import argparse
import collections
import json
import math
import random

OUR_UID = "u_a7f7c67bb14a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]

    # 每房：我们的胜局数、总局数、场次耗时
    rooms = collections.defaultdict(lambda: {"wins": 0, "rounds": 0, "durations": []})
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        seats = doc.get("seats") or []
        our = next((i for i, s in enumerate(seats) if s.get("user_id") == OUR_UID), None)
        if our is None:
            continue
        room = doc.get("room_id")
        ts_all = []
        cur = None
        for b in doc["blocks"]:
            if b.get("round_no") != cur:
                cur = b.get("round_no")
                rooms[room]["rounds"] += 1
            for e in b.get("events") or []:
                ts = e.get("ts")
                if isinstance(ts, (int, float)):
                    ts_all.append(ts)
                if e.get("type") == "round_ended" and not (e.get("data") or {}).get("draw") \
                        and e.get("seat") == our:
                    rooms[room]["wins"] += 1
        if ts_all:
            rooms[room]["durations"].append(max(ts_all) - min(ts_all))

    rooms = {r: v for r, v in rooms.items() if v["rounds"] > 0}
    room_names = sorted(rooms)
    total_wins = sum(v["wins"] for v in rooms.values())
    total_rounds = sum(v["rounds"] for v in rooms.values())
    p = total_wins / total_rounds
    n_rooms = len(room_names)

    binom_sd = math.sqrt(p * (1 - p) / total_rounds)

    rng = random.Random(args.seed)
    # 单臂：按房重抽
    single = []
    weights = [(v["wins"], v["rounds"]) for v in (rooms[r] for r in room_names)]
    for _ in range(args.boot):
        w = r_ = 0
        for _ in range(n_rooms):
            wi, ri = weights[rng.randrange(n_rooms)]
            w += wi
            r_ += ri
        single.append(w / r_)
    sd_single = _sd(single)

    # 双臂：房内对半分，取臂间胜率差（同一份数据，理论零差 → 纯噪声）
    half = n_rooms // 2
    diffs = []
    for _ in range(args.boot):
        idx = list(range(n_rooms))
        rng.shuffle(idx)
        w1 = r1 = w2 = r2 = 0
        for k, i in enumerate(idx):
            wi, ri = weights[i]
            if k < half:
                w1 += wi
                r1 += ri
            else:
                w2 += wi
                r2 += ri
        diffs.append(w1 / r1 - w2 / r2)
    sd_diff = _sd(diffs)

    # 场次耗时 → 检出时长（串行假设）
    durations = sorted(d for v in rooms.values() for d in v["durations"] if d > 0)
    med_dur = durations[len(durations) // 2] if durations else 0

    # 检出 1.25pp 需要的「每场对」数量与小时（两侧各 n_matches 场）：
    # SD_diff(随机场配对) ≈ sd_diff × sqrt(n_rooms / n_matches_per_arm...) ——
    # 直接按 SD 随 1/sqrt(房数) 缩放：sd_diff_at(m) = sd_diff * sqrt(half / m)，m=每臂房数
    target = 0.0125
    z = 1.96
    # 需求：z * sd_diff * sqrt(half / m) = target  →  m = half * (z*sd_diff/target)^2
    m_needed = half * (z * sd_diff / target) ** 2
    matches_per_room = total_rounds / n_rooms / 8
    hours = (2 * m_needed) * med_dur / 3600 if med_dur else 0

    print(f"房数={n_rooms} 总局数={total_rounds} 我们胜率={p:.4f}")
    print(f"场次中位时长={med_dur/60:.1f} 分钟  每场≈8局")
    print(f"二项 SD(p̂)          = {binom_sd:.4%}")
    print(f"按房 bootstrap SD(p̂) = {sd_single:.4%}  (B={args.boot})")
    print(f"方差膨胀倍数         = {sd_single/binom_sd:.2f}x")
    print(f"两臂差 SD(对半分房)  = {sd_diff:.4%}")
    print(f"检出 1.25pp 需每臂 ≈{m_needed:.0f} 房（全场≈{2*m_needed:.0f} 房）"
          f"≈串行 {hours:.0f} 小时（按中位时长，未计并发）")


def _sd(xs):
    n = len(xs)
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))


if __name__ == "__main__":
    main()
