#!/usr/bin/env python3
"""副露/响应决策的同质性检验（A 的 19:30 请求：比候选级克隆值钱一个量级）。

问题：我们副露 0.591/局 vs 对手 1.093/局。对手间是否同质？
  - 若同质（大家都在 ~1.09 附近）→ 差距是「我们 vs 环境」的系统性差异，改吃碰闸门
  - 若不同质（强弱对手分化）→ 差异集中在响应决策的类型/风格，另行拆解

方法（纯事件计数，不重建手牌——响应窗口在事件流里是显式的）：
  - peng 接受率 = peng 事件 / (peng 事件 + timeout[window=peng] + 显式 pass*)
  - chi 接受率  = chi 事件 / (chi 事件 + timeout[window=chi] + 显式 pass*)
  - 明杠接受率同理（window=gang）
  * pass 事件 data 为空，归属规则：出牌事件后、下一次摸牌前的 pass 记为对该出牌的
    响应放弃；碰/吃窗口对同座可能同时开，pass 只计一次——在解读时注明此口径。
  - 另报「窗口次数 vs 出牌次数」的覆盖率 sanity check。

分层：按 opponent user_id（跨房同一人可合并）与按房强度（对手胜率中位分层）。
用法：nice -n 15 uv run python verify/response_homogeneity.py
"""
import collections
import glob
import json

import numpy as np

OUR_UID = "u_a7f7c67bb14a"


def main():
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    # per user_id: accepts/declines per window kind
    acc = collections.defaultdict(collections.Counter)   # uid -> Counter
    room_of = {}
    seat_uid_room = collections.defaultdict(collections.Counter)  # (room,seat) -> Counter
    rooms_wins = collections.Counter()
    rooms_rounds = collections.Counter()
    windows_seen = collections.Counter()
    discards_seen = 0

    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        seats = doc.get("seats") or []
        uids = [s.get("user_id") for s in seats]
        our = next((i for i, u in enumerate(uids) if u == OUR_UID), None)
        if our is None:
            continue
        room = doc.get("room_id")
        for b in doc["blocks"]:
            evs = b.get("events") or []
            for i, e in enumerate(evs):
                et = e["type"]
                s = e.get("seat")
                if et == "tile_discarded":
                    discards_seen += 1
                elif et in ("peng", "chi"):
                    if s not in (0, 1, 2, 3) or s == our:
                        continue
                    acc[uids[s]][f"{et}_accept"] += 1
                    seat_uid_room[(room, s)][f"{et}_accept"] += 1
                elif et == "gang":
                    if s in (0, 1, 2, 3) and s != our and (e.get("data") or {}).get("kind") == "ming":
                        acc[uids[s]]["gang_accept"] += 1
                        seat_uid_room[(room, s)]["gang_accept"] += 1
                elif et == "timeout":
                    d = e.get("data") or {}
                    if d.get("kind") != "response" or s not in (0, 1, 2, 3) or s == our:
                        continue
                    w = d.get("window")
                    windows_seen[w] += 1
                    acc[uids[s]][f"{w}_decline"] += 1
                    seat_uid_room[(room, s)][f"{w}_decline"] += 1
                elif et == "round_ended":
                    d = e.get("data") or {}
                    if d.get("draw"):
                        continue
                    for s2 in range(4):
                        if s2 != our:
                            rooms_rounds[room] += 1
                            if e.get("seat") == s2:
                                rooms_wins[room] += 1

    print(f"窗口计数: {dict(windows_seen)}  出牌总数={discards_seen}")
    print(f"对手身份数={len(acc)}  (房×座位记录={len(seat_uid_room)})")

    def rate(counter, kind):
        a, d = counter[f"{kind}_accept"], counter[f"{kind}_decline"]
        return a / (a + d) if a + d else None

    def dist(name, rows):
        vals = [r for r in rows if r is not None]
        if not vals:
            print(f"{name}: 无样本")
            return
        arr = np.array(vals)
        print(f"{name}: n={len(arr)} 均值={arr.mean():.1%} 中位={np.median(arr):.1%} "
              f"std={arr.std():.1%} IQR=[{np.percentile(arr,25):.1%}, {np.percentile(arr,75):.1%}]")

    # 按身份（跨房合并）
    for kind in ("peng", "chi"):
        rows = [rate(c, kind) for c in acc.values()
                if c[f"{kind}_accept"] + c[f"{kind}_decline"] >= 10]
        dist(f"按身份 {kind} 接受率(≥10窗口)", rows)

    # 按房×座位
    for kind in ("peng", "chi"):
        rows = [rate(c, kind) for c in seat_uid_room.values()
                if c[f"{kind}_accept"] + c[f"{kind}_decline"] >= 5]
        dist(f"按房×座位 {kind} 接受率(≥5窗口)", rows)

    # 强度分层（对手胜率）
    med = np.median([rooms_wins[r] / rooms_rounds[r] for r in rooms_rounds if rooms_rounds[r] > 0])
    tier = {r: ("strong" if rooms_wins[r] / rooms_rounds[r] >= med else "weak")
            for r in rooms_rounds if rooms_rounds[r] > 0}
    for kind in ("peng", "chi"):
        for t in ("strong", "weak"):
            rows = [rate(c, kind) for (room, s), c in seat_uid_room.items()
                    if tier.get(room) == t and c[f"{kind}_accept"] + c[f"{kind}_decline"] >= 5]
            dist(f"{t}房 {kind} 接受率", rows)


if __name__ == "__main__":
    main()
