#!/usr/bin/env python3
"""首次进入 1 向听时刻的进张数（ukeire）对比：区分「打得差」vs「牌形到得差」。

若我们进入 1 向听时进张已窄于对手 → 病灶在上游（shanten-2 的构形），ukeire-early 对症；
若进张相同但转化慢 → 病灶在 1 向听处的决策，另有其因。
配套：进入 2 向听时刻的同口径测量作对照。

用法：nice -n 15 uv run python verify/arrival_shape.py
"""
import collections
import json

import numpy as np

from majiang.rules import tiles
from majiang.rules.shanten import shanten_any, ukeire

OUR_UID = "u_a7f7c67bb14a"


def split_rounds(doc):
    rounds, cur = [], None
    for b in doc["blocks"]:
        rn = b.get("round_no")
        if cur is None or rn != cur["round_no"]:
            cur = {"round_no": rn, "hands": None, "events": []}
            rounds.append(cur)
            sh = b.get("start_hands")
            if sh and sh[0] is not None:
                cur["hands"] = sh
        cur["events"].extend(b.get("events") or [])
    return rounds


def scan(our, rnd, memo, out):
    if rnd["hands"] is None:
        return
    hand = [collections.Counter(h) for h in rnd["hands"]]
    melds = [0, 0, 0, 0]
    arrived = [set() for _ in range(4)]  # 每座位已记录的向听等级
    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if s not in (0, 1, 2, 3):
            continue
        if et == "tile_drawn":
            hand[s][tile] += 1
        elif et == "tile_discarded":
            hand[s][tile] -= 1
        elif et == "chi":
            need = collections.Counter(data.get("tiles") or [])
            need[tile] -= 1
            if need[tile] == 0:
                del need[tile]
            for t, c in need.items():
                hand[s][t] -= c
            melds[s] += 1
        elif et == "peng":
            hand[s][tile] -= 2
            melds[s] += 1
        elif et == "gang":
            kind = data.get("kind")
            hand[s][tile] -= {"an": 4, "bu": 1, "ming": 3}.get(kind, 0)
            if kind != "bu":
                melds[s] += 1
        else:
            continue
        # 每次手牌变化后（出牌后状态），测向听与进张
        if et not in ("tile_discarded", "chi", "peng", "gang"):
            continue
        counts = [0] * tiles.TILE_KINDS
        for code, c in hand[s].items():
            counts[tiles.parse(code)] = c
        try:
            sh = shanten_any(counts, melds[s], memo=memo)
        except Exception:
            continue
        for level in (1, 2):
            if sh == level and level not in arrived[s]:
                arrived[s].add(level)
                who = "us" if s == our else "opp"
                try:
                    u = len(ukeire(counts, melds[s]))  # 进张牌种数
                except Exception:
                    u = -1
                out[(who, level, "ukeire")].append(u)
                break


def main():
    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    # ukeire 精确版单次 0.1-0.2s，全量太贵 → 按房分层抽样：每房取前 2 场
    by_room = collections.defaultdict(list)
    for p in files:
        by_room[p.split("/")[-3]].append(p)
    files = [p for paths in by_room.values() for p in sorted(paths)[:2]]
    out = collections.defaultdict(list)
    memo: dict = {}
    n = 0
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        uids = [s.get("user_id") for s in doc.get("seats") or []]
        if OUR_UID not in uids:
            continue
        n += 1
        for rnd in split_rounds(doc):
            scan(uids.index(OUR_UID), rnd, memo, out)

    print(f"场次={n}")
    for level in (2, 1):
        for who, label in (("us", "我们"), ("opp", "对手")):
            arr = np.array([v for v in out[(who, level, "ukeire")] if v >= 0])
            print(f"进入 {level} 向听 [{label}] n={len(arr)} 进张牌种: "
                  f"均值={arr.mean():.2f} 中位={np.median(arr):.0f} "
                  f"p25={np.percentile(arr,25):.0f} p75={np.percentile(arr,75):.0f}")
        a = np.array([v for v in out[("us", level, "ukeire")] if v >= 0])
        b = np.array([v for v in out[("opp", level, "ukeire")] if v >= 0])
        print(f"  → 差（对手−我们）: {b.mean()-a.mean():+.2f} 种\n")


if __name__ == "__main__":
    main()
