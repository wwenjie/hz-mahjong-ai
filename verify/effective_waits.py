#!/usr/bin/env python3
"""有效听口测量：首次听牌时的「理论进张 vs 有效进张（扣已见牌）」对比，我们 vs 对手。

动机（A 的 23:25 问题 + 牌理「死张换活张」）：我们听口窄 13.6% 是理论值；
若扣掉已见张后差距更大，说明我们不仅窄、还常听在死张上——「换听」杠杆就有实锤。

口径：
- 首次听牌时刻 = 该座位出牌后手牌首次 shanten==0（出牌后口径，与 measure_strength 一致）
- 理论进张 = winning_draws 的剩余张数（按全场 4 张计，不扣任何可见）
- 有效进张 = 扣掉：自己手牌、桌面弃牌、全场副露（该座位当时可知的全部信息）
- 分组：我们 vs 对手；另报「死听率」（有效进张 ≤2 的占比）

用法：nice -n 15 uv run python verify/effective_waits.py
"""
import collections
import glob
import json

import numpy as np

from majiang.rules import tiles, win
from majiang.rules.shanten import shanten_any

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


def scan_round(our_seat, rnd, memo, out):
    if rnd["hands"] is None:
        return
    hand = [collections.Counter(h) for h in rnd["hands"]]
    meld_tiles = [collections.Counter() for _ in range(4)]  # 各座位副露里的牌（公开）
    meld_sets = [0, 0, 0, 0]
    discards = collections.Counter()
    first_tenpai = [False] * 4  # 是否已记录首次听牌

    def check_tenpai(s):
        if first_tenpai[s]:
            return
        counts = [0] * tiles.TILE_KINDS
        for code, c in hand[s].items():
            counts[tiles.parse(code)] = c
        try:
            if shanten_any(counts, meld_sets[s], memo=memo) != 0:
                return
            waits = win.winning_draws(counts, meld_sets[s])
        except Exception:
            return
        if not waits:
            return
        first_tenpai[s] = True
        theo = 0
        eff = 0
        for w in waits:
            total_left_theo = 4 - counts[w]  # 理论：只扣自己手牌
            visible = (counts[w] + discards[w]
                       + sum(meld_tiles[o][w] for o in range(4)))
            theo += max(0, total_left_theo)
            eff += max(0, 4 - visible)
        who = "us" if s == our_seat else "opp"
        out[(who, "theo")].append(theo)
        out[(who, "eff")].append(eff)
        out[(who, "dead")].append(1 if eff <= 2 else 0)

    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if s not in (0, 1, 2, 3):
            continue
        if et == "tile_drawn":
            hand[s][tile] += 1
        elif et == "tile_discarded":
            hand[s][tile] -= 1
            discards[tiles.parse(tile)] += 1  # 索引空间（winning_draws 返回索引）
            check_tenpai(s)
        elif et == "chi":
            for t in data.get("tiles") or []:
                meld_tiles[s][tiles.parse(t)] += 1
            need = collections.Counter(data.get("tiles") or [])
            need[tile] -= 1
            if need[tile] == 0:
                del need[tile]
            for t, c in need.items():
                hand[s][t] -= c
            meld_sets[s] += 1
        elif et == "peng":
            meld_tiles[s][tiles.parse(tile)] += 3
            hand[s][tile] -= 2
            meld_sets[s] += 1
        elif et == "gang":
            kind = data.get("kind")
            meld_tiles[s][tiles.parse(tile)] += 4
            hand[s][tile] -= {"an": 4, "bu": 1, "ming": 3}.get(kind, 0)
            if kind != "bu":
                meld_sets[s] += 1


def main():
    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
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
        our = uids.index(OUR_UID)
        n += 1
        for rnd in split_rounds(doc):
            scan_round(our, rnd, memo, out)

    print(f"场次={n}")
    for who, label in (("us", "我们"), ("opp", "对手")):
        theo = np.array(out[(who, "theo")])
        eff = np.array(out[(who, "eff")])
        dead = np.array(out[(who, "dead")])
        print(f"[{label}] 首次听牌样本={len(theo)} 理论进张={theo.mean():.2f} "
              f"有效进张={eff.mean():.2f} 有效率={eff.mean()/max(theo.mean(),1e-9):.0%} "
              f"死听率(≤2)={dead.mean():.1%}")
    if out[("us", "eff")] and out[("opp", "eff")]:
        d_eff = np.mean(out[("opp", "eff")]) - np.mean(out[("us", "eff")])
        d_theo = np.mean(out[("opp", "theo")]) - np.mean(out[("us", "theo")])
        print(f"\n差距: 理论 {d_theo:+.2f} 张  有效 {d_eff:+.2f} 张"
              f"  → {'有效差距更大：我们听死在已见张上，换听杠杆成立' if d_eff > d_theo * 1.15 else '差距主要来自理论窄'}")


if __name__ == "__main__":
    main()
