#!/usr/bin/env python3
"""真值表扩展：0 向听（已听牌）状态按「可见听口张数」分桶的自摸率与均番。

答 A 14:05：现表在「同向听同财神」平面给不出分辨力，缺的轴是听口宽度。
本表只取 shanten==0 的出牌后状态点，按有效听口（winning_draws 扣公开可见张）分桶：
  桶：1-4 / 5-8 / 9-12 / 13-20 / 21+（爆头级宽听）
结局：P(最终自摸)、E[番]。
分层抽样控制 ukeire/winning_draws 成本：每房取前 3 场。

合规边界同 win_rate_table.py：对手手牌重建仅离线，真机决策只用自手公开量。
用法：nice -n 19 uv run python verify/wait_width_table.py
"""
import collections
import glob
import json

import numpy as np

from majiang.rules import tiles, win
from majiang.rules.shanten import shanten_any

OUR_UID = "u_a7f7c67bb14a"
PER_ROOM = 3


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


def width_bucket(w):
    if w <= 4:
        return "1-4"
    if w <= 8:
        return "5-8"
    if w <= 12:
        return "9-12"
    if w <= 20:
        return "13-20"
    return "21+"


def scan(our, rnd, memo, points, key):
    if rnd["hands"] is None:
        return
    hand = [collections.Counter(h) for h in rnd["hands"]]
    meld_tiles = [collections.Counter() for _ in range(4)]
    melds = [0, 0, 0, 0]
    discards = collections.Counter()
    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if s not in (0, 1, 2, 3):
            continue
        if et == "tile_drawn":
            hand[s][tile] += 1
        elif et == "tile_discarded":
            hand[s][tile] -= 1
            discards[tiles.parse(tile)] += 1
            if data.get("catch_play"):
                continue
            counts = [0] * tiles.TILE_KINDS
            for code, c in hand[s].items():
                counts[tiles.parse(code)] = c
            try:
                if shanten_any(counts, melds[s], memo=memo) != 0:
                    continue
                waits = win.winning_draws(counts, melds[s])
            except Exception:
                continue
            if not waits:
                continue
            eff = 0
            for w in waits:
                visible = counts[w] + discards[w] + sum(meld_tiles[o][w] for o in range(4))
                eff += max(0, 4 - visible)
            who = "us" if s == our else "opp"
            points.append((who, width_bucket(eff), key, s))
        elif et == "chi":
            for t in data.get("tiles") or []:
                meld_tiles[s][tiles.parse(t)] += 1
            need = collections.Counter(data.get("tiles") or [])
            need[tile] -= 1
            if need[tile] == 0:
                del need[tile]
            for t, c in need.items():
                hand[s][t] -= c
            melds[s] += 1
        elif et == "peng":
            meld_tiles[s][tiles.parse(tile)] += 3
            hand[s][tile] -= 2
            melds[s] += 1
        elif et == "gang":
            kind = data.get("kind")
            meld_tiles[s][tiles.parse(tile)] += 4
            hand[s][tile] -= {"an": 4, "bu": 1, "ming": 3}.get(kind, 0)
            if kind != "bu":
                melds[s] += 1


def main():
    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    by_room = collections.defaultdict(list)
    for p in files:
        by_room[p.split("/")[-3]].append(p)
    files = [p for paths in by_room.values() for p in sorted(paths)[:PER_ROOM]]
    memo: dict = {}
    points = []
    win_by_key = {}
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        uids = [s.get("user_id") for s in doc.get("seats") or []]
        if OUR_UID not in uids:
            continue
        our = uids.index(OUR_UID)
        for rnd in split_rounds(doc):
            key = (p, rnd["round_no"])
            win_seat, win_fan = None, 0
            for e in rnd["events"]:
                if e["type"] == "round_ended":
                    d = e.get("data") or {}
                    if not d.get("draw"):
                        win_seat = e.get("seat")
                        win_fan = d.get("fan") or 0
            win_by_key[key] = (win_seat, win_fan)
            scan(our, rnd, memo, points, key)

    agg = collections.Counter()
    for who, b, key, s in points:
        agg[(who, b, "n")] += 1
        w, fan = win_by_key.get(key, (None, 0))
        if w == s:
            agg[(who, b, "win")] += 1
            agg[(who, b, "fan")] += fan

    print(f"状态点={len(points)}（听牌后出牌点）")
    print(f"{'听口桶':>6} | {'us胜率':>7} {'us番':>5} {'n':>6} | {'opp胜率':>8} {'opp番':>6} {'n':>7}")
    out = {}
    for b in ("1-4", "5-8", "9-12", "13-20", "21+"):
        row = {}
        for who in ("us", "opp"):
            n = agg[(who, b, "n")]
            w = agg[(who, b, "win")]
            f = agg[(who, b, "fan")]
            row[who] = {"n": n, "p_win": w / n if n else 0, "fan_avg": f / w if w else 0}
        if row["us"]["n"] < 30 or row["opp"]["n"] < 90:
            continue
        out[b] = row
        print(f"{b:>6} | {row['us']['p_win']:7.1%} {row['us']['fan_avg']:5.2f} {row['us']['n']:>6} | "
              f"{row['opp']['p_win']:8.1%} {row['opp']['fan_avg']:6.2f} {row['opp']['n']:>7}")
    with open("verify/out/wait_width_table.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("JSON -> verify/out/wait_width_table.json")


if __name__ == "__main__":
    main()
