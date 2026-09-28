#!/usr/bin/env python3
"""真值表：P(降向听/最终自摸 | 向听, 财神数, 副露数, 阶段)，我们 vs 对手分建。

用途（A 的「统一期望得分」前置）：把手调线性组合的隐含定价与实测价格对拍。
分桶：shanten ∈ {0,1,2,3,4+} × god ∈ {0,1,≥2} × melds ∈ {0,1,≥2} × 阶段（摸序号 ≤4 / 5-8 / ≥9）。
结局：P(下一摸后向听下降) 与 P(本局最终自摸胜)。
过滤：抓打圈强制的出牌不参与（那是被迫的）。

用法：nice -n 19 uv run python verify/win_rate_table.py
"""
import collections
import json

import numpy as np

from majiang.rules import tiles
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


def bucket(sh, god, melds, draw_no):
    shb = min(sh, 4)
    godb = min(god, 2)
    mb = min(melds, 2)
    phase = 0 if draw_no <= 4 else (1 if draw_no <= 8 else 2)
    return (shb, godb, mb, phase)


def scan(our, rnd, memo, out, winner):
    if rnd["hands"] is None:
        return
    hand = [collections.Counter(h) for h in rnd["hands"]]
    melds = [0, 0, 0, 0]
    draw_no = [0, 0, 0, 0]
    catch_play = False
    last_drawn = [None] * 4
    # 记录每座位在 bucket 状态下的后续：下一摸后向听是否降、本局是否自摸胜
    prev_sh = [None] * 4
    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if et == "round_ended":
            return
        if s not in (0, 1, 2, 3):
            continue
        if et == "tile_drawn":
            hand[s][tile] += 1
            draw_no[s] += 1
            last_drawn[s] = tile
        elif et == "tile_discarded":
            if data.get("catch_play"):
                catch_play = True
            hand[s][tile] -= 1
            # 出牌后评估（自由决策样本）
            counts = [0] * tiles.TILE_KINDS
            for code, c in hand[s].items():
                counts[tiles.parse(code)] = c
            god = counts[tiles.GOD]
            try:
                sh = shanten_any(counts, melds[s], memo=memo)
            except Exception:
                continue
            who = "us" if s == our else "opp"
            b = bucket(sh, god, melds[s], draw_no[s])
            # 到听/自摸结局后填：先记账状态点
            out["points"].append((who, b, s))
            prev_sh[s] = sh
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


def main():
    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    memo: dict = {}
    # 两遍：先收集状态点（谁/桶/局id/座位），再按局结局回填
    # 简化：扫一遍，记录 (who,bucket,seat,file,round) 与最终赢家，再聚合
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
            win_seat = None
            win_fan = 0
            for e in rnd["events"]:
                if e["type"] == "round_ended":
                    d = e.get("data") or {}
                    if not d.get("draw"):
                        win_seat = e.get("seat")
                        win_fan = d.get("fan") or 0
            win_by_key[key] = (win_seat, win_fan)
            out = {"points": []}
            scan(our, rnd, memo, out, win_seat)
            for who, b, s in out["points"]:
                points.append((who, b, key, s))

    agg = collections.Counter()
    for who, b, key, s in points:
        agg[(who, b, "n")] += 1
        w, fan = win_by_key.get(key, (None, 0))
        if w == s:
            agg[(who, b, "win")] += 1
            agg[(who, b, "fan")] += fan

    # 打印：按 (shanten, god, melds) 合并阶段，先看主效应
    print("== P(最终自摸 | 向听, 财神, 副露) 与胡时均番 —— 我们 vs 对手 ==")
    print(f"{'向听':>3} {'财神':>3} {'副露':>3} | {'us胜率':>7} {'us番':>5} {'n':>6} | "
          f"{'opp胜率':>8} {'opp番':>6} {'n':>7} | {'胜率差':>6}")
    for shb in range(5):
        for godb in range(3):
            for mb in range(3):
                us_n = sum(agg[("us", (shb, godb, mb, ph), "n")] for ph in range(3))
                op_n = sum(agg[("opp", (shb, godb, mb, ph), "n")] for ph in range(3))
                if us_n < 200 or op_n < 500:
                    continue
                us_w = sum(agg[("us", (shb, godb, mb, ph), "win")] for ph in range(3))
                op_w = sum(agg[("opp", (shb, godb, mb, ph), "win")] for ph in range(3))
                us_f = sum(agg[("us", (shb, godb, mb, ph), "fan")] for ph in range(3))
                op_f = sum(agg[("opp", (shb, godb, mb, ph), "fan")] for ph in range(3))
                ur, orr = us_w / us_n, op_w / op_n
                uf = us_f / us_w if us_w else 0
                of = op_f / op_w if op_w else 0
                print(f"{shb:>3} {godb:>3} {mb:>3} | {ur:7.1%} {uf:5.2f} {us_n:>6} | "
                      f"{orr:8.1%} {of:6.2f} {op_n:>7} | {orr-ur:+6.1%}")


if __name__ == "__main__":
    main()
