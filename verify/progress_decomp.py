#!/usr/bin/env python3
"""进度差分解：差距集中在哪个向听等级（回 A 的「阈值微调是否到头」+ 校验 ukeire-early 落点）。

hand_progress 显示差距从第 2 摸起单调扩大。本脚本回答：差距在「3→2」「2→1」「1→0」
哪一段？若集中在 2~3 向听区，ukeire-early（把精确进张扩展到 shanten≤3）打在正点；
若集中在 1→0（听牌转化），则是别的病。

口径：每座位每次摸牌后记录精确向听（shanten_any），按摸序号 n（2..8）与向听值分桶。
用法：nice -n 15 uv run python verify/progress_decomp.py
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


def scan(our, rnd, memo, out):
    if rnd["hands"] is None:
        return
    hand = [collections.Counter(h) for h in rnd["hands"]]
    melds = [0, 0, 0, 0]
    draw_no = [0, 0, 0, 0]
    prev_shanten = [None] * 4
    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if s not in (0, 1, 2, 3):
            continue
        if et == "tile_drawn":
            hand[s][tile] += 1
            draw_no[s] += 1
            counts = [0] * tiles.TILE_KINDS
            for code, c in hand[s].items():
                counts[tiles.parse(code)] = c
            try:
                sh = shanten_any(counts, melds[s], memo=memo)
            except Exception:
                continue
            who = "us" if s == our else "opp"
            n = draw_no[s]
            if 1 <= n <= 8:
                out[(who, n, "shanten")].append(sh)
                ps = prev_shanten[s]
                if ps is not None and ps > 0:
                    bucket = min(ps, 3)
                    out[(who, bucket, "drop_n")].append(1 if sh < ps else 0)
            prev_shanten[s] = sh
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
        n += 1
        for rnd in split_rounds(doc):
            scan(uids.index(OUR_UID), rnd, memo, out)

    print(f"场次={n}")
    print("\n== 每摸序号均向听（幸存者偏差提示：n 大端可比） ==")
    print(f"{'n':>3} {'us':>6} {'opp':>6} {'差':>7} {'us样本':>7} {'opp样本':>8}")
    for k in range(1, 9):
        a, b = out[("us", k, "shanten")], out[("opp", k, "shanten")]
        if a and b:
            print(f"{k:>3} {np.mean(a):6.3f} {np.mean(b):6.3f} {np.mean(b)-np.mean(a):+7.3f} "
                  f"{len(a):>7} {len(b):>8}")
    print("\n== 向听下降概率 P(sh 降 | 上一摸后为 sh)（按上一摸的向听分桶） ==")
    for bucket in (3, 2, 1):
        a = out[("us", bucket, "drop_n")]
        b = out[("opp", bucket, "drop_n")]
        if a and b:
            print(f"{bucket}→更低: us {np.mean(a):.1%}(n={len(a)})  "
                  f"opp {np.mean(b):.1%}(n={len(b)})  差 {np.mean(b)-np.mean(a):+.1%}")


if __name__ == "__main__":
    main()
