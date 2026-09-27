#!/usr/bin/env python3
"""弃胡求爆头的对手行为基线（扩展 declined_wins：只测过我们 → 全座位 + 分身份）。

回答：场上对手弃胡率多少？自补率多少？——决定我们 chase 策略在决赛的相对位置：
若对手普遍不追（稳收），我们 chase 是差异化优势；若对手都追，chase 只是标配。

口径同 declined_wins.py：compute_fan 判可胡，下一张动作是出牌=弃胡。
用法：nice -n 15 uv run python verify/declined_wins_all.py
"""
import collections
import json

import numpy as np

from majiang.rules import tiles
from majiang.rules.fan import compute_fan

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


def scan_round(uids, rnd, per_uid, memo):
    """对 4 个座位同时跟踪弃胡。"""
    if rnd["hands"] is None:
        return
    hands = []
    for h in rnd["hands"]:
        counts = [0] * tiles.TILE_KINDS
        for code in h:
            counts[tiles.parse(code)] += 1
        hands.append(counts)
    melds = [0, 0, 0, 0]
    open_state = [None] * 4  # None | ("winnable", fan) | ("declined", fan)

    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if et == "round_ended":
            w = e.get("seat")
            draw = bool(data.get("draw"))
            for s2 in range(4):
                st = open_state[s2]
                if st is None:
                    continue
                uid = uids[s2]
                if st[0] == "winnable":
                    per_uid[uid]["took"] += 1
                else:
                    if not draw and w == s2:
                        per_uid[uid]["self_win"] += 1
                    else:
                        per_uid[uid]["lost"] += 1
            return
        if s not in (0, 1, 2, 3):
            continue
        if et == "tile_drawn":
            t = tiles.parse(tile)
            res = compute_fan(hands[s], t, melds[s])
            if res.hu:
                per_uid[uids[s]]["winnable"] += 1
                if open_state[s] is None:
                    open_state[s] = ("winnable", res.fan)
            hands[s][t] += 1
        elif et == "tile_discarded":
            if open_state[s] and open_state[s][0] == "winnable":
                per_uid[uids[s]]["declined"] += 1
                open_state[s] = ("declined", open_state[s][1])
            hands[s][tiles.parse(tile)] -= 1
        elif et in ("chi", "peng", "gang"):
            if et == "chi":
                need = collections.Counter(data.get("tiles") or [])
                need[tile] -= 1
                if need[tile] == 0:
                    del need[tile]
                for code, c in need.items():
                    hands[s][tiles.parse(code)] -= c
                melds[s] += 1
            elif et == "peng":
                hands[s][tiles.parse(tile)] -= 2
                melds[s] += 1
            else:
                kind = data.get("kind")
                hands[s][tiles.parse(tile)] -= {"an": 4, "bu": 1, "ming": 3}.get(kind, 0)
                if kind != "bu":
                    melds[s] += 1
            open_state[s] = None


def main():
    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    per_uid = collections.defaultdict(collections.Counter)
    memo: dict = {}
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        uids = [s.get("user_id") for s in doc.get("seats") or []]
        if OUR_UID not in uids:
            continue
        for rnd in split_rounds(doc):
            scan_round(uids, rnd, per_uid, memo)

    def rate(c, k, k2):
        return c[k] / c[k2] if c[k2] else 0.0

    us = per_uid[OUR_UID]
    opps = {u: c for u, c in per_uid.items()
            if u != OUR_UID and c["winnable"] >= 20}
    opp_decline = [rate(c, "declined", "winnable") for c in opps.values()]
    opp_recover = [c["self_win"] / c["declined"] for c in opps.values() if c["declined"] >= 5]

    print(f"我们: 可胡 {us['winnable']} 弃胡 {us['declined']}"
          f"（{rate(us, 'declined', 'winnable'):.1%}）"
          f" 自补率 {us['self_win'] / max(1, us['declined']):.1%}")
    print(f"对手身份数(≥20 可胡): {len(opps)}")
    print(f"对手弃胡率: 中位 {np.median(opp_decline):.1%} "
          f"IQR [{np.percentile(opp_decline,25):.1%}, {np.percentile(opp_decline,75):.1%}]"
          f" 最大 {max(opp_decline):.1%}")
    if opp_recover:
        print(f"对手自补率(弃胡≥5): 中位 {np.median(opp_recover):.1%} "
              f"IQR [{np.percentile(opp_recover,25):.1%}, {np.percentile(opp_recover,75):.1%}]")


if __name__ == "__main__":
    main()
