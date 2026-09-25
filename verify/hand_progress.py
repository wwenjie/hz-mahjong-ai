#!/usr/bin/env python3
"""独立复核「手牌进度曲线」（A 的 analyze_hand_progress 只在 60 文件/6 房上跑过）。

口径：每座位第 n 次摸牌后，用 shanten_any（含财神百搭）算向听，听牌 = 向听 0。
按 us/opp 分组报 P(听牌) 与样本量。幸存者偏差：只比 n≤8（A 的教训）。

与 A 的工具的差异：自己从原始 JSON 重建手牌（复用我在 invariants.py 验证过的
切局/跟踪逻辑），不 import A 的 analyze_hand_progress；向听函数用共享规则层
（rules.shanten，官方对拍测试锁定，非 A 的测量代码）。

用法：uv run python verify/hand_progress.py [--max-n 12]
"""
import argparse
import collections
import json

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


def scan_round(our_seat, rnd, memo, out):
    """对一局逐座位跟踪：第 n 次摸牌后向听。"""
    if rnd["hands"] is None:
        return
    hand = [collections.Counter() for _ in range(4)]
    for s in range(4):
        hand[s].update(rnd["hands"][s])
    meld_sets = [0, 0, 0, 0]
    draw_no = [0, 0, 0, 0]

    def record(seat):
        counts = [0] * tiles.TILE_KINDS
        for code, c in hand[seat].items():
            counts[tiles.parse(code)] = c
        n = draw_no[seat]
        who = "us" if seat == our_seat else "opp"
        try:
            sh = shanten_any(counts, meld_sets[seat], memo=memo)
        except Exception:
            out["errors"] += 1
            return
        out[(who, n, "tenpai")] += int(sh == 0)
        out[(who, n, "total")] += 1
        out[(who, n, "shanten_sum")] += sh

    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if et == "tile_drawn" and s in (0, 1, 2, 3):
            hand[s][tile] += 1
            draw_no[s] += 1
            record(s)
        elif et == "tile_discarded" and s in (0, 1, 2, 3):
            hand[s][tile] -= 1
        elif et == "chi" and s in (0, 1, 2, 3):
            need = collections.Counter(data.get("tiles") or [])
            need[tile] -= 1
            if need[tile] == 0:
                del need[tile]
            for t, c in need.items():
                hand[s][t] -= c
            meld_sets[s] += 1
        elif et == "peng" and s in (0, 1, 2, 3):
            hand[s][tile] -= 2
            meld_sets[s] += 1
        elif et == "gang" and s in (0, 1, 2, 3):
            kind = data.get("kind")
            hand[s][tile] -= {"an": 4, "bu": 1, "ming": 3}.get(kind, 0)
            if kind != "bu":
                meld_sets[s] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=12)
    ap.add_argument("--manifest", default="notes/manifest-20260926.txt")
    args = ap.parse_args()

    files = [l.strip() for l in open(args.manifest, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    out = collections.Counter()
    memo: dict = {}
    n_matches = 0
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        seats = doc.get("seats") or []
        our = next((i for i, s in enumerate(seats) if s.get("user_id") == OUR_UID), None)
        if our is None:
            continue
        n_matches += 1
        for rnd in split_rounds(doc):
            scan_round(our, rnd, memo, out)

    print(f"场次={n_matches} 向听调用异常={out['errors']}")
    print(f"{'n':>3} | {'us 听牌率':>9} {'样本':>7} | {'opp 听牌率':>10} {'样本':>8} | {'差(opp-us)':>9}")
    for n in range(1, args.max_n + 1):
        ut, ot = out[("us", n, "total")], out[("opp", n, "total")]
        if not ut or not ot:
            continue
        up = out[("us", n, "tenpai")] / ut
        op = out[("opp", n, "tenpai")] / ot
        flag = " " if n <= 8 else "(幸存者偏差区)"
        print(f"{n:>3} | {up:9.1%} {ut:>7} | {op:10.1%} {ot:>8} | {op - up:+9.1%} {flag}")


if __name__ == "__main__":
    main()
