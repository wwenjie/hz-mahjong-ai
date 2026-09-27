#!/usr/bin/env python3
"""P1：独立复算 v2 切换的到听率差分（A 报：我们 22.0%→23.8%，对手 +0.7，DiD +1.1pp）。
不复用 tools/analyze_wait_quality.py，自建重建与统计。

口径（显式声明，避免重蹈「摸牌前/出牌前」混淆）：
- **出牌后判定**：每次 tile_discarded 事件后，该座位手牌 shanten==0 记为「到听」
- 到听率 = 到听出牌数 / 总出牌数（分 us / opp）
- 时段划分（用账本 sessions.jsonl 的 decider+started_at，不用事件内容推）：
  v1  = 03:13~04:50 的 heuristic 行（configure 修复后、v2 默认前；当时 heuristic=blocks）
  v2i = 同窗的 ukeire-exact 行（交错对照，与 v2 同构）
  v2d = 05:46~今天16:40 的 heuristic 行（v2 默认档，缓冲带避开切换边界；
        上限避开 16:45 起混入的 first-legal 交错臂）
- DiD = (v2 到听率 − v1 到听率) − (同期对手到听率差)

用法：uv run python verify/wait_quality.py
"""
import collections
import json

from majiang.rules import tiles
from majiang.rules.shanten import shanten_any

OUR_UID = "u_a7f7c67bb14a"
LEDGER = "data/auto_sessions/sessions.jsonl"

ARMS = {  # arm -> (decider, started_at 下界, 上界)
    "v1": ("heuristic", "2026-09-26T03:13", "2026-09-26T04:50"),
    "v1w": ("heuristic", "", "2026-09-26T04:50"),  # 加厚 v1：切换前全部 heuristic 行
    "v2i": ("ukeire-exact", "2026-09-26T03:13", "2026-09-26T04:50"),
    "v2d": ("heuristic", "2026-09-26T05:46", "2026-09-27T16:40"),
}


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


def scan_file(path, our_seat, memo, out):
    """累计该场 4 座位「出牌后到听」计数。out[(who,'t')]/(who,'n')。"""
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("status") != "finished":
        return
    for rnd in split_rounds(doc):
        if rnd["hands"] is None:
            continue
        hand = [collections.Counter(h) for h in rnd["hands"]]
        melds = [0, 0, 0, 0]
        for e in rnd["events"]:
            et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
            if s not in (0, 1, 2, 3):
                continue
            if et == "tile_drawn":
                hand[s][tile] += 1
            elif et == "tile_discarded":
                hand[s][tile] -= 1
                counts = [0] * tiles.TILE_KINDS
                for code, c in hand[s].items():
                    counts[tiles.parse(code)] = c
                who = "us" if s == our_seat else "opp"
                out[(who, "n")] += 1
                try:
                    if shanten_any(counts, melds[s], memo=memo) == 0:
                        out[(who, "t")] += 1
                except Exception:
                    out[(who, "err")] += 1
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
    ledger = [json.loads(l) for l in open(LEDGER, encoding="utf-8")]
    memo: dict = {}
    per_arm = {}
    for arm, (decider, lo, hi) in ARMS.items():
        rooms = [r["room_id"] for r in ledger
                 if r.get("decider") == decider and lo <= r.get("started_at", "") < hi]
        out = collections.Counter()
        n_files = 0
        for room in rooms:
            import glob
            for p in glob.glob(f"data/auto_sessions/{room}/events/*.json"):
                doc0 = json.load(open(p, encoding="utf-8"))
                if doc0.get("status") != "finished":
                    continue
                seats = doc0.get("seats") or []
                our = next((i for i, s in enumerate(seats)
                            if s.get("user_id") == OUR_UID), None)
                if our is None:
                    continue
                scan_file(p, our, memo, out)
                n_files += 1
        per_arm[arm] = (out, len(rooms), n_files)

    print(f"{'arm':>4} {'房':>3} {'场':>4} | {'us到听率':>8} {'us出牌':>7} | "
          f"{'opp到听率':>9} {'opp出牌':>8}")
    rates = {}
    for arm, (out, nrooms, nfiles) in per_arm.items():
        ur = out[("us", "t")] / out[("us", "n")] if out[("us", "n")] else 0
        orr = out[("opp", "t")] / out[("opp", "n")] if out[("opp", "n")] else 0
        rates[arm] = (ur, orr)
        print(f"{arm:>4} {nrooms:>3} {nfiles:>4} | {ur:8.2%} {out[('us','n')]:>7} | "
              f"{orr:9.2%} {out[('opp','n')]:>8}  (err={out[('us','err')]})")

    print("\n— 差分（A 报：us +1.8pp, opp +0.7pp, DiD +1.1pp）—")
    if "v1" in rates and "v2i" in rates:
        d_us = rates["v2i"][0] - rates["v1"][0]
        d_op = rates["v2i"][1] - rates["v1"][1]
        print(f"交错窗(v2i vs v1): us {d_us:+.2%} opp {d_op:+.2%} DiD {d_us - d_op:+.2%}")
    if "v1w" in rates and "v2d" in rates:
        d_us = rates["v2d"][0] - rates["v1w"][0]
        d_op = rates["v2d"][1] - rates["v1w"][1]
        print(f"前后段(v2d vs v1w 加厚): us {d_us:+.2%} opp {d_op:+.2%} DiD {d_us - d_op:+.2%}")


if __name__ == "__main__":
    main()
