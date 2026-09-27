#!/usr/bin/env python3
"""v1→v2 时代的胜率差分中差分（DiD）：我们的胜率变化 − 同场对手的变化。

v1w = 账本中 2026-09-26T04:50 之前的 heuristic 行（=blocks 档）
v2d = 2026-09-26T05:46 ~ 2026-09-27T16:40 的 heuristic 行（=exact-ukeire 默认档）
对手口径 = 同房三家合计胜率（±时间内对场强漂移做差分消除）。

用法：nice -n 15 uv run python verify/era_did.py
"""
import collections
import glob
import json

OUR = "u_a7f7c67bb14a"
WINDOWS = {
    "v1w": ("heuristic", "", "2026-09-26T04:50"),
    "v2d": ("heuristic", "2026-09-26T05:46", "2026-09-27T16:40"),
}


def main():
    ledger = [json.loads(l) for l in open("data/auto_sessions/sessions.jsonl", encoding="utf-8")]
    for arm, (decider, lo, hi) in WINDOWS.items():
        rooms = [r["room_id"] for r in ledger
                 if r.get("decider") == decider and lo <= r.get("started_at", "") < hi]
        us_w = us_n = op_w = op_n = 0
        for room in rooms:
            for p in glob.glob(f"data/auto_sessions/{room}/events/*.json"):
                doc = json.load(open(p, encoding="utf-8"))
                if doc.get("status") != "finished":
                    continue
                uids = [s.get("user_id") for s in doc.get("seats") or []]
                if OUR not in uids:
                    continue
                our = uids.index(OUR)
                seen = set()
                for b in doc["blocks"]:
                    for e in b.get("events") or []:
                        if e["type"] != "round_ended":
                            continue
                        d = e.get("data") or {}
                        if d.get("draw") or d.get("round_no") in seen:
                            continue
                        seen.add(d.get("round_no"))
                        us_n += 1
                        op_n += 3
                        w = e.get("seat")
                        if w == our:
                            us_w += 1
                        elif w in (0, 1, 2, 3):
                            op_w += 1
        print(f"[{arm}] 房={len(rooms)} 局={us_n} 我们胜率={us_w/us_n:.2%} "
              f"对手胜率={op_w/op_n:.2%}")


if __name__ == "__main__":
    main()
