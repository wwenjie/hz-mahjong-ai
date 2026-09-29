#!/usr/bin/env python3
"""agent-c: 探查 replay 状态的可用字段（弃牌顺序/巡目），为 recency 特征铺路。只读。"""
import glob
import json

from majiang.sim import replay
from majiang.rules.situation import PHASE_DRAW

OUR = "u_a7f7c67bb14a"

f = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[0]
doc = json.loads(open(f, encoding="utf-8").read())
ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
print("文件:", f, "| ids:", ids, "| OUR in ids:", OUR in ids)
mine = ids.index(OUR) if OUR in ids else 0

n = 0
for event, state in replay.iter_before_each_event(doc):
    n += 1
    if n != 40:
        continue
    print("\n=== 第 40 个事件前的 state ===")
    print("event type:", event.get("type"))
    print("state 属性:", [a for a in dir(state) if not a.startswith("_")])
    seat = state.seats[mine]
    print("seat[我方] 属性:", [a for a in dir(seat) if not a.startswith("_")])
    print("seat[我方].discards 类型/长度:", type(seat.discards), len(seat.discards))
    print("seat[我方].discards 内容:", list(seat.discards)[:20])
    print("seat[对手0].discards 内容:", list(state.seats[(mine + 1) % 4].discards)[:20])
    print("state.table 属性:", [a for a in dir(state.table) if not a.startswith("_")])
    print("draws_made:", getattr(state.table, "draws_made", None),
          "| wall_remaining:", getattr(state.table, "wall_remaining", None),
          "| progress:", getattr(state.table, "progress", None))
    # situation
    sit = state.situation_for(mine, phase=PHASE_DRAW)
    print("situation 属性:", [a for a in dir(sit) if not a.startswith("_")])
    print("sit.discards:", list(sit.discards)[:20], "len=", len(sit.discards))
    print("sit.table 属性:", [a for a in dir(sit.table) if not a.startswith("_")])
    break

if n < 40:
    print("只有", n, "个事件")
