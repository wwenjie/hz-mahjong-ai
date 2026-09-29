#!/usr/bin/env python3
"""agent-c: 判清服务端 `timeout` 事件的真实语义——是「我们没按时响应」还是「该窗口无人动作」。
只读。"""
import glob
import json
from collections import Counter
from pathlib import Path

OUR = "u_a7f7c67bb14a"

# 1) 一局里，按 seat 统计 timeout 事件分布（若四家都有大量 timeout ⇒ 它是「该窗口无人动作」的日志）
f = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[0]
doc = json.loads(Path(f).read_text(encoding="utf-8"))
ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
mine = ids.index(OUR) if OUR in ids else 0
print("文件:", Path(f).name, "| mine =", mine)
by_seat = Counter()
by_seat_kind = Counter()
for blk in doc.get("blocks") or []:
    for e in (blk.get("events") or []):
        if e.get("type") == "timeout":
            s = int(e.get("seat", -1))
            by_seat[s] += 1
            by_seat_kind[(s, (e.get("data") or {}).get("kind"))] += 1
print("该局 timeout 按 seat:", dict(by_seat))
print("该局 timeout 按 (seat,kind):", dict(by_seat_kind))
print()

# 2) 找一个「我方座位 discard timeout」的上下文
found = 0
for blk in doc.get("blocks") or []:
    evs = blk.get("events") or []
    for i, e in enumerate(evs):
        if e.get("type") == "timeout" and int(e.get("seat", -1)) == mine \
                and (e.get("data") or {}).get("kind") == "discard":
            lo = max(0, i - 3)
            hi = min(len(evs), i + 4)
            print(f"--- 我方 discard-timeout 上下文 (block dealer={blk.get('dealer')}) ---")
            for j in range(lo, hi):
                x = evs[j]
                print(f"  [{j}]{'>>' if j == i else '  '}", json.dumps(x, ensure_ascii=False)[:200])
            found += 1
            break
    if found >= 2:
        break
if found == 0:
    print("(该局我方无 discard timeout)")
print()

# 3) 全局：我方座位 timeout 的 kind × 是否有对应 decision.made
print("=== 全局统计（前 200 房抽样）===")
tot = Counter()
for p in sorted(glob.glob("data/auto_sessions/*/events/*.json"))[:200]:
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        continue
    idl = [str(s.get("user_id", "")) for s in (d.get("seats") or [])]
    if len(idl) != 4 or OUR not in idl:
        continue
    mi = idl.index(OUR)
    for blk in d.get("blocks") or []:
        for e in (blk.get("events") or []):
            if e.get("type") == "timeout":
                tot[("ALL", (e.get("data") or {}).get("kind"))] += 1
                if int(e.get("seat", -1)) == mi:
                    tot[("MINE", (e.get("data") or {}).get("kind"))] += 1
print("抽样 200 房 timeout 计数:", dict(tot))
