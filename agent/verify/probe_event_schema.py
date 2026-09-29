#!/usr/bin/env python3
"""agent-c: 深挖事件流 schema（事件在 rounds/blocks 下）。只读。"""
import glob
import json
from collections import Counter

f = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[0]
doc = json.loads(open(f, encoding="utf-8").read())
print("文件:", f)
print("顶层键:", list(doc.keys()))
print("rounds 类型:", type(doc.get("rounds")), "len=", len(doc.get("rounds") or []))
print("blocks 类型:", type(doc.get("blocks")), "len=", len(doc.get("blocks") or []))
print()
rounds = doc.get("rounds") or []
if rounds:
    r0 = rounds[0]
    print("round[0] 键:", list(r0.keys()) if isinstance(r0, dict) else type(r0))
    print("round[0] 摘要:", json.dumps(r0, ensure_ascii=False)[:600])
print()
blocks = doc.get("blocks") or []
if blocks:
    b0 = blocks[0]
    print("block[0] 类型:", type(b0))
    print("block[0]:", json.dumps(b0, ensure_ascii=False)[:600])
print()
# 找所有事件类型
types = Counter()
def walk(o):
    if isinstance(o, dict):
        if "type" in o and isinstance(o["type"], str):
            types[o["type"]] += 1
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(doc)
print("所有 type 计数:", types.most_common(20))
