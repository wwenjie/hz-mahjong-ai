#!/bin/sh
# agent-c: recency 特征跑守望探针。只读。
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "occupancy_gonogo.py" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "仅复制 seen 查表" agent/out/occupancy-recency.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
