#!/bin/sh
# agent-c: 占用 Go/No-Go（第二跑，含偏差诊断）守望探针。只读。
# 输出 "<running 0|1> <has_result 0|1>"。
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "occupancy_gonogo.py" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "学习占用 hat" agent/out/occupancy-gonogo2.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
