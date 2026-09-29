#!/bin/sh
# agent-c: 容量扫描守望探针。只读。
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "occupancy_capacity_sweep.py" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "McNemar" agent/out/occupancy-capacity.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
