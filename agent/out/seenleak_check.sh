#!/bin/sh
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "occupancy_gonogo.py.*features base" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "seen 查表(test建" agent/out/occupancy-seenleak.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
