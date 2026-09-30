#!/bin/sh
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "reach_vs_convert.py" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "缺口主要落在" agent/out/reach-vs-convert.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
