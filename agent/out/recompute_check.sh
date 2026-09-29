#!/bin/sh
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "recompute_metrics_probe.py" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "不变量违规数" agent/out/recompute-metrics-full.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
