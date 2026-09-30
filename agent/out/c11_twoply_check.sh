#!/bin/sh
# C11 追加：two-ply 补种子结果守望（落盘即报）
cd /home/wuwenjie01/majiang_ai || exit 1
need="data/experiments/logs/two-ply-vs-v3-s31415927-seed31415927.log data/experiments/logs/two-ply-vs-v3-s27182818-seed27182818.log data/experiments/logs/two-ply-vs-v3-s16180339-seed16180339.log"
have=0; for f in $need; do
  if [ -f "$f" ] && grep -q "逐场配对差分" "$f"; then have=$((have+1)); fi
done
if pgrep -f "ab_test.py --treatment two-ply" >/dev/null 2>&1; then run=1; else run=0; fi
echo "$have $run"
