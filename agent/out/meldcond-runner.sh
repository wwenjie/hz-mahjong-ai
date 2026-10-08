#!/bin/bash
# 1a 自愈 runner：幂等续跑 + PROBE_DONE 收口 + CANCEL 外部取消通道
cd /home/wuwenjie01/majiang_ai
LOG=agent/out/meldcond-runner.log
CANCEL=agent/out/meldcond-runner.cancel
for i in $(seq 1 40); do
  [ -f "$CANCEL" ] && echo "$(date '+%F %T') CANCELLED" >> "$LOG" && exit 0
  .venv/bin/python agent/verify/meld_cond_dist_probe.py --rooms 0 --chunk-size 500 >> "$LOG" 2>&1
  grep -q PROBE_DONE "$LOG" && echo "$(date '+%F %T') PROBE_DONE rc=0" >> "$LOG" && exit 0
  echo "$(date '+%F %T') restart round $i" >> "$LOG"
  sleep 5
done
echo "$(date '+%F %T') EXHAUSTED" >> "$LOG"
