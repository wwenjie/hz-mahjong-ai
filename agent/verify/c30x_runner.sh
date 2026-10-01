#!/usr/bin/env bash
set -u
cd /home/wuwenjie01/majiang_ai
LOG=agent/out/c30x-runner.log
PY=.venv/bin/python
PROBE=agent/verify/convert_by_remaining_probe.py
echo "$(date '+%F %H:%M:%S') runner 启动 pid=$$" >> "$LOG"
for i in $(seq 1 60); do
  if grep -q "^PROBE_DONE$" "$LOG" 2>/dev/null; then
    echo "$(date '+%F %H:%M:%S') PROBE_DONE，退出" >> "$LOG"
    exit 0
  fi
  echo "$(date '+%F %H:%M:%S') iter $i: 启动/续跑" >> "$LOG"
  nice -n 19 "$PY" -u "$PROBE" --chunk-size 250 --cross >> "$LOG" 2>&1
  rc=$?
  echo "$(date '+%F %H:%M:%S') iter $i rc=$rc" >> "$LOG"
  sleep 5
done
echo "$(date '+%F %H:%M:%S') 超 60 次重冲，放弃" >> "$LOG"
exit 1
