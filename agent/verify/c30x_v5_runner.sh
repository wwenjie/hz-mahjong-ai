#!/usr/bin/env bash
set -u
cd /home/wuwenjie01/majiang_ai
LOG=agent/out/c30x-v5-runner.log
PY=.venv/bin/python
echo "$(date '+%F %H:%M:%S') runner 启动 pid=$$" >> "$LOG"
# 先重建全量映射（含历史 + v6 窗口）
"$PY" agent/verify/build_arm_map.py >> "$LOG" 2>&1
for i in $(seq 1 40); do
  if grep -q "^PROBE_DONE$" "$LOG" 2>/dev/null; then
    echo "$(date '+%F %H:%M:%S') PROBE_DONE，退出" >> "$LOG"; exit 0
  fi
  echo "$(date '+%F %H:%M:%S') iter $i: v5 臂续跑" >> "$LOG"
  nice -n 19 "$PY" -u agent/verify/convert_by_remaining_probe.py \
    --chunk-size 250 --cross --arm-map agent/out/arm_map.json --arm v5 >> "$LOG" 2>&1
  echo "$(date '+%F %H:%M:%S') iter $i rc=$?" >> "$LOG"
  sleep 5
done
exit 1
