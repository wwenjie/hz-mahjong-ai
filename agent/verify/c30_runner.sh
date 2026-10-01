#!/usr/bin/env bash
# C30 全量自愈 runner：本机会周期性回收长跑进程——被杀就靠已落盘 chunk 幂等续跑，
# 直到探针打印 PROBE_DONE。输出直接 >> 日志（不走管道，防回收时丢缓冲）。
set -u
cd /home/wuwenjie01/majiang_ai
LOG=agent/out/c30-full-runner.log
PY=.venv/bin/python
PROBE=agent/verify/convert_by_remaining_probe.py
echo "$(date '+%F %H:%M:%S') runner 启动 pid=$$" >> "$LOG"
for i in $(seq 1 60); do
  if grep -q "^PROBE_DONE$" "$LOG" 2>/dev/null; then
    echo "$(date '+%F %H:%M:%S') 检测到 PROBE_DONE，runner 退出" >> "$LOG"
    exit 0
  fi
  echo "$(date '+%F %H:%M:%S') iter $i: 启动/续跑探针" >> "$LOG"
  nice -n 19 "$PY" -u "$PROBE" --rooms 0 --chunk-size 250 >> "$LOG" 2>&1
  rc=$?
  echo "$(date '+%F %H:%M:%S') iter $i 结束 rc=$rc" >> "$LOG"
  sleep 5
done
echo "$(date '+%F %H:%M:%S') 超 60 次重冲仍未完成，runner 放弃" >> "$LOG"
exit 1
