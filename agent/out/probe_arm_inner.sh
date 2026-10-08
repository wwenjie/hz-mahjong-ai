#!/usr/bin/env bash
# 三臂内层等价性探针（耐久版）：独立脚本 + 无缓冲 + 外层重冲。
set +e
cd /home/wuwenjie01/majiang_ai || exit 1
LOG=agent/out/probe-arm-inner.log
: >"$LOG"
for attempt in 1 2 3 4 5 6; do
  echo "--- attempt $attempt $(date -Is) ---" >>"$LOG"
  nice -n 19 .venv/bin/python -u agent/verify/arm_inner_equivalence_probe.py >>"$LOG" 2>&1
  rc=$?
  if grep -q PROBE_INNER_DONE "$LOG"; then echo "完成 rc=$rc $(date -Is)" >>"$LOG"; break; fi
  echo "attempt $attempt rc=$rc 未成，15s 后续跑" >>"$LOG"
  sleep 15
done
