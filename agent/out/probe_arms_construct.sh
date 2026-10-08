#!/usr/bin/env bash
set +e
cd /home/wuwenjie01/majiang_ai || exit 1
LOG=agent/out/probe-arms-construct.log
: >"$LOG"
for attempt in 1 2 3 4 5; do
  echo "--- attempt $attempt $(date -Is) ---" >>"$LOG"
  nice -n 19 .venv/bin/python -u agent/verify/probe_arms_construct.py >>"$LOG" 2>&1
  rc=$?
  if grep -q PROBE_ARMS_CONSTRUCT_OK "$LOG"; then echo "完成 rc=$rc $(date -Is)" >>"$LOG"; break; fi
  echo "attempt $attempt rc=$rc 未成，15s 后续跑" >>"$LOG"
  sleep 15
done
