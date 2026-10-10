#!/usr/bin/env bash
# 场地代表性 A/B 的**本机**跑次（远端同时跑另两个种子 ⇒ 合起来 4 种子）。A 2026-10-10。
#
# 用 `nice -n 15`：采集器是 `nice 0` 且窗口只有 600ms，内核会立刻抢占它；
# 本跑只吃闲置核（本机 16 核，采集器用 ~1 核）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=.venv/bin/python
LOG=${LOG:-/tmp/field_ab_local.log}
TREATMENT=${TREATMENT:-v7m-keepchi}
BASELINE=${BASELINE:-v7}
: > "$LOG"

for field in ${FIELDS:-meld-equal}; do
  for seed in ${SEEDS:-20260923 20261011}; do
    echo "=== [本机] treatment=$TREATMENT field=$field seed=$seed 开始 $(date -Is) ===" >> "$LOG"
    nice -n 15 "$PY" tools/ab_test.py --treatment "$TREATMENT" --baseline "$BASELINE" --field "$field" \
      --matches "${MATCHES:-200}" --rounds "${ROUNDS:-8}" --seed "$seed" --jobs "${JOBS:-12}" \
      >> "$LOG" 2>&1
    echo "=== [本机] treatment=$TREATMENT field=$field seed=$seed 结束 $(date -Is) ===" >> "$LOG"
  done
done
echo "=== [本机] 场地 A/B 完成 $(date -Is) ===" >> "$LOG"
