#!/usr/bin/env bash
# 单旋钮重筛（112 核新机）。A 2026-10-10 14:30。
#
# 每键叠在 `v7` 上、只差一项；**同场配对**四座位旋转；口径 `每场名次分`（与平台同）。
# 输出按 `ARM_SEED` 一段一段追加到 `LOG`，每段自带完整读数（便于事后判读，不依赖汇总）。
#
# 用法（远端 112 核）::
#     ARMS="v7-cand5 v7-presel5 ..." FIELDS=meld-equal SEEDS="20261008 771013 20260923 20261011" \
#     JOBS=24 bash tools/run_sweep.sh
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

PY=/root/miniconda3/bin/python
LOG=${LOG:-/tmp/sweep.log}
MATCHES=${MATCHES:-200}
ROUNDS=${ROUNDS:-8}
JOBS=${JOBS:-24}
SEEDS=${SEEDS:-"20261008 771013 20260923 20261011"}
ARMS=${ARMS:-"v7-cand5 v7-presel5 v7-piao05 v7-piao12 v7-edge7 v7-u4 v7-twoply v7-goodshape v7-natural"}
BASELINE=${BASELINE:-v7}
: > "$LOG"

for field in ${FIELDS:-meld-equal}; do
  for arm in $ARMS; do
    for seed in $SEEDS; do
      echo "### arm=$arm field=$field seed=$seed 开始 $(date -Is)" >> "$LOG"
      $PY tools/ab_test.py --treatment "$arm" --baseline "$BASELINE" --field "$field" \
        --matches "$MATCHES" --rounds "$ROUNDS" --seed "$seed" --jobs "$JOBS" >> "$LOG" 2>&1
    done
  done
done
echo "### 重筛全部完成 $(date -Is)" >> "$LOG"
