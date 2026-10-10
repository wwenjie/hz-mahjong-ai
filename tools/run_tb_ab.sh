#!/usr/bin/env bash
# 破平层收口臂：整场非劣/优门（远端 14 核）。A 2026-10-10。
#
# **只有在条件对拍（`cf-tbcover-slack0`）为正且 t≥2 时才该跑这里**——预登记判据③。
# 4 种子 × 200 场 × 8 局 × 四座位旋转，逐种子先出读数，不合并（先看符号一致性）。
#
# 判读：合并「每场名次分」**正显著（t≥2 且 >0）⇒ 建 v8 快照**；
# 不显著为正但也不显著为负 ⇒ 只在有独立机制证据时才动默认档；
# 显著为负 ⇒ 关闭本轴。
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

PY=/root/miniconda3/bin/python
MATCHES=${MATCHES:-200}
ROUNDS=${ROUNDS:-8}
LOG=/tmp/tb_ab.log
: > "$LOG"

for seed in ${SEEDS:-20261008 771013 20260923 20261011}; do
  echo "=== 种子 $seed 开始 $(date -Is) ===" >> "$LOG"
  $PY tools/ab_test.py --treatment v5-tieslack0 --baseline v5 \
    --matches "$MATCHES" --rounds "$ROUNDS" --seed "$seed" --jobs 14 \
    >> "$LOG" 2>&1
  echo "=== 种子 $seed 结束 $(date -Is) ===" >> "$LOG"
done
echo "=== A/B 全部完成 $(date -Is) ===" >> "$LOG"
