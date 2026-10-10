#!/usr/bin/env bash
# 场地代表性假设：**放宽吃碰闸门，在一个「像真机那样副露」的场地上是否成立？**（远端 14 核）
#
# **为什么重开这条已被关闭的轴**（A 2026-10-10 11:10，有新证据才重开）：
# ① `tools/meld_rate_census.py`（真机 1,200 房 / 9,584 局，同批局内对比）：
#    **我方副露 0.623/局/座，三家各 1.113/局/座（1.79×）** ⇒ 真机场地副露率是我们的 1.79 倍。
# ② `tools/selfplay_meld_rate.py`：`v7` 自对弈 0.979/局/座；`meld-equal` 1.562；`v7m` 1.583。
# ③ ⇒ `ab_test` 默认场地（三座 baseline）副露只有真机场地的 ~56%（0.979→相对 1.113 仍偏低，
#    而真机我方实测只 0.623）⇒ **当年「放宽闸门」的关闭结论是在一个不副露的场地上得到的**，
#    外部效度有洞。这正是本项目已出现两次的「结论随基座/场地反转」模式。
#
# 设计：`--field` 把另三座换成**副露更多的臂**，被测臂仍只坐旋转座；
# 两臂在同一场同一副牌里配对（`ab_test` 的四座位旋转），差分可归因。
#   treatment = `v7m-keepchi`（v7 + 吃法按 `action.tiles` 算 + 同向听吃法次排序 + `equal` 容差）
#   baseline  = `v7`（冠军档，STRICT 闸门）
#   field     = `meld-equal`（正常策略、副露 1.56/局/座 ≈ 真机的 1.4×，压力场）
#
# 判读（预登记）：两场地下**同为「正且 t≥2」⇒ 提出换档（赛前须用户拍板，风险项）**；
# 任一为负 ⇒ 关闭「场地代表性」假设，冠军档不动。
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

PY=/root/miniconda3/bin/python
MATCHES=${MATCHES:-200}
ROUNDS=${ROUNDS:-8}
LOG=/tmp/field_ab.log
: > "$LOG"

for field in ${FIELDS:-meld-equal natural}; do
  for seed in ${SEEDS:-20261008 771013}; do
    echo "=== field=$field seed=$seed 开始 $(date -Is) ===" >> "$LOG"
    $PY tools/ab_test.py --treatment v7m-keepchi --baseline v7 --field "$field" \
      --matches "$MATCHES" --rounds "$ROUNDS" --seed "$seed" --jobs 14 \
      >> "$LOG" 2>&1
    echo "=== field=$field seed=$seed 结束 $(date -Is) ===" >> "$LOG"
  done
done
echo "=== 场地代表性 A/B 全部完成 $(date -Is) ===" >> "$LOG"
