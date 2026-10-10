#!/usr/bin/env bash
# 破平层轴：**关键跑次**条件对拍（远端 14 核）。A 2026-10-10。
#
# 只跑 `tb-cover × v5-tieslack0`（修复臂读数）——它是本轴判定的**决定性**读数，
# 另两跑（`v5-maxtotal` 上界、`tb-tie` 反例保护）排在场地 A/B 之后，有时间再补。
#
# 定向臂：`--force-trigger` 只在这一次出牌上用处理臂，之后两分支都用基线 `v5` 续跑 ⇒
# 差分只含「这一次换牌」的效应。
#
# 触发面 = 全部（`--limit 12000`，由 `merge_tb_shards.py --limit-cover 12000` 洗牌后抽样）。
# 为什么这个 n：B' 用 197 点的 se 是 1.244 ⇒ `n=12,000` 时 se≈0.16、MDE≈0.45，
# 才够分辨「−1.38 是真是假」。
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

PY=/root/miniconda3/bin/python
LIMIT=${LIMIT:-12000}
LOG=/tmp/cf_tbrun.log

run() {
  echo "=== $1 开始 $(date -Is) ===" >> "$LOG"
  $PY tools/trigger_counterfactual.py --points "agent/out/trigger-points/$2.jsonl" \
    --mode discard --force-trigger --baseline v5 --opponents v5 --jobs 14 --limit "$LIMIT" \
    --treatment "$3" --out "agent/out/trigger-points/cf-$1.jsonl" > "/tmp/cf_$1.log" 2>&1
  echo "=== $1 结束 $(date -Is) ===" >> "$LOG"
  tail -10 "/tmp/cf_$1.log" >> "$LOG"
}

run tbcover-slack0 tb-cover v5-tieslack0
