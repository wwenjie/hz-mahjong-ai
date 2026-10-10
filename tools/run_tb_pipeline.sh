#!/usr/bin/env bash
# 无人值守流水线（远端 14 核）。A 2026-10-10。
#
#   ① 等破平层普查 14 分片退出 → 合并分层（`tb-cover` 洗牌抽样 12,000）
#   ② 破平层关键对拍：`tb-cover × v5-tieslack0`（决定这条轴怎么判）
#   ③ 场地代表性 A/B：`v7m-keepchi vs v7`，`--field meld-equal`（2 种子）
#
# 为什么串成一条：三步各自 10~50min，中间要等；手工轮询会浪费算力也容易漏
# （2026-10-08 有过「一批后台分片静默消失、日志 0 行」的教训）。
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

PY=/root/miniconda3/bin/python
LOG=/tmp/tb_pipeline.log

echo "=== 流水线启动 $(date -Is) ===" >> "$LOG"
while pgrep -f 'tools/trigger_census_tiebreak.py' > /dev/null; do
  sleep 60
done
echo "=== ① 普查分片全部退出 $(date -Is) ===" >> "$LOG"

$PY tools/merge_tb_shards.py --limit-cover 12000 >> "$LOG" 2>&1
echo "=== ① 合并完成 $(date -Is) ===" >> "$LOG"

LIMIT=12000 bash tools/run_tb_cf.sh >> "$LOG" 2>&1
echo "=== ② 破平层对拍完成 $(date -Is) ===" >> "$LOG"

FIELDS=meld-equal SEEDS="20261008 771013" bash tools/run_field_ab.sh >> "$LOG" 2>&1
echo "=== ③ 场地 A/B 完成 $(date -Is) ===" >> "$LOG"
echo "=== 流水线结束 $(date -Is) ===" >> "$LOG"
