#!/usr/bin/env bash
# 破平层轴：**本机**版流水线（普查 → 合并分层 → 关键条件对拍）。A 2026-10-10。
#
# 为什么另起本机版：远端 14 核那一批已跑 50+ 分钟未落盘（远端 vCPU 明显比本机慢），
# 而本机 16 核在场地 A/B 跑完后是空的。两条路用**不同房样本**（本机 `--rooms 1200` 抽样、
# 远端全量）⇒ 结果是**互为交叉复核**，不是重复。
#
# `nice -n 15`：采集器是 `nice 0`、响应窗口只有 600ms，内核会立刻抢占它。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=.venv/bin/python
OUT=agent/out/trigger-points
SHARDS=${SHARDS:-10}
ROOMS=${ROOMS:-1200}
LOG=/tmp/tb_local_pipeline.log
: > "$LOG"

echo "=== [本机] 普查启动 $(date -Is) （$SHARDS 片 × 抽样 $ROOMS 房）===" >> "$LOG"
for i in $(seq 0 $((SHARDS - 1))); do
  setsid nohup nice -n 15 "$PY" tools/trigger_census_tiebreak.py \
    --rooms "$ROOMS" --shards "$SHARDS" --shard "$i" --progress 0 \
    --out "$OUT/tbl-shard$i.jsonl" > "/tmp/tbl_$i.log" 2>&1 < /dev/null &
done
# 等分片：**按产物文件数轮询**，不用 `wait`（`setsid` 会让子进程脱离本 shell 的作业表，
# 也不要用 `pgrep -f <脚本名>`——轮询命令自己的命令行会匹配上，导致恒定「还在跑」）。
while [ "$(ls "$OUT"/tbl-shard*.jsonl 2>/dev/null | wc -l)" -lt "$SHARDS" ]; do
  sleep 15
done
echo "=== [本机] 普查分片全部落盘 $(date -Is) ===" >> "$LOG"

"$PY" tools/merge_tb_shards.py --glob "$OUT/tbl-shard*.jsonl" --out-dir "$OUT" \
  --limit-cover 12000 >> "$LOG" 2>&1
echo "=== [本机] 合并完成 $(date -Is) ===" >> "$LOG"

nice -n 15 "$PY" tools/trigger_counterfactual.py --points "$OUT/tb-cover.jsonl" \
  --mode discard --force-trigger --baseline v5 --opponents v5 --treatment v5-tieslack0 \
  --jobs "${JOBS:-12}" --limit 12000 --out "$OUT/cf-tbcover-slack0-local.jsonl" \
  > /tmp/cf_tbcover_slack0_local.log 2>&1
echo "=== [本机] 条件对拍完成 $(date -Is) ===" >> "$LOG"
tail -12 /tmp/cf_tbcover_slack0_local.log >> "$LOG"
echo "=== [本机] 流水线结束 $(date -Is) ===" >> "$LOG"
