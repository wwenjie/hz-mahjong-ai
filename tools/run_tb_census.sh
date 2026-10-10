#!/usr/bin/env bash
# 破平层覆盖主分：全量触发点普查（14 分片，远端 14 核）。A 2026-10-10。
#
# 为什么分片：普查对每个弃牌点要跑 3 次完整决策（v5 / 显式旋钮 v5 / 主分优先），
# 单核 ~2.4s/房，全量 11,733 房单核要 ~8h；14 分片 ⇒ ~35min。
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

PY=/root/miniconda3/bin/python
OUT=agent/out/trigger-points
SHARDS=${SHARDS:-14}
ROOMS=${ROOMS:-0}

for i in $(seq 0 $((SHARDS - 1))); do
  setsid nohup "$PY" tools/trigger_census_tiebreak.py \
    --rooms "$ROOMS" --shards "$SHARDS" --shard "$i" --progress 0 \
    --out "$OUT/tb-shard$i.jsonl" \
    > "/tmp/tb_$i.log" 2>&1 < /dev/null &
  echo "started shard $i pid $!"
done
echo "全部 $SHARDS 片已启动；日志 /tmp/tb_*.log；产物 $OUT/tb-shard*.jsonl"
