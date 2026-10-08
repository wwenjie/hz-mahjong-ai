#!/bin/bash
# 并行生成 v5 BC 数据：7 个分片 × 100 matches，3 并发，nice-19
# 关键：setsid 让每个分片脱离 exec 会话组，防止被回收
set -u
cd /home/wuwenjie01/majiang_rl2
mkdir -p data/bc_v5_parts logs

run_shard() {
    local shard_id=$1
    local seed=$((20261001 + shard_id))
    local out="data/bc_v5_parts/train_part$(printf %02d $shard_id).npz"
    local log="logs/gen_v5_part$(printf %02d $shard_id).log"
    if [ -s "$out" ]; then
        echo "[shard $shard_id] 已存在，跳过: $out"
        return 0
    fi
    echo "[shard $shard_id] 开始: seed=$seed -> $out (pid=$$)"
    PYTHONPATH=src:/home/wuwenjie01/majiang_ai/src nice -n 19 \
        /home/wuwenjie01/majiang_ai/.venv/bin/python scripts/gen_bc_data_v5.py \
        --matches 100 --rounds 8 --seed "$seed" --out "$out" > "$log" 2>&1
    local rc=$?
    echo "[shard $shard_id] 结束 rc=$rc ($(tail -1 "$log" 2>/dev/null))"
    return $rc
}

export -f run_shard

# 3 并发跑 7 个分片（约 211k 样本）
for i in 0 1 2 3 4 5 6; do
    echo "$i"
done | xargs -P 3 -I {} setsid bash -c 'run_shard "$@"' _ {}

echo "=== 所有分片完成 ==="
ls -la data/bc_v5_parts/
