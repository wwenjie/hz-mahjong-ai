#!/bin/bash
# 补充第 9 分片：100 matches，seed 递增，nice-19；失败自动重试 1 次
set -u
cd /home/wuwenjie01/majiang_rl2
out="data/bc_v5_parts/train_part08.npz"
log="logs/gen_v5_part08.log"
seed=20261009
for attempt in 1 2; do
    if [ -s "$out" ]; then echo "[shard 8] 已存在: $out"; exit 0; fi
    echo "[shard 8] 开始 attempt=$attempt seed=$seed -> $out"
    PYTHONPATH=src:/home/wuwenjie01/majiang_ai/src nice -n 19 \
        /home/wuwenjie01/majiang_ai/.venv/bin/python scripts/gen_bc_data_v5.py \
        --matches 100 --rounds 8 --seed "$seed" --out "$out" > "$log" 2>&1
    rc=$?
    echo "[shard 8] 结束 attempt=$attempt rc=$rc ($(tail -1 "$log" 2>/dev/null))"
    [ $rc -eq 0 ] && exit 0
    seed=$((seed + 1000))
done
exit 1
