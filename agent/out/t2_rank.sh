#!/bin/bash
cd /home/wuwenjie01/majiang_ai
for seed in 20260928 1 2 3 4 5 6 7; do
  echo "[rank] seed $seed $(date +%H:%M:%S)" >> agent/out/t2_rank.log
  nice -n 19 uv run python research/t2_seq_vs_agg.py \
    --train 'data/value_seq_train.w*.c*.npz' --valid 'data/value_seq_valid.w*.c*.npz' \
    --epochs 60 --patience 10 --seed $seed --name t2-rank-$seed \
    >> agent/out/t2_rank.log 2>&1
done
echo "[rank] ALL DONE $(date +%H:%M:%S)" >> agent/out/t2_rank.log
