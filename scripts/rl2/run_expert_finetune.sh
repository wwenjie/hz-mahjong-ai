#!/bin/bash
cd /home/wuwenjie01/majiang_rl2
exec env CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/wuwenjie01/majiang_ai/src:src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/wuwenjie01/majiang_ai/.venv/bin/python -u scripts/train_bc_v7.py \
  --train-data data/expert_bc_v7/expert_bc_v7.npz \
  --resume /tmp/bc_v7_base_clean.pt \
  --epochs 5 \
  --lr 1e-5 \
  --batch-size 1024 \
  --out runs/bc_v7_expert_ft.pt \
  >> logs/expert_finetune.log 2>&1
