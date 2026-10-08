#!/bin/bash
# 大规模 BC 数据生成 + 训练 + A/B 对拍一体化脚本
# 用法: bash scripts/run_full_pipeline.sh
set -e
cd /home/wuwenjie01/majiang_rl2

export PYTHONPATH=src:/home/wuwenjie01/majiang_ai/src
PY=/home/wuwenjie01/majiang_ai/.venv/bin/python
LOG=runs/pipeline.log
mkdir -p runs records data

echo "=== Pipeline 启动 $(date) ===" | tee -a $LOG

# Step 1: 生成更多 BC 数据（如果当前数据 < 30k 样本）
CURRENT_SAMPLES=$($PY -c "import numpy as np; print(len(np.load('data/bc_train.npz')['y']))" 2>/dev/null || echo "0")
echo "当前训练样本: $CURRENT_SAMPLES" | tee -a $LOG

if [ "$CURRENT_SAMPLES" -lt 30000 ]; then
    echo "=== 生成更多 BC 数据 ===" | tee -a $LOG
    # 用串行版（更稳定）；50 场约需 ~5 分钟
    $PY scripts/gen_bc_data.py --train-matches 200 --valid-matches 30 --rounds 8 --train-seed 888 --valid-seed 999 >> $LOG 2>&1
    echo "数据生成完成 $(date)" | tee -a $LOG
    ls -la data/ | tee -a $LOG
fi

# Step 2: 重新训练 BC（用全部数据，train 集自制 valid）
echo "=== BC 训练 ===" | tee -a $LOG
$PY scripts/train_bc.py \
    --epochs 30 --batch-size 64 \
    --d-model 128 --nhead 8 --num-layers 4 \
    --lr 1e-3 \
    --out runs/bc_v1.pt >> $LOG 2>&1
echo "BC 训练完成 $(date)" | tee -a $LOG

# Step 3: A/B 对拍
echo "=== A/B 对拍 ===" | tee -a $LOG
$PY scripts/run_ab.py --model runs/bc_v1.pt --matches 20 --rounds 8 --seed 20260928 >> $LOG 2>&1
echo "A/B 对拍完成 $(date)" | tee -a $LOG

echo "=== Pipeline 全部完成 $(date) ===" | tee -a $LOG
tail -50 $LOG
