#!/bin/bash
# T2 全链自愈脚本：生成（可续跑）→ 分析（重试）。
# 长跑进程会被环境周期性回收，故每一段都放在可重入循环里，产物落盘。
cd /home/wuwenjie01/majiang_ai
log() { echo "[$(date +%H:%M:%S)] $*" >> agent/out/t2_run.log; }

# --- 1) 训练数据（150 分片）---
for a in $(seq 1 500); do
  n=$(ls data/value_seq_train.w*.c*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 150 ]; then log "train shards ok ($n)"; break; fi
  log "train gen attempt $a (have $n/150)"
  nice -n 19 uv run python research/gen_value_seq_data.py \
    --rounds 3000 --workers 4 --chunk 20 --seed 20260928 \
    --out data/value_seq_train.npz >> agent/out/t2_gen_train.log 2>&1
  sleep 2
done

# --- 2) 验证数据（30 分片）---
for a in $(seq 1 500); do
  n=$(ls data/value_seq_valid.w*.c*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 30 ]; then log "valid shards ok ($n)"; break; fi
  log "valid gen attempt $a (have $n/30)"
  nice -n 19 uv run python research/gen_value_seq_data.py \
    --rounds 600 --workers 2 --chunk 20 --seed 777001 \
    --out data/value_seq_valid.npz >> agent/out/t2_gen_valid.log 2>&1
  sleep 2
done

# --- 3) 分析（重试）---
for a in $(seq 1 100); do
  log "analysis attempt $a"
  if nice -n 19 uv run python research/t2_seq_vs_agg.py \
      --train 'data/value_seq_train.w*.c*.npz' --valid 'data/value_seq_valid.w*.c*.npz' \
      > agent/out/t2_analysis.log 2>&1; then
    log "ANALYSIS OK"
    echo "[$(date +%H:%M:%S)] T2 DONE" >> agent/out/t2_analysis.log
    exit 0
  fi
  log "analysis failed rc=$?, retry"
  sleep 5
done
log "ANALYSIS GAVE UP"
exit 1
