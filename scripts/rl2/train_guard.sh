#!/usr/bin/env bash
# BC v5 训练守护：死锁自动检测 + 从最新 checkpoint 续训
# 判定死锁：日志文件超过 STALL_SECONDS 秒没有更新，且进程还在 -> 杀掉重启
set -u
cd /home/wuwenjie01/majiang_rl2

PY=/home/wuwenjie01/majiang_ai/.venv/bin/python
LOG=logs/train_bc_v5_resume.log
OUT=runs/bc_v5.pt
STALL_SECONDS=600        # 日志 10 分钟无更新 -> 判定死锁
MAX_RESTARTS=8           # 最多自动重启次数
EPOCHS=30

restart_count=0

latest_ckpt() {
  # 找最新的 bc_v5_epNN.pt（排除 _timeout），没有则用 runs/bc_v5.pt
  local f
  f=$(ls -t runs/bc_v5_ep*.pt 2>/dev/null | grep -v _timeout | head -1)
  if [ -n "$f" ]; then echo "$f"; else echo "$OUT"; fi
}

while [ "$restart_count" -le "$MAX_RESTARTS" ]; do
  ckpt=$(latest_ckpt)
  echo "[$(date '+%F %T')] === 启动/续训 #$restart_count 从 $ckpt ===" >> "$LOG"

  PYTHONPATH=src:/home/wuwenjie01/majiang_ai/src \
    "$PY" -u scripts/train_bc_v3.py \
    --train-data data/bc_v5_train.npz \
    --valid-data data/bc_v5_valid.npz \
    --epochs "$EPOCHS" --batch-size 256 \
    --d-model 128 --nhead 8 --num-layers 4 \
    --lr 1e-3 --resume "$ckpt" \
    --timeout-per-epoch 900 \
    --out "$OUT" >> "$LOG" 2>&1 &
  TRAIN_PID=$!
  echo "[$(date '+%F %T')] 训练 PID=$TRAIN_PID" >> "$LOG"

  stalled=0
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
    if [ -f "$LOG" ]; then
      now=$(date +%s)
      mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
      age=$(( now - mtime ))
      if [ "$age" -gt "$STALL_SECONDS" ]; then
        echo "[$(date '+%F %T')] !! 日志 ${age}s 无更新，判定死锁，杀掉 PID=$TRAIN_PID" >> "$LOG"
        kill -9 "$TRAIN_PID" 2>/dev/null
        stalled=1
        break
      fi
    fi
  done

  wait "$TRAIN_PID" 2>/dev/null
  rc=$?

  if [ "$stalled" -eq 0 ]; then
    echo "[$(date '+%F %T')] 训练进程正常退出 rc=$rc" >> "$LOG"
    # 正常结束（rc=0）或超时保护退出（rc=2）都停
    if [ "$rc" -eq 0 ]; then
      echo "[$(date '+%F %T')] === 训练全部完成 ===" >> "$LOG"
      break
    fi
  fi

  restart_count=$(( restart_count + 1 ))
  echo "[$(date '+%F %T')] 准备第 $restart_count 次重启..." >> "$LOG"
  sleep 10
done

echo "[$(date '+%F %T')] 守护脚本结束，共重启 $restart_count 次" >> "$LOG"
