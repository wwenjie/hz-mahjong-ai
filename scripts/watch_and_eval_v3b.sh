#!/bin/bash
# v3b checkpoint 到达监视器：文件出现且大小稳定后自动启动 A/B 评测
# 部署目标：AutoDL 53838，nohup 后台跑
set -u
CKPT="/root/autodl-tmp/ab/majiang_rl/runs/ppo_v7_v3b.pt"
LOG="/root/autodl-tmp/ab/majiang_rl/logs/ab_v3b.log"
OUT="/root/autodl-tmp/ab/majiang_rl/runs/ab_v3b.json"
DONE="/root/autodl-tmp/ab/majiang_rl/runs/ab_v3b.done"

echo "[watcher] start $(date '+%F %T'), waiting for $CKPT" >> "$LOG"

# 等文件出现
while [ ! -f "$CKPT" ]; do sleep 30; done
echo "[watcher] file appeared $(date '+%F %T')" >> "$LOG"

# 等大小稳定（传输中不写半成品）
prev=0
while true; do
  cur=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
  if [ "$cur" -gt 1000000 ] && [ "$cur" = "$prev" ]; then break; fi
  prev=$cur
  sleep 20
done
echo "[watcher] size stable ($cur bytes), launching A/B $(date '+%F %T')" >> "$LOG"

cd /root/autodl-tmp/ab/majiang_rl
/root/miniconda3/bin/python -u scripts/run_ab_v7.py \
  --model "$CKPT" --matches 40 --rounds 8 \
  --seeds 20261003,771014 --device cpu --out "$OUT" >> "$LOG" 2>&1
rc=$?

echo "[watcher] A/B exited rc=$rc $(date '+%F %T')" >> "$LOG"
touch "$DONE"
