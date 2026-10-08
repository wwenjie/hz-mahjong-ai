#!/bin/bash
# v3a/v3b 双 checkpoint 到达监视器：每个文件出现且大小稳定后自动启动对应 A/B 评测
# 部署目标：AutoDL 53838，nohup 后台跑
set -u
BASE="/root/autodl-tmp/ab/majiang_rl"

wait_stable() {
  local f="$1"
  while [ ! -f "$f" ]; do sleep 30; done
  local prev=0
  while true; do
    local cur
    cur=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$cur" -gt 1000000 ] && [ "$cur" = "$prev" ]; then break; fi
    prev=$cur
    sleep 20
  done
}

run_ab() {
  local tag="$1"   # v3a or v3b
  local ckpt="$BASE/runs/ppo_v7_${tag}.pt"
  local log="$BASE/logs/ab_${tag}.log"
  local out="$BASE/runs/ab_${tag}.json"
  local donef="$BASE/runs/ab_${tag}.done"

  echo "[watcher] $tag: waiting for $ckpt $(date '+%F %T')" >> "$log"
  wait_stable "$ckpt"
  echo "[watcher] $tag: stable, launching A/B $(date '+%F %T')" >> "$log"

  cd "$BASE"
  /root/miniconda3/bin/python -u scripts/run_ab_v7.py \
    --model "$ckpt" --matches 40 --rounds 8 \
    --seeds 20261003,771014 --device cpu --out "$out" >> "$log" 2>&1
  local rc=$?
  echo "[watcher] $tag: A/B exited rc=$rc $(date '+%F %T')" >> "$log"
  touch "$donef"
}

# 串行执行（避免 CPU 争抢；A800 CPU 核数有限，v5 教师搜索推理是 CPU 密集）
run_ab v3a
run_ab v3b
echo "[watcher] all done $(date '+%F %T')" >> "$BASE/logs/ab_v3watcher.log"
