#!/bin/bash
# v3b 最终版 + Oracle Guiding 最终版 双 checkpoint watcher
# 等文件落盘且大小稳定后串行启动 A/B 评测（40场×8局×2种子 vs 3×v5）
# 部署：AutoDL 53838
set -u
BASE="/root/autodl-tmp/ab/majiang_rl"
mkdir -p "$BASE/logs"

wait_stable() {
  local f="$1"
  while [ ! -f "$f" ]; do sleep 60; done
  local prev=0
  while true; do
    local cur
    cur=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$cur" -gt 1000000 ] && [ "$cur" = "$prev" ]; then break; fi
    prev=$cur
    sleep 30
  done
}

run_ab() {
  local tag="$1"
  local ckpt="$2"
  local log="$BASE/logs/ab_${tag}.log"
  local out="$BASE/runs/ab_${tag}.json"
  local donef="$BASE/runs/ab_${tag}.done"

  # 已跑过则跳过（幂等，重启 watcher 安全）
  if [ -f "$donef" ]; then
    echo "[watcher] $tag: already done, skip" >> "$log"
    return 0
  fi

  echo "[watcher] $tag: waiting for $ckpt $(date '+%F %T')" >> "$log"
  wait_stable "$ckpt"
  echo "[watcher] $tag: stable ($(stat -c %s "$ckpt") bytes), launching A/B $(date '+%F %T')" >> "$log"

  cd "$BASE"
  /root/miniconda3/bin/python -u scripts/run_ab_v7.py \
    --model "$ckpt" --matches 40 --rounds 8 \
    --seeds 20261003,771014 --device cpu --out "$out" >> "$log" 2>&1
  local rc=$?
  echo "[watcher] $tag: A/B exited rc=$rc $(date '+%F %T')" >> "$log"
  touch "$donef"
}

# 串行：v3b 最终版先（预计 ~01:55 落盘），Oracle Guiding 后（预计 ~03:30-04:00）
run_ab v3b_final     "$BASE/runs/ppo_v7_v3b_final.pt"
run_ab oracle_guided "$BASE/runs/ppo_v7_oracle_guided.pt"
echo "[watcher] all done $(date '+%F %T')" >> "$BASE/logs/ab_final_watcher.log"
