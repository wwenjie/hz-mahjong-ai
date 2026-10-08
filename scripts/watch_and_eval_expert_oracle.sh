#!/bin/bash
# expert_ft + Oracle 双评测 watcher：串行跑两个模型
set -u
BASE="/root/autodl-tmp/ab/majiang_rl"

run_ab() {
  local tag="$1"   # expert_ft or oracle
  local ckpt="$BASE/runs/${tag}.pt"
  local log="$BASE/logs/ab_${tag}.log"
  local out="$BASE/runs/ab_${tag}.json"
  local donef="$BASE/runs/ab_${tag}.done"

  echo "[watcher] $tag: waiting for $ckpt $(date '+%F %T')" >> "$log"
  while [ ! -f "$ckpt" ]; do sleep 30; done
  local prev=0
  while true; do
    local cur
    cur=$(stat -c %s "$ckpt" 2>/dev/null || echo 0)
    if [ "$cur" -gt 1000000 ] && [ "$cur" = "$prev" ]; then break; fi
    prev=$cur
    sleep 20
  done
  echo "[watcher] $tag: stable, launching A/B $(date '+%F %T')" >> "$log"

  cd "$BASE"
  /root/miniconda3/bin/python -u scripts/run_ab_v7.py \
    --model "$ckpt" --matches 40 --rounds 8 \
    --seeds 20261003,771014 --device cpu --out "$out" >> "$log" 2>&1
  local rc=$?
  echo "[watcher] $tag: A/B exited rc=$rc $(date '+%F %T')" >> "$log"
  touch "$donef"
}

# 串行：先 expert_ft，后 oracle
run_ab bc_v7_expert_ft
run_ab oracle_v7
echo "[watcher] all done $(date '+%F %T')" >> "$BASE/logs/ab_expert_oracle_watcher.log"
