#!/usr/bin/env bash
# agent-c：等 agent-d 的三臂对拍产物落盘、且负载安静后，独立复核（第二层复算）。
#
# 依赖顺序（串行，避免与 search-v3 复算抢核）：
#   1) agent-d 的 `majiang_rl/records/ab-arms-vs-v3.json` 存在
#   2) 我的 search-v3 复算主产物 `agent/out/recompute-search-v3.json` 已存在
#   3) 安静窗口：链 / after_chain / run_ab / queue_arms 全退 且 load<4
# 完成后写 `agent/out/recompute-arms.status`，由 `agent-c-arms-watch` 唤醒收尾。
set +e
cd /home/wuwenjie01/majiang_ai || exit 1
PY=/home/wuwenjie01/majiang_ai/.venv/bin/python
LOG=agent/out/recompute-arms-runner.log
STATUS=agent/out/recompute-arms.status
OUT=agent/out/recompute-arms.json
CHUNKS=agent/out/recompute-arms-chunks
REF=/home/wuwenjie01/majiang_rl/records/ab-arms-vs-v3.json
mkdir -p "$CHUNKS"
rm -f "$STATUS"

{
  echo "=== 等 agent-d 三臂产物 + search-v3 复算 + 安静窗口 $(date -Is) ==="
  for i in $(seq 1 720); do            # 最长约 6 小时
    chain=0; after=0; rl=0; qarms=0
    pgrep -f '[c]hain_search_v3\.py' >/dev/null 2>&1 && chain=1
    pgrep -f '[a]fter_chain_rl_v3\.sh' >/dev/null 2>&1 && after=1
    pgrep -f '[r]un_ab\.py' >/dev/null 2>&1 && rl=1
    pgrep -f '[q]ueue_arms_vs_v3\.sh' >/dev/null 2>&1 && qarms=1
    load=$(cut -d' ' -f1 /proc/loadavg)
    quiet=$(awk -v l="$load" 'BEGIN{print (l<4.0)?1:0}')
    ref_ok=0; [ -f "$REF" ] && ref_ok=1
    sv3_ok=0; [ -f agent/out/recompute-search-v3.json ] && sv3_ok=1
    if [ "$chain" = 0 ] && [ "$after" = 0 ] && [ "$rl" = 0 ] && [ "$qarms" = 0 ] \
       && [ "$quiet" = 1 ] && [ "$ref_ok" = 1 ] && [ "$sv3_ok" = 1 ]; then
      echo "条件齐（轮 $i，load=$load ref=$ref_ok sv3=$sv3_ok）"; break
    fi
    [ $((i % 15)) -eq 0 ] && echo "  等待中（轮 $i，load=$load chain=$chain after=$after rl=$rl qarms=$qarms ref=$ref_ok sv3=$sv3_ok）"
    sleep 30
  done

  echo "=== 开跑三臂复算 $(date -Is) ==="
  for attempt in 1 2 3 4 5 6 7 8; do   # 自愈：被杀就重跑，已落块自动跳过
    echo "--- attempt $attempt $(date -Is) ---"
    nice -n 19 "$PY" agent/verify/recompute_search_v3.py \
        --arms rl,mlp-value,policy-bc --seeds 20260928,771014 \
        --matches 20 --rounds 8 --workers 6 \
        --chunks "$CHUNKS" --ref "$REF" --out "$OUT"
    rc=$?
    [ -f "$OUT" ] && { echo "完成 rc=$rc $(date -Is)"; break; }
    echo "attempt $attempt rc=$rc 未成，30s 后续跑"
    sleep 30
  done
} >>"$LOG" 2>&1

echo "DONE $(date -Is)" >"$STATUS"
