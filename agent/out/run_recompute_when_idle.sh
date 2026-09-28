#!/usr/bin/env bash
# agent-c：等链与 RL 对拍都退出、负载降到安静窗口后，跑独立复算。
#
# 为什么这样写（见 mahjong-durable-long-jobs）：本机会周期性回收长跑进程，
# 所以完成路径交给「分块幂等落盘 + 外层重冲 + 调度器收口」，不用前台等待。
# 复算脚本自身按块（arm@seed）原子落盘，重跑时已存在的块直接跳过。
set +e
cd /home/wuwenjie01/majiang_ai || exit 1
PY=/home/wuwenjie01/majiang_ai/.venv/bin/python
LOG=agent/out/recompute-runner.log
STATUS=agent/out/recompute-runner.status
OUT=agent/out/recompute-search-v3.json
mkdir -p agent/out/recompute-chunks
rm -f "$STATUS"

{
  echo "=== 等安静窗口 $(date -Is) ==="
  for i in $(seq 1 480); do            # 最长约 4 小时
    chain=0; after=0; rl=0
    pgrep -f '[c]hain_search_v3\.py' >/dev/null 2>&1 && chain=1
    pgrep -f '[a]fter_chain_rl_v3\.sh' >/dev/null 2>&1 && after=1
    pgrep -f '[r]un_ab\.py' >/dev/null 2>&1 && rl=1
    load=$(cut -d' ' -f1 /proc/loadavg)
    quiet=$(awk -v l="$load" 'BEGIN{print (l<4.0)?1:0}')
    if [ "$chain" = 0 ] && [ "$after" = 0 ] && [ "$rl" = 0 ] && [ "$quiet" = 1 ]; then
      echo "安静窗口到（轮 $i，load=$load）"; break
    fi
    [ $((i % 10)) -eq 0 ] && echo "  等待中（轮 $i，load=$load chain=$chain after=$after rl=$rl）"
    sleep 60
  done

  echo "=== 开跑复算 $(date -Is) ==="
  for attempt in 1 2 3 4 5 6 7 8; do   # 自愈：被杀就重跑，已落块自动跳过
    echo "--- attempt $attempt $(date -Is) ---"
    nice -n 19 "$PY" agent/verify/recompute_search_v3.py \
        --arms search-v3,search-deep-v3 --seeds 20260928,771014 \
        --matches 40 --workers 8 --identity \
        --out "$OUT"
    rc=$?
    [ -f "$OUT" ] && { echo "完成 rc=$rc $(date -Is)"; break; }
    echo "attempt $attempt rc=$rc 未成，20s 后续跑"
    sleep 20
  done
} >>"$LOG" 2>&1

echo "DONE $(date -Is)" >"$STATUS"
