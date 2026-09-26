#!/usr/bin/env bash
# 离线实验队列的守护循环。
#
# 与 `collector_supervisor.sh` 的两处刻意的不同：
#
# 1. **用 `--loop`**：队列跑空后不退出，而是等新 job 被写进 `queue.json` —— 这样
#    「预先登记实验」才成立：写进文件即开工，不必有人在旁边敲命令。
# 2. **崩溃后退避更久**（60s）。它跑的是 CPU 密集的配对自对弈，密集重启只会把
#    真机采集的决策窗口挤到超时。
#
# 安全边界：本脚本只运行 `tools/iterate_loop.py`，后者只调用 `ab_test.py`（离线自对弈），
# **不发任何平台请求**。真机实验必须人工决定。
#
# 用法::
#
#     nohup setsid tools/queue_supervisor.sh >> /tmp/iterate.log 2>&1 < /dev/null &
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

BACKOFF="${MAJIANG_QUEUE_BACKOFF:-60}"
child=0
stopping=0
trap 'stopping=1; [ "$child" -ne 0 ] && kill -TERM "$child" 2>/dev/null' TERM INT

echo "$(date -Is) 实验队列守护启动（只跑离线自对弈，不碰平台）"
while [ "$stopping" -eq 0 ]; do
  uv run python tools/iterate_loop.py --loop &
  child=$!
  wait "$child"
  code=$?
  child=0
  if [ "$stopping" -ne 0 ]; then
    echo "$(date -Is) 收到停机信号，退出（子进程码 $code）"
    break
  fi
  echo "$(date -Is) iterate_loop 退出（码 $code），${BACKOFF}s 后重启"
  sleep "$BACKOFF"
done
