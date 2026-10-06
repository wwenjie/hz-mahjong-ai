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
# 并行度。**核预算 = WORKERS × JOBS，默认 2×5 = 10 核（本机 16 核）**，余量留给真机采集
# 的 600 ms 碰吃窗口与系统本身。子进程一律 `nice -n 15`、采集是 `nice 0`，所以真机随时可抢占，
# 但真被压满时抢占带来的抖动仍可能让决策算不完 600 ms —— 所以留余量，别顶满。
#
# 变更记录（2026-09-29）：原先 WORKERS=3 且 ab_test **单线程**，实测 CPU 用不到 1/3。
# `tools/ab_test.py` 多进程化后（`--jobs`，并行与串行输出逐位相同），改成「少 job × 多进程」——
# 同样的核数下进程都是同粒度任务、调度更平，且 job 少了以后单 job 的墙钟时间不再被 job 数拖慢。
# 变更记录（2026-10-03 A 22:25 裁决）：默认 JOBS 5→4（2×4=8 核）。真机采集侧出现
# action.rejected / 600ms 决策超预算，归因为离线队列把核顶太满。降到 8 核给真机留更多余量，
# 按小时复核 action.rejected 是否下降（判据命令见 THREAD 记录）。
WORKERS="${MAJIANG_QUEUE_WORKERS:-2}"
JOBS="${MAJIANG_QUEUE_JOBS:-4}"
child=0
stopping=0
trap 'stopping=1; [ "$child" -ne 0 ] && kill -TERM "$child" 2>/dev/null' TERM INT

echo "$(date -Is) 实验队列守护启动（只跑离线自对弈，不碰平台）"
while [ "$stopping" -eq 0 ]; do
  # `--timeout`：**单 job 的墙钟上限**。默认 14400s（4h）是照「v5-vs-v5 场」定的（一场约 2 秒）；
# 但**换慢 field 后必须抬高**——`botlike` 场另三座每决策都跑 `_score_discard`（含精确进张）⇒ 实测 **≈5s/决策**、
# 一场 8 局要几分钟 ⇒ `matches=40` 需 4h+，**正好卡死 14400s**（2026-10-06 因此白烧两批：120 场的 4 条 + 40 场的 2 条）。
# 现设 21600s（6h）= 实测所需 ~4h 的 1.5 倍余量。
TIMEOUT="${MAJIANG_QUEUE_TIMEOUT:-21600}"
uv run python tools/iterate_loop.py --loop --workers "$WORKERS" --jobs "$JOBS" --timeout "$TIMEOUT" &
  child=$!
  wait "$child"
  code=$?
  child=0
  if [ "$stopping" -ne 0 ]; then
    echo "$(date -Is) 收到停机信号，退出（子进程码 $code）"
    exit 0
  fi
  echo "$(date -Is) iterate_loop 退出（码 $code），${BACKOFF}s 后重启"
  sleep "$BACKOFF"
done
