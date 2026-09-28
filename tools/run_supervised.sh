#!/usr/bin/env bash
# 通用一次性任务的守护壳：**被回收就重启，正常退出就收工**。
#
# 为什么需要它：本环境会周期性回收长跑进程，实测表现为「日志文件被创建但 0 字节、
# 进程消失、没有 Traceback」——`nohup setsid` 只能防挂断，防不住回收。
# 与 `queue_supervisor.sh` 的区别：那个是**常驻**（跑空后等新任务），
# 这个是**一次性**（子进程以 0 退出即认为任务完成，不再重启）。
#
# 判据刻意选得很保守：只有**退出码非 0** 才重启。这样「任务真的跑完了」与
# 「任务被杀了」能区分开；否则重启会把一个已经成功的任务再做一遍，
# 凭空多出一份互相覆盖的输出。
#
# 用法::
#
#     nohup setsid tools/run_supervised.sh <日志路径> -- <命令...> >> /tmp/sup.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

LOG="${1:?用法: run_supervised.sh <日志路径> -- <命令...>}"
shift
[ "${1:-}" = "--" ] && shift
[ "$#" -gt 0 ] || { echo "缺少命令"; exit 2; }

BACKOFF="${SUPERVISE_BACKOFF:-30}"
attempt=0

echo "$(date -Is) 守护启动：$* -> $LOG"
while :; do
  attempt=$((attempt + 1))
  echo "$(date -Is) 第 $attempt 次启动"
  "$@" >>"$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "$(date -Is) 第 $attempt 次以 0 退出（视为完成），守护收工"
    exit 0
  fi
  echo "$(date -Is) 第 $attempt 次非 0 退出（码 $code），${BACKOFF}s 后重启"
  # 关机信号要能被接住：否则守护退出了、子进程还在跑，变成两个进程写同一个日志。
  sleep "$BACKOFF" || exit 0
done
