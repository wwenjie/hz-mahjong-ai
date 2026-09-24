#!/usr/bin/env bash
# 进程守护（tasks.md 7.5）。
#
# 比赛期间程序必须持续在线——平台的在线判据是「最近 90 秒内有已认证请求」，进程死掉即
# 让位。本脚本的目的只有一个：**进程崩了立刻拉起来**。
#
# 两条刻意的策略：
#
# 1. **只在非零退出时重启**。``Runtime.run`` 在赛事进入终态时会正常返回（退出码 0），
#    此时重启只会重新走注册流程，而赛事已关闭会得到 ``TOURNAMENT_CLOSED``（永久性错误），
#    于是变成「退出→重启→再退出」的空转。因此退出码 0 视为正常收工，跳出循环。
# 2. **连续快速失败即熔断**。配置类错误（令牌写错、服务器地址不对）会让进程秒退，若无
#    限制就变成无限重启、持续打平台请求。因此「启动后 MIN_UPTIME 秒内就退出」连续达到
#    MAX_FAST_FAIL 次即停止守护并要求人工介入。这与 systemd 单元的
#    StartLimitIntervalSec/StartLimitBurst 是同一套语义。
#
# 用法::
#
#     export MAJIANG_TOKEN_QINGLONG=...   # 令牌（按前缀收集，全部经环境变量）
#     ./scripts/supervise.sh
#
# 可用环境变量覆盖：MAJIANG_ENV_PREFIX / MAJIANG_DECIDER / MAJIANG_MODE /
# MAJIANG_BACKOFF / MAJIANG_MAX_FAST_FAIL / MAJIANG_MIN_UPTIME /
# MAJIANG_LOG_DIR / MAJIANG_EXTRA_ARGS
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PREFIX="${MAJIANG_ENV_PREFIX:-MAJIANG_TOKEN_}"
DECIDER="${MAJIANG_DECIDER:-heuristic}"
MODE="${MAJIANG_MODE:-qualifier}"
BACKOFF="${MAJIANG_BACKOFF:-3}"
MAX_FAST_FAIL="${MAJIANG_MAX_FAST_FAIL:-5}"
MIN_UPTIME="${MAJIANG_MIN_UPTIME:-60}"
LOG_DIR="${MAJIANG_LOG_DIR:-logs}"
EXTRA="${MAJIANG_EXTRA_ARGS:-}"
SUPERVISOR_LOG="${LOG_DIR}/supervisor.log"

mkdir -p "$LOG_DIR"

stopping=0
child=0
fast_fails=0
note() { echo "$(date -Is) $*" | tee -a "$SUPERVISOR_LOG"; }

# 转发信号给子进程：cli.py 已注册 SIGTERM/SIGINT 做优雅收尾（提交中的动作不会被截断）
trap 'stopping=1; if [ "$child" -ne 0 ]; then kill -TERM "$child" 2>/dev/null; fi' TERM INT

if ! command -v uv >/dev/null 2>&1; then
  note "找不到 uv，请先安装 uv 或改用 python -m majiang"
  exit 1
fi

note "守护启动：decider=$DECIDER mode=$MODE 令牌前缀=$PREFIX 退避=${BACKOFF}s 日志=$LOG_DIR"

while [ "$stopping" -eq 0 ]; do
  note "拉起进程"
  started=$(date +%s)
  # shellcheck disable=SC2086
  uv run python -m majiang \
    --env-prefix "$PREFIX" \
    --decider "$DECIDER" \
    --mode "$MODE" \
    --log-dir "$LOG_DIR" \
    $EXTRA &
  child=$!
  wait "$child"
  code=$?
  child=0
  uptime=$(( $(date +%s) - started ))

  if [ "$stopping" -ne 0 ]; then
    note "已停止（退出码 $code，存活 ${uptime}s）"
    break
  fi
  if [ "$code" -eq 0 ]; then
    note "进程正常收工（赛事终态，存活 ${uptime}s），守护退出"
    break
  fi

  if [ "$uptime" -lt "$MIN_UPTIME" ]; then
    fast_fails=$(( fast_fails + 1 ))
  else
    fast_fails=0
  fi
  if [ "$fast_fails" -ge "$MAX_FAST_FAIL" ]; then
    note "连续 ${fast_fails} 次启动后 ${MIN_UPTIME}s 内即退出，判定为配置错误，守护停止"
    note "请检查日志与令牌后手动重启（这是刻意熔断，避免持续打平台请求）"
    exit 1
  fi

  note "进程异常退出（退出码 $code，存活 ${uptime}s），${BACKOFF}s 后重启（快速失败 ${fast_fails}/${MAX_FAST_FAIL}）"
  sleep "$BACKOFF"
done
