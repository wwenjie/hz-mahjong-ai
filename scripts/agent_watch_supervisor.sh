#!/usr/bin/env bash
# agent_watch 守护循环：无论退出码都重启（周期回收环境的对策，同 A 的 collector_supervisor）。
# 启动：nohup setsid bash scripts/agent_watch_supervisor.sh >/dev/null 2>&1 &
# 停止：kill <pid>（pid 在 verify/out/watch_supervisor.pid；SIGTERM 会转发给子进程）
set -u
cd "$(dirname "$0")/.."

PIDFILE=verify/out/watch_supervisor.pid
mkdir -p verify/out

# 防重复启动：pid 文件存在且进程存活且确为 supervisor 则退出
if [ -f "$PIDFILE" ]; then
    old=$(cat "$PIDFILE" 2>/dev/null || echo "")
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null \
        && grep -q agent_watch_supervisor "/proc/$old/cmdline" 2>/dev/null; then
        echo "agent_watch_supervisor 已在运行 (pid=$old)，退出" >&2
        exit 1
    fi
fi
echo $$ > "$PIDFILE"

child=0
forward() {
    if [ "$child" -ne 0 ]; then kill -TERM "$child" 2>/dev/null; fi
    exit 0
}
trap forward TERM INT

while true; do
    uv run python scripts/agent_watch.py &
    child=$!
    wait "$child"
    code=$?
    echo "[$(date '+%F %T')] agent_watch 退出(code=$code)，3s 后重启" >> verify/out/watch.log
    sleep 3
done
