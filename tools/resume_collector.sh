#!/usr/bin/env bash
# **开机一键恢复采集器 + 自检**（2026-10-09 17:40 A 留；比赛 10-10 19:00）。
#
# 背景：平台只能从**公司内网**访问 ⇒ 采集器必须留在本机；本机关机 = 我们停止对局。
# 本脚本幂等、可重复跑：
#   1. 采集守护未在跑 ⇒ 用**冠军档 v7** 启动（脚本默认值已是 v7，无需环境变量）；
#   2. 自检：进程签名 + 最近日志的 decider 签名 + 延迟护栏（`tools/check_latency.py`）；
#   3. 打印回退口令。
#
# 用法::
#     bash tools/resume_collector.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "== 1) 启动采集器 =="
if ps -eo cmd | grep -q '[c]ollector_supervisor.sh'; then
  echo "  已在运行（跳过启动）"
else
  # `.env` 含平台地址等配置；缺它 auto_session 连不上内网服务。
  if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
  setsid nohup tools/collector_supervisor.sh >> /tmp/autoloop.log 2>&1 < /dev/null &
  sleep 25
  echo "  已启动（日志 /tmp/autoloop.log）"
fi

echo
echo "== 2) 自检 =="
echo -n "  进程签名："
ps -eo cmd | grep '[a]uto_session.py' | head -1 | grep -o -- '--decider [^ ]*' || echo "（未找到 auto_session！）"
LATEST=$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
  echo "  最近日志：$LATEST"
  echo -n "  decider 签名："
  grep -o '"decider": *"[^"]*"' "$LATEST" 2>/dev/null | tail -1 | cut -c1-90 || echo "（尚无 decision.made）"
fi
echo "  延迟护栏："
"$PY" tools/check_latency.py || echo "  （check_latency 失败，可手工看 logs/*.jsonl 的 elapsed_ms）"

echo
echo "== 3) 回退口令（仅当上一节「不通过」）=="
echo "  kill \$(pgrep -f 'collector_supervisor.sh')"
echo "  cd ~/majiang_ai && MAJIANG_COLLECT_DECIDERS=v5 setsid nohup tools/collector_supervisor.sh >> /tmp/autoloop.log 2>&1 < /dev/null &"
echo
echo "  v7 采纳依据：src/majiang/strategy/versions.py 的 v7 note（定义性 35,143/0 不一致 + 触发点 +1.445(t 5.46) + 4 种子非劣）。"
