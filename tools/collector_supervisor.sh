#!/usr/bin/env bash
# 采集进程的守护循环。
#
# **为什么需要它**：以任何方式启动的长跑进程在这个环境里都会被周期性回收
# （实测记录：01:51 被杀、03:05 收到 SIGTERM 优雅退出；日志里 0 个 Traceback、
# dmesg 无 OOM，所以不是程序崩了，是外部清理）。没有守护就会静默断采，
# 而赛前数据是「越早采越好、错过就没了」。
#
# 两条与 `scripts/supervise.sh`（那是给正式赛事用的）不同的策略：
#
# 1. **无论退出码都重启**。采集是幂等的、没有「赛事终态」这回事；`auto_session.py`
#    的 `--sessions 0` 本来就是无限循环，它退出即意味着需要重来。
# 2. **决策器轮换由账本恢复**（`prior_sessions()`），所以重启不会让交错 A/B
#    偏向轮换表的第一个档位。
#
# 用法::
#
#     set -a; . ./.env; set +a
#     nohup setsid tools/collector_supervisor.sh >> /tmp/autoloop.log 2>&1 < /dev/null &
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# **默认值必须是冠军臂**（2026-10-07 事故后的系统性修复）：
# 原来是 `heuristic,meld-equal`（legacy 档位 + 一个已判负的档位）。机器/WSL 重启后（实测 10-07 09:41 与 10-06 ~23:18 各一次），
# **常驻服务（如 C 的 `openclaw-gateway.service`，`Restart=always`）会自动回来并重新拉起采集器**——
# 若拉起时没带环境变量，就会**静默跑 legacy 档位 14 小时**（10-06 23:18→10-07 13:39，污染平台战绩）。
# ⇒ 把默认值改成本身就是冠军的臂，**任何来源的重启都安全**。
#
# **2026-10-09 14:30 换档 v5 → v7**（用户 14:25 批准）：`v7` = v5 + `meld_chi_best=True`
# （修「多种吃法共用 `combinations()[0]`」的**度量错**）。采纳依据（全部写进 `versions.py` 的 v7 note）：
# ① 定义性正确：真机 35,143 个可判定点上「选中的吃法全是最小 `after_shanten`」，不一致 0；
# ② 触发点条件对拍：2,445 点 **+1.445 净分/触发（t +5.46）**；
# ③ 整场非劣门：4 种子 `每场名次分` 合并 **−0.0045（t≈−0.35）**、不显著为负 ⇒ 过门；
# ④ 门 3：与 v5 在真机响应窗口分歧 3.0%（方向单一）。
# **护栏（v5 换档时写下的承诺照旧生效）**：换档后立即实测真机 `elapsed_ms`；
# **p99 > 1500ms 或出现任何超预算事件 ⇒ 立刻回退**（把下面默认值改回 `v5` 或临时 `MAJIANG_COLLECT_DECIDERS=v5` 重启）。
DECIDERS="${MAJIANG_COLLECT_DECIDERS:-v7}"
CAP="${MAJIANG_COLLECT_SESSION_CAP:-10800}"
INTERVAL="${MAJIANG_COLLECT_HARVEST_INTERVAL:-10}"
OUT="${MAJIANG_COLLECT_OUT:-data/auto_sessions}"
BACKOFF="${MAJIANG_COLLECT_BACKOFF:-20}"
# 给对照臂设会话数上限（形如 `first-legal=8`），跑到量自动停用该臂。
# 无人值守必需：否则对照臂会一直占掉一半样本，而人不在、没人把它从轮换表里删掉。
ARM_LIMIT="${MAJIANG_COLLECT_ARM_LIMIT:-}"

child=0
stopping=0
# 收到停机信号：转发给子进程（它有 SIGTERM 处理器，会打完当前会话再退），然后自己也退，
# 否则父进程被强杀会让子进程变成孤儿而继续跑。
trap 'stopping=1; [ "$child" -ne 0 ] && kill -TERM "$child" 2>/dev/null' TERM INT

echo "$(date -Is) 采集守护启动：deciders=$DECIDERS 每会话上限=${CAP}s 采集间隔=${INTERVAL}s"
while [ "$stopping" -eq 0 ]; do
  uv run python tools/auto_session.py \
    --sessions 0 \
    --session-cap "$CAP" \
    --harvest-interval "$INTERVAL" \
    --decider "$DECIDERS" \
    ${ARM_LIMIT:+--arm-limit "$ARM_LIMIT"} \
    --out "$OUT" &
  child=$!
  wait "$child"
  code=$?
  child=0
  if [ "$stopping" -ne 0 ]; then
    echo "$(date -Is) 收到停机信号，退出（子进程码 $code）"
    exit 0
  fi
  echo "$(date -Is) auto_session 退出（码 $code），${BACKOFF}s 后重启"
  sleep "$BACKOFF"
done
