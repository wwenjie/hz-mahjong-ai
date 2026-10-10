#!/usr/bin/env bash
# 单旋钮重筛（新机 `connect.nma1.seetacloud.com:17407`）。A 2026-10-10 14:55。
#
# **新机的真相（重要，别再踩）**：`nproc` 报 **112**，但 cgroup 配额是
# `cpu.cfs_quota_us=1400000 / period=100000` ⇒ **只有 14 核可用**（`/sys/fs/cgroup/cpu.max` 不存在，
# 必须看老的 `cpu.cfs_quota_us`；`cpu.max` 缺失会让人误以为"112 核随便用"）。
# **实测后果**：14:20 我用 `--jobs 40/50` 跑了两个计时任务，92 个进程抢 14 核 ⇒ 20+ 分钟没跑完
# （那是**配额**不是性能）。⇒ 本脚本一律 **`JOBS=13`、单路**（一路就能吃满 13/14 核，
# 多路只会互相拖慢且不增加吞吐）。
#
# **成本口径**：一次 `--matches M --field <≠baseline>` = `4M`(baseline) + `4M`(treatment) = 8M 个
# match 任务，每个 **≈4.8 核·秒** ⇒ `--matches 120` ⇒ 960 任务 ≈ 4,600 核·秒 ≈ **5.9 分钟/跑**（13 核）。
# 6 臂 × 2 种子 = 12 跑 ≈ **71 分钟**。
#
# **为什么 `--field meld-equal`**：本日三次证明结论随基座/场地/口径反转，故统一到
# **代表场地**（副露 1.56/局/座 ≈ 真机的 1.4×）。**代价**：不走 `same_field` 快捷路径（8M 而非 5M 任务）。
#
# 用法::
#     bash tools/run_sweep_new.sh            # 启动
#     bash tools/run_sweep_new.sh --status   # 进度
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

LOG=/tmp/sweep.log
if [ "${1:-}" = "--status" ]; then
  printf '已完成段数: %s\n' "$(grep -c '^### arm=' "$LOG" 2>/dev/null || echo 0)"
  tail -3 "$LOG" 2>/dev/null
  exit 0
fi

ARMS=${ARMS:-"v7-cand5 v7-piao05 v7-piao12 v7-edge7 v7-u4 v7-presel5"}
SEEDS=${SEEDS:-"20261008 771013"}

ARMS="$ARMS" SEEDS="$SEEDS" FIELDS="${FIELDS:-meld-equal}" JOBS=13 MATCHES="${MATCHES:-120}" \
  ROUNDS=8 BASELINE=v7 \
  setsid nohup bash tools/run_sweep.sh > /tmp/sweep.out 2>&1 < /dev/null &
echo "已启动：ARMS=$ARMS SEEDS=$SEEDS JOBS=13 MATCHES=${MATCHES:-120} 场地=${FIELDS:-meld-equal}"
echo "日志 $LOG（进度：bash tools/run_sweep_new.sh --status）"
