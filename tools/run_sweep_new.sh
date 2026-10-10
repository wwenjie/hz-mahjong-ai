#!/usr/bin/env bash
# 单旋钮重筛：**112 核新机上的四路并行**（A 2026-10-10 14:40）。用户新给的机器。
#
# 为什么四路：一次 200 场 × 4 旋转 × 8 局 = 1,600 个 match 任务，本机实测 ≈ **4.8 核·秒/match**
# ⇒ 单跑需 ≈ 7,680 核·秒。112 核若只跑一路，利用率只有 1/4（`ab_test` 单进程池 28 任务够吃 28 核），
# 所以开四路 × 28 jobs 把核吃满，总耗时 ≈ 36 跑 × 7680 / 112 ≈ **41 分钟**。
#
# `v7-natural` 故意留在 S1 里当**阴性对照**：它在本机/老远端两场地 4 种子全显著为负
# （−0.491/−0.745/−0.681/−0.75×），若在新机上复现同一符号与量级 ⇒ 跨机同构、harness 可信。
#
# 用法::
#     bash tools/run_sweep_new.sh          # 启动四路（各自日志）
#     bash tools/run_sweep_new.sh --status # 只看进度
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

if [ "${1:-}" = "--status" ]; then
  for g in s1 s2 s3 s4; do
    printf '%s: %s 行 / 完成段 %s\n' "$g" "$(wc -l < /tmp/sweep_$g.log 2>/dev/null || echo 0)" \
      "$(grep -c '^### arm=' /tmp/sweep_$g.log 2>/dev/null || echo 0)"
  done
  exit 0
fi

SEEDS="20261008 771013 20260923 20261011"
COMMON="FIELDS=meld-equal SEEDS=$SEEDS JOBS=28 MATCHES=200 ROUNDS=8 BASELINE=v7"

sets=(
  "s1:v7-cand5 v7-u4 v7-natural"
  "s2:v7-presel5 v7-twoply v7-goodshape"
  "s3:v7-piao05 v7-piao12"
  "s4:v7-edge7"
)
for entry in "${sets[@]}"; do
  name=${entry%%:*}
  arms=${entry#*:}
  ARMS="$arms" LOG="/tmp/sweep_$name.log" setsid nohup env $COMMON ARMS="$arms" \
    bash tools/run_sweep.sh > "/tmp/sweep_$name.out" 2>&1 < /dev/null &
  echo "启动 $name: $arms (pid $!)"
done
echo "四路已启动；日志 /tmp/sweep_s{1,2,3,4}.log"
