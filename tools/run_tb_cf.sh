#!/usr/bin/env bash
# 破平层覆盖主分：条件对拍（远端 14 核）。A 2026-10-10。
#
# 三跑，都用 `--force-trigger`（**定向臂**：只在这一次出牌上用处理臂，之后两分支都用基线
# `v5` 续跑）⇒ 差分只含「这一次换牌」的效应，不含换档累积效应。
#
#   ① `tb-cover` × `v5-tieslack0` —— **本轴的修复臂读数**。触发点定义即「破平层换掉了
#      主分最高者」，故这正是收口所影响的那一面。
#   ② `tb-cover` × `v5-maxtotal` —— 同一点集上的**上界**：把破平层整层关掉。
#      若连上界都非正，则 `slack=0` 不可能靠「覆盖面」赚钱。
#   ③ `tb-tie` × `v5-tieslack0` —— **反例保护**：`gap==0` 的真平局点（这层字面义的作用面，
#      实测占决策 11.9%）上，收口后仍有并列、仍由进张裁决 ⇒ 预期近零效应；
#      若这里显著为负，说明并列层的进张排序本身在亏，那是另一条轴。
#
# 判读（预登记，见 `src/majiang/cli.py` 的 `v5-maxtotal` 注释）：
#   ① 为正且 t≥2 ⇒ 破平层在该面有害、立修复案；显著为负 ⇒ 该面有价值、只在 gap 小的子层收口。
set -uo pipefail
cd /root/autodl-tmp/majiang_ai

PY=/root/miniconda3/bin/python
COMMON="--mode discard --force-trigger --baseline v5 --opponents v5 --jobs 14"

run() {  # name points treatment
  echo "=== $1 开始 $(date -Is) ===" >> /tmp/cf_tbrun.log
  $PY tools/trigger_counterfactual.py --points "agent/out/trigger-points/$2.jsonl" \
    $COMMON --treatment "$3" --out "agent/out/trigger-points/cf-$1.jsonl" \
    > "/tmp/cf_$1.log" 2>&1
  echo "=== $1 结束 $(date -Is) ===" >> /tmp/cf_tbrun.log
  tail -8 "/tmp/cf_$1.log" >> /tmp/cf_tbrun.log
}

run tbcover-slack0  tb-cover v5-tieslack0
run tbcover-maxtotal tb-cover v5-maxtotal
run tbtie-slack0    tb-tie   v5-tieslack0
echo "=== 全部完成 $(date -Is) ===" >> /tmp/cf_tbrun.log
