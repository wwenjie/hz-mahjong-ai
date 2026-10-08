#!/usr/bin/env bash
# Q2 状态检查（供 cron trigger）：输出 v5runner 完成标记 + v6 臂到听数估计
set -u
cd /home/wuwenjie01/majiang_ai
if grep -q "^PROBE_DONE$" agent/out/c30x-v5-runner.log 2>/dev/null; then
  echo V5DONE
fi
pgrep -f c30x_v5_runner >/dev/null 2>&1 && echo V5ALIVE || echo V5DEAD
# v6 臂到听数估计：v6 房数 × 每房到听 ~1.85（C30 全量 5436 房 23594 到听里我方 21745 → 实际按房估）
N_V6=$(.venv/bin/python -c "import json; d=json.load(open('agent/out/arm_map.json')); print(sum(1 for v in d['map'].values() if v=='v6'))" 2>/dev/null || echo 0)
echo "V6ROOMS=$N_V6"
