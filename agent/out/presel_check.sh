#!/usr/bin/env bash
# presel 批状态检查（供 cron trigger 调用）：输出 PEND=n DONE=m + 队列存活标记
set -u
cd /home/wuwenjie01/majiang_ai
.venv/bin/python - <<'PY'
import json
d = json.load(open('notes/experiments.json'))
jobs = d.get('jobs', [])
pend = done = 0
for j in jobs:
    t = str(j.get('treatment', ''))
    if t.startswith('v5-presel'):
        if j.get('status') == 'done':
            done += 1
        else:
            pend += 1
print(f"PEND={pend} DONE={done}")
PY
pgrep -f 'ab_test.py' >/dev/null 2>&1 && echo QUEUE_ALIVE || echo QUEUE_DEAD
