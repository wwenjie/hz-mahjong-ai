#!/bin/sh
# agent-c: 回收守望用的只读探针。输出 "<产物数> <新 specialist 会话数>"。
# 零平台请求；只读文件系统与 OpenClaw 状态库。
cd /home/wuwenjie01/majiang_ai || exit 1
arts=$(ls research/mahjong-tenpai-speed-survey.md research/code-review-discard-strategy.md 2>/dev/null | wc -l)
newses=$(.venv/bin/python - <<'PY' 2>/dev/null || echo 0
import sqlite3
con = sqlite3.connect("file:/home/wuwenjie01/.openclaw/state/openclaw.sqlite?mode=ro", uri=True)
row = con.execute(
    "select count(*) from session_state_heads where agent_id in ('team-researcher','team-reviewer') and updated_at > 1790651975000"
).fetchone()
print(row[0] if row else 0)
PY
)
echo "$arts $newses"
