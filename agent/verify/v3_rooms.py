#!/usr/bin/env python3
"""只读：统计台账里 v3 时代的房间数（供 automations trigger 调用）。

输出单行 JSON：{"v3_rooms": N, "latest_started": "..."}
对平台零请求；只读 data/auto_sessions/sessions.jsonl。

用法：.venv/bin/python agent/verify/v3_rooms.py [--decider v3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS = ROOT / "data" / "auto_sessions" / "sessions.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", default="v3")
    args = ap.parse_args()

    n = 0
    latest = None
    if SESSIONS.exists():
        for line in SESSIONS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("decider") != args.decider:
                continue
            n += 1
            s = e.get("started_at")
            if s and (latest is None or s > latest):
                latest = s
    print(json.dumps({"v3_rooms": n, "latest_started": latest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
