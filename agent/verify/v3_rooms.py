#!/usr/bin/env python3
"""只读：统计台账里 v3 时代的房间数（供 automations trigger 调用）。

输出单行 JSON：{"v3_rooms": N, "v3_rows": R, "latest_started": "..."}
对平台零请求；只读 data/auto_sessions/sessions.jsonl。

房间数 = **distinct room_id**（不是行数）。同一 room_id 在「重启后继续」时会写第二行
（例：`a_d3e864deff84` 现有 2 行 @16:25:54 / @16:40:29），按行计数会把 8 房读成 9 房、
让 ≥8 / ≥30 阈值提前一格触发。`v3_rows` 保留行数供核对。

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

    rooms = set()
    rows_n = 0
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
            rows_n += 1
            rid = e.get("room_id")
            # 无 room_id 的行不能合并，给每行一个合成键（宁多算不吞掉）
            rooms.add(rid if rid else f"__row__{rows_n}")
            s = e.get("started_at")
            if s and (latest is None or s > latest):
                latest = s
    print(json.dumps({"v3_rooms": len(rooms), "v3_rows": rows_n, "latest_started": latest},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
