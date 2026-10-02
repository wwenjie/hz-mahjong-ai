#!/usr/bin/env python3
"""从采集日志构建 game_id → 决策臂（v5/v6/...）映射，落 agent/out/arm_map.json。

用途：Q2（v6 口径重切 C30x）需要按臂分桶。事件流文件本身不含 decider 字段，
但 `logs/*.jsonl` 的 `decision.made` 事件带 `game_id` + `decider`（knobs 全文），
按 knobs 特征分类：
  - 含 `ukeire-preselect=5`        → v6（presel5+piao13）
  - 含 `ukeire-candidates=3`（无 preselect） → v5
  - 其余（无 candidates 字段等）    → old（v4 及更早）

只读 logs/，幂等：重复跑覆盖写同一文件（原子替换）。对平台零请求。

用法：`.venv/bin/python agent/verify/build_arm_map.py [--since '2026-10-02 15:52:00']`
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "agent" / "out" / "arm_map.json"


def classify(decider: str) -> str:
    if "ukeire-preselect=5" in decider:
        return "v6"
    if "ukeire-candidates=3" in decider:
        return "v5"
    return "old"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="只扫 mtime 晚于该时刻的日志文件（如 '2026-10-02 15:52:00'）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    since_ts = None
    if args.since:
        since_ts = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M:%S"))

    arm_of = {}
    n_dec = collections.Counter()
    files = sorted(glob.glob(str(REPO / "logs" / "*.jsonl")))
    for path in files:
        if since_ts is not None and os.path.getmtime(path) < since_ts:
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if '"decision.made"' not in line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    gid = ev.get("game_id")
                    dec = ev.get("decider")
                    if not gid or not dec:
                        continue
                    arm = classify(str(dec))
                    n_dec[arm] += 1
                    prev = arm_of.get(gid)
                    if prev is not None and prev != arm:
                        # 同一房间出现两种臂——采集切换瞬间的边界房，标记出来不硬归类
                        arm_of[gid] = "mixed"
                    else:
                        arm_of[gid] = arm
        except OSError:
            continue

    payload = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "since": args.since,
        "n_games": len(arm_of),
        "arms": dict(collections.Counter(arm_of.values())),
        "decisions_by_arm": dict(n_dec),
        "map": arm_of,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, args.out)
    print(f"games={len(arm_of)} arms={payload['arms']} decisions={dict(n_dec)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
