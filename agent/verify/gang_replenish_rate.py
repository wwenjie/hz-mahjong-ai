#!/usr/bin/env python3
"""统计「杠后补牌成胡」的真机经验频率（A 点名要的盈亏平衡点输入）。

用途：新问题 1（能胡时不考虑杠）的判据。若要放弃当下这一胡去杠，收益是
「杠开 ×2 且链 +1」，代价是「放弃确定的这一胡」——因此需要一个关键参数：
**杠之后紧接的补牌，真的成胡的比例是多少？**

口径：在真机 `logs/*.jsonl` 里，对每个 game_id 按 mono_ms 排序取本人 `decision.made`，
找 `choice` 以 `gang(` 开头的决策，看**紧随其后**的本人决策是否为 `phase=draw` 且 `choice=hu`。

局限（写在结论旁边）：
- 这是**决策层**的「补牌后选择胡」，不等于「补牌在牌型上成胡」——但两者在
  heuristic 下高度一致（补牌能胡且无飘机会时必选 hu）。
- 只统计**紧邻**的下一条本人决策；若中间夹了别人的副露窗口，该样本被计入「非胡」。
- 日志是运行中仍在增长的，数字是某一时点快照。

用法：nice -n 19 uv run python agent/verify/gang_replenish_rate.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

LOG_GLOB = "logs/*.jsonl"
GANG = "gang("


def main() -> int:
    # game_id -> [(mono_ms, choice, phase)]
    per_game: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    files = sorted(glob.glob(LOG_GLOB))
    if not files:
        print("未找到日志文件", file=sys.stderr)
        return 2
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if '"decision.made"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") != "decision.made":
                    continue
                per_game[record.get("game_id", "?")].append(
                    (float(record.get("mono_ms") or 0), str(record.get("choice")), str(record.get("phase")))
                )

    gangs = 0
    follow_draw = 0
    follow_hu = 0
    follow_piao = 0  # 补牌后弃胡（打牌而非胡）
    for entries in per_game.values():
        entries.sort()
        for index, (_, choice, _) in enumerate(entries):
            if not choice.startswith(GANG):
                continue
            gangs += 1
            if index + 1 >= len(entries):
                continue
            _, next_choice, next_phase = entries[index + 1]
            if next_phase != "draw":
                continue
            follow_draw += 1
            if next_choice == "hu":
                follow_hu += 1
            elif next_choice.startswith("discard"):
                follow_piao += 1

    print(f"日志文件 {len(files)} 个，涉及 {len(per_game)} 个 game_id")
    print(f"本人打出/暗杠决策（choice 以 gang( 开头）: {gangs}")
    print(f"其中紧接着是本人 phase=draw 决策的:        {follow_draw}")
    if follow_draw:
        print(f"  该补牌选择 hu:      {follow_hu}  ({follow_hu / follow_draw:.1%})")
        print(f"  该补牌选择弃牌:    {follow_piao}  ({follow_piao / follow_draw:.1%})")
        print(f"  → 杠后补牌成胡率（决策层）≈ {follow_hu / follow_draw:.1%}")
    print("\n对照：chase_baotou 的 88.7% 自补率是「弃胡追爆头后自补成胡」的比例。")
    print("本指标是「杠后补牌成胡」，两者同族（放弃当下小收益换倍数），但入口不同。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
