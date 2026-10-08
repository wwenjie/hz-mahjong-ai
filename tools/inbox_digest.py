"""A 收件箱摘要：把「需要 A 裁决」与「机器状态」压成一页，供每轮唤醒时快速消费。

**为什么需要它**：A 是会话型 agent、不能自唤醒；而 `TO A` 的条目会累积
（2026-10-02 一度积到 47 条、其中 9 条判读压了 14 小时）。每次醒来若从 47 条里
人工挑，成本高且容易漏。本脚本按**标题契约**分流：

- 标题含 `[判读]` / `[机械]` / `巡检` / `心跳` / `收讫` / `回执` ⇒ **机械/知会**，A 只需扫一眼；
- 其余 `TO A` 条目 ⇒ **可能是裁决项**，优先列出。

口径依据：`notes/PROTOCOL.md` §3（THREAD 格式）与 A 在 2026-10-02 立的标题约定
（「需要 A 决策的内容请在标题写 `[待A决]`」）。

用法::

    uv run python tools/inbox_digest.py            # 只看可能的裁决项
    uv run python tools/inbox_digest.py --all      # 连机械项一起列
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

THREAD = Path("notes/THREAD.md")
QUEUE = Path("notes/experiments.json")
HEAD = re.compile(r"^### (\d{4}-\d\d-\d\d \d\d:\d\d) FROM (\S+?) TO (\S+) . (.+)$")
MECH = ("[判读]", "[机械]", "巡检", "心跳", "收讫", "回执", "[已核]", "DONE")


def main() -> int:
    parser = argparse.ArgumentParser(description="A 的收件箱摘要")
    parser.add_argument("--all", action="store_true", help="连机械/知会项也列出")
    args = parser.parse_args()

    entries = []
    for line in THREAD.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if not m:
            continue
        ts, sender, to, title = m.groups()
        if "A" not in to.replace("A,", "A").replace(",A", "A") and to != "A":
            if not re.search(r"\bA\b", to):
                continue
        entries.append((ts, sender, title))
    tail = entries[-40:]
    decisions = [e for e in tail if not any(tag in e[2] for tag in MECH)]
    mech = [e for e in tail if any(tag in e[2] for tag in MECH)]

    print(f"=== 近 40 条 TO A：**可能需裁决 {len(decisions)}** / 机械知会 {len(mech)}")
    print("\n【可能需裁决】")
    for ts, sender, title in decisions:
        print(f"  {ts}  {sender:<8} {title[:80]}")
    if args.all:
        print("\n【机械知会（扫一眼即可）】")
        for ts, sender, title in mech[-12:]:
            print(f"  {ts}  {sender:<8} {title[:80]}")

    if QUEUE.exists():
        q = json.loads(QUEUE.read_text(encoding="utf-8"))
        c = Counter(j.get("status") for j in q["jobs"])
        print(f"\n=== 队列：跑 {c.get('running',0)} / 待 {c.get('pending',0)} / 完 {c.get('done',0)}")
        # 队列空 = 今天最大的浪费来源，必须显眼
        if c.get("pending", 0) == 0 and c.get("running", 0) == 0:
            print("  ⚠ **队列空**：立刻补臂或执行降并发（见 docs/ops.md 的零浪费时机）")
        elif c.get("pending", 0) < 4:
            print("  ⚠ 队列低于 4 条，约 1 小时内会跑空 ⇒ 该补了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
