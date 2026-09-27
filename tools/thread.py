"""`notes/THREAD.md` 的未读检测与确认（A ↔ B 的消息通道）。

**为什么单独做工具**：用户已经提了两次「你要自己定时检查 B 的消息」——说明靠人/agent
记着是不可靠的。做法是把「有没有新消息」变成一个**可计算的事实**，并让状态页
`notes/STATUS.md` 每次都把它顶在最上面。

机制：`notes/.thread_seen_a` 记录我**已处理**的 `TO A` 条目数。未读 = 当前条数 − 已处理数。

用法::

    uv run python tools/thread.py          # 列出未读（从 B 来的）
    uv run python tools/thread.py --all    # 列出全部条目
    uv run python tools/thread.py --ack    # 把当前条数记为「已处理」
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

THREAD = Path("notes/THREAD.md")
MARKER = Path("notes/.thread_seen_a")
TITLE = re.compile(r"^###\s+(\S+ \S+)\s+FROM\s+(\S+)\s+TO\s+(\S+)\s+—\s*(.*)$")


def entries() -> list[tuple[str, str, str, str, bool]]:
    """返回 ``[(时间, 发, 收, 主题, 已处理), ...]``。"""
    if not THREAD.exists():
        return []
    lines = THREAD.read_text(encoding="utf-8").splitlines()
    parsed: list[tuple[str, str, str, str]] = []
    for line in lines:
        match = TITLE.match(line.strip())
        if match:
            parsed.append((match[1], match[2], match[3], match[4]))
    seen = 0
    if MARKER.exists():
        try:
            seen = int(MARKER.read_text().strip() or "0")
        except ValueError:
            seen = 0
    # 已处理数是按**从 B 到 A 的条目**计的，所以序号只在那一类里推进
    result: list[tuple[str, str, str, str, bool]] = []
    handled = 0
    for when, sender, receiver, topic in parsed:
        if receiver == "A":
            handled += 1
            result.append((when, sender, receiver, topic, handled <= seen))
        else:
            result.append((when, sender, receiver, topic, True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="THREAD 未读检测")
    parser.add_argument("--all", action="store_true", help="列出全部条目")
    parser.add_argument("--ack", action="store_true", help="把当前 TO A 条数记为已处理")
    args = parser.parse_args(argv)

    items = entries()
    to_a = [item for item in items if item[2] == "A"]
    if args.ack:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(str(len(to_a)), encoding="utf-8")
        print(f"已确认：{len(to_a)} 条 TO A")
        return 0

    unread = [item for item in to_a if not item[4]]
    if args.all:
        for when, sender, _receiver, topic, handled in items:
            flag = "  " if handled else "**"
            print(f"{flag}{when} {sender:>2} — {topic}")
        print()
    if unread:
        print(f"**{len(unread)} 条未读（TO A）**")
        for when, sender, _receiver, topic, _ in unread:
            print(f"  {when} FROM {sender} — {topic}")
        return 1
    print("没有未读消息")
    return 0


if __name__ == "__main__":
    sys.exit(main())
