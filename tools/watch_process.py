"""长稳测试旁观采样（tasks.md 7.1）。

比赛期间进程要连续在线数小时，因此需要能证明「跑久了不会变坏」：内存不涨、文件描述符
不涨、线程数不涨、活动速率稳定。本工具**不接触被测进程**，只读 ``/proc``，因此可以随时
挂在正在挂机的参赛进程旁边观察。

采样项：

- ``rss_kb`` —— 常驻内存。持续单调上升即为泄漏信号
- ``fd_count`` —— 打开的文件描述符（连接、日志句柄）
- ``threads`` —— 线程数（每场对局占一个工作线程，应稳定在并发数附近）
- ``cpu_seconds`` —— 累计 CPU 时间，用于算平均占用
- ``log_bytes`` —— 日志总字节数，其增量反映活动速率是否稳定

用法::

    uv run python tools/watch_process.py --match auto_session --interval 60 --duration 7200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")


def find_pid(pattern: str) -> int | None:
    """在 /proc 里按命令行匹配进程，返回最新的一个。"""
    best: tuple[float, int] | None = None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if pattern not in cmdline or "watch_process" in cmdline:
                continue
            started = float((entry / "stat").read_text().split()[21])
        except (OSError, IndexError, ValueError):
            continue
        if best is None or started > best[0]:
            best = (started, int(entry.name))
    return None if best is None else best[1]


def sample(pid: int) -> dict | None:
    root = Path("/proc") / str(pid)
    try:
        status = (root / "status").read_text()
        stat = (root / "stat").read_text()
        fds = len(list((root / "fd").iterdir()))
    except OSError:
        return None
    fields: dict[str, str] = {}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    # stat 里 comm 可能含空格，用右括号定位
    tail = stat[stat.rindex(")") + 2 :].split()
    return {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "rss_kb": int(fields.get("VmRSS", "0 kB").split()[0]),
        "vm_kb": int(fields.get("VmSize", "0 kB").split()[0]),
        "threads": int(fields.get("Threads", "0")),
        "fd_count": fds,
        "cpu_seconds": round((int(tail[11]) + int(tail[12])) / CLK_TCK, 2),
    }


def log_bytes(directory: str) -> int:
    total = 0
    for path in Path(directory).glob("*.jsonl"):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def report(records: list[dict]) -> None:
    if len(records) < 2:
        print("采样点太少，无法给出趋势")
        return
    first, last = records[0], records[-1]
    hours = max(1e-6, len(records) * (records[1].get("interval_sec") or 0) / 3600.0)
    print(f"采样 {len(records)} 次，跨度约 {hours:.2f} 小时")
    for key, label in (("rss_kb", "常驻内存(KB)"), ("fd_count", "文件描述符"), ("threads", "线程数")):
        values = [r[key] for r in records]
        print(
            f"  {label:12s} 首 {first[key]:>8} 末 {last[key]:>8} 最小 {min(values):>8} "
            f"最大 {max(values):>8} 变化 {last[key] - first[key]:+d}"
        )
    cpu = last["cpu_seconds"] - first["cpu_seconds"]
    print(f"  平均 CPU 占用 {cpu / (hours * 3600) * 100:.1f}%")
    growth = last.get("log_bytes", 0) - first.get("log_bytes", 0)
    print(f"  日志增长 {growth / 1024:.0f} KB（活动速率{'稳定' if growth else '停滞！'}）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="长稳测试旁观采样")
    parser.add_argument("--match", default="auto_session", help="命令行关键字，用于定位进程")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 表示一直采样到进程消失")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out", default="data/stability")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / f"watch_{int(time.time())}.jsonl"
    deadline = None if args.duration <= 0 else time.monotonic() + args.duration
    records: list[dict] = []
    pid = args.pid
    print(f"采样开始（进程 {'pid=%d' % pid if pid else args.match}）→ {ledger}", flush=True)

    while True:
        if pid and not Path(f"/proc/{pid}").exists():
            print("目标进程已退出，停止采样", flush=True)
            break
        if not pid:
            pid = find_pid(args.match) or 0
            if not pid:
                time.sleep(args.interval)
                continue
        record = sample(pid)
        if record is None:
            pid = 0
            continue
        record["interval_sec"] = args.interval
        record["log_bytes"] = log_bytes(args.log_dir)
        records.append(record)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if deadline is not None and time.monotonic() >= deadline:
            break
        time.sleep(args.interval)

    report(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
