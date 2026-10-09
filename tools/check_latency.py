"""延迟护栏自检：最近会话里 `decision.made` 的 `elapsed_ms` 分布 + 超预算计数。

**用途**（2026-10-09 17:40，比赛前）：换档（v5→v7）后按承诺复测真机延迟；
`p99 > 1500ms` 或**出现任何超预算事件** ⇒ 立即回退 `MAJIANG_COLLECT_DECIDERS=v5`。

用法::
    .venv/bin/python tools/check_latency.py                # 自动取最近 3 个日志
    .venv/bin/python tools/check_latency.py logs/a_xxx.jsonl
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    files = args or sorted(
        glob.glob("logs/*.jsonl"), key=lambda p: Path(p).stat().st_mtime, reverse=True
    )[:3]
    rows: list[tuple[float, float | None]] = []
    by_decider: dict[str, list[float]] = {}
    for path in files:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if "decision.made" not in line:
                continue
            try:
                record = json.loads(line[line.index("{"):])
            except Exception:  # noqa: BLE001
                continue
            value = record.get("elapsed_ms")
            if value is None:
                continue
            rows.append((float(value), record.get("budget_ms")))
            sig = str(record.get("decider", ""))[:40]
            by_decider.setdefault(sig, []).append(float(value))
    if not rows:
        print("未找到 decision.made（日志为空或路径不对）")
        return 1
    values = sorted(v for v, _ in rows)
    n = len(values)
    over = sum(1 for v, b in rows if b and v > b)
    print(f"文件 {len(files)} 个；决策 {n} 条")
    print(f"  延迟 elapsed_ms: p50 {values[n // 2]:.1f} / p90 {values[int(0.9 * n)]:.1f} / "
          f"p99 {values[min(n - 1, int(0.99 * n))]:.1f} / max {values[-1]:.1f}")
    print(f"  超预算：{over} 条；>1500ms：{sum(1 for v in values if v > 1500)} 条")
    for sig, vals in sorted(by_decider.items(), key=lambda kv: -len(kv[1])):
        vals.sort()
        print(f"  档位 {sig}：n={len(vals)} p99 {vals[min(len(vals) - 1, int(0.99 * len(vals)))]:.1f}ms")
    verdict = "**不通过：按 resume_collector.sh 第 3 节回退 v5**" if (
        over or values[-1] > 1500
    ) else "护栏通过 ✓"
    print(f"  ⇒ {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
