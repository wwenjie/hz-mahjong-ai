"""解析 `tools/run_sweep.sh` 的日志，按臂汇总（含**合并 t**，不只看单种子符号）。

**为什么要合并 t 而不是"数正号"**：4 个种子各自的 t 只说明"该种子显著与否"，
而本项目的判据是**合并估计**（各种子差分合并后的均值/se）。本工具按
`均值合并 = Σ(mean_i × n_i)/Σn_i`、`se合并 = sqrt(Σ se_i²)/k`（各种子场数相同 ⇒ 等权）
给出合并读数与合并 t，并附"正号数"作稳健性旁证。

用法::
    .venv/bin/python tools/sweep_summary.py --log /tmp/sweep.log --metric 每场名次分
"""
from __future__ import annotations

import argparse
import collections
import math
import re

BLOCK = re.compile(r"^### arm=(?P<arm>\S+) field=(?P<field>\S+) seed=(?P<seed>\S+)")
ROW = re.compile(r"^\s+(?P<metric>\S+)\s+均值\s+(?P<mean>[-+0-9.]+)\s+标准误\s+(?P<se>[0-9.]+)"
                 r"\s+t\s+(?P<t>[-+0-9.]+)")


def main() -> int:
    ap = argparse.ArgumentParser(description="重筛日志汇总")
    ap.add_argument("--log", default="/tmp/sweep.log")
    ap.add_argument("--metric", default="每场名次分")
    args = ap.parse_args()

    rows: dict[tuple[str, str], list[tuple[str, float, float, float]]] = collections.defaultdict(list)
    arm = field = seed = None
    with open(args.log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            hit = BLOCK.match(line)
            if hit:
                arm, field, seed = hit["arm"], hit["field"], hit["seed"]
                continue
            row = ROW.match(line)
            if row and row["metric"] == args.metric and arm:
                rows[(arm, field)].append(
                    (seed, float(row["mean"]), float(row["se"]), float(row["t"]))
                )

    print(f"口径 = {args.metric}（正值＝处理臂更好）；合并 = 各种子等权")
    print(f"{'臂':<16}{'场地':<12}{'种子数':>6}{'正号':>5}{'合并均值':>10}{'合并se':>8}"
          f"{'合并t':>8}   各种子均值")
    verdicts = []
    for (arm, field), items in sorted(rows.items()):
        count = len(items)
        mean = sum(i[1] for i in items) / count
        se = math.sqrt(sum(i[2] ** 2 for i in items)) / count
        t = mean / se if se else 0.0
        positive = sum(1 for i in items if i[1] > 0)
        flag = ""
        if t >= 2 and positive >= 3:
            flag = "  ← **候候选（过第一道）**"
            verdicts.append((arm, field, mean, t))
        elif t <= -2:
            flag = "  ← 显著为负、关闭"
        print(f"{arm:<16}{field:<12}{count:>6}{positive:>5}{mean:>10.3f}{se:>8.3f}{t:>8.2f}   "
              + " ".join(f"{i[0]}:{i[1]:+.3f}" for i in items) + flag)
    if verdicts:
        print("\n需在**第二场地**复现的候候选：")
        for arm, field, mean, t in verdicts:
            print(f"  {arm}（{field}，{mean:+.3f}，t {t:+.2f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
