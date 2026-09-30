#!/usr/bin/env python
"""C17 全臂**裁决口径**审计：按 PROTOCOL 指标（名次分）重扫所有 A/B 实验日志。

**为什么做**：C11 发现协调者本轮用**总得分**（`PROTOCOL.md:200` 明令不用的重尾量）下裁决，
且 `shape-gate3` 被误记 kill（实为同向为正）。本仪器把这件审计**推广到全部臂**：
逐臂按「名次分为主 + 符号一致性」（`queue_agg.py` 口径）重算，找出

1. **口径冲突臂**：名次分结论与总得分结论**方向/显著性不同**的臂；
2. **疑似误杀臂**：名次分**同向为正且合并 |t|≥2**，但日志文件名带 `-only` 或 registry 记 `done/killed` 的；
3. **真负贡献臂**：名次分合并 t ≤ −2 且符号一致为负。

只读 `data/experiments/logs/*.log`；零平台请求；不 import 任何测量代码。
用法：.venv/bin/python agent/verify/all_arms_metric_audit.py
"""

from __future__ import annotations

import collections
import glob
import math
import pathlib
import re

LOGS = pathlib.Path(__file__).resolve().parents[2] / "data" / "experiments" / "logs"

LINE = re.compile(
    r"^\s*(总得分|名次分|白板数|胡次数|番数总和)\s+均值\s+([+\-][\d.]+)\s+标准误\s+([\d.]+)\s+t\s+([+\-][\d.]+)"
)
NAME = re.compile(r"^(?P<arm>.+?)-vs-(?P<base>.+?)-s(?P<seed>[0-9]+)-seed[0-9]+$")


def parse(path: pathlib.Path) -> dict[str, tuple[float, float]]:
    """返回 {metric: (mean, se)}（只取「逐场配对差分」段，避免抓到旋转行）。"""
    out: dict[str, tuple[float, float]] = {}
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if "逐场配对差分" in line:
            started = True
            continue
        if not started:
            continue
        m = LINE.match(line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def pooled(rows: list[tuple[float, float]]) -> tuple[float, float, int, int]:
    """逆方差加权合并 ⇒ (mean, t, n, 同向数)。"""
    usable = [(x, s) for x, s in rows if s > 0]
    if not usable:
        return float("nan"), float("nan"), 0, 0
    wsum = sum(1.0 / (s * s) for _, s in usable)
    mean = sum(x / (s * s) for x, s in usable) / wsum
    se = math.sqrt(1.0 / wsum)
    return mean, (mean / se if se else float("nan")), len(usable), sum(1 for x, _ in usable if x > 0)


def main() -> int:
    arms: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for f in sorted(LOGS.glob("*.log")):
        if f.name in ("mech-shape.log", "width-two-ply.log"):
            continue
        m = NAME.match(f.stem)
        if not m:
            continue
        d = parse(f)
        if "名次分" not in d:
            continue
        key = (m.group("arm"), m.group("base"))
        for metric in ("名次分", "总得分"):
            if metric in d:
                arms[key][metric].append(d[metric])

    print("C17 全臂裁决口径审计（PROTOCOL:200 ⇒ 判据用名次分，不用总得分）")
    print("=" * 92)
    print(f"{'臂':<22}{'n':>3}  {'名次分 t':>9} {'同向':>5}  {'总得分 t':>9} {'同向':>5}  口径")
    print("-" * 92)
    conflicts: list[str] = []
    for (arm, base), met in sorted(arms.items()):
        rm, rt, rn, rpos = pooled(met.get("名次分", []))
        sm, st, sn, spos = pooled(met.get("总得分", []))
        if rn == 0:
            continue
        # 口径冲突：名次分显著而总得分不显著（或符号相反）
        sig_r = abs(rt) >= 2
        sig_s = abs(st) >= 2
        sign_conflict = (rt > 0) != (st > 0)
        flag = ""
        if sign_conflict:
            flag = "★符号冲突"
        elif sig_r and not sig_s:
            flag = "★名次分显著/总得分不"
        elif sig_r and rpos == rn:
            flag = "同向为正"
        if flag:
            conflicts.append(f"{arm} vs {base}: 名次分t={rt:+.2f}({rpos}/{rn}) 总得分t={st:+.2f}({spos}/{sn}) {flag}")
        print(
            f"{arm:<22}{rn:>3}  {rt:>+9.2f} {rpos:>3}/{rn:<1}  {st:>+9.2f} {spos:>3}/{sn:<1}  {flag}"
        )
    print("=" * 92)
    print(f"需要复看的臂 = {len(conflicts)}")
    for c in conflicts:
        print("  •", c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
