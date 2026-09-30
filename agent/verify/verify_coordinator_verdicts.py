#!/usr/bin/env python
"""C11 独立复算协调者本轮裁决（只读实验日志，零平台请求）。

**要核的裁决**（coordinator 2026-09-30 10:55）：
- `blocks-gate3`：s20260927 t=0.69 / s771014 t=1.99 / s31415926 t=1.29 → 「1/3 显著，不排臂」
- `shape-blocks`：s771014 t=2.37 / s20260927 t=0.86 / s31415926 t=0.29 → 「1/3 显著，不排臂」
- `shape-gate3`：三种子 t=1.16/1.47/0.09 → kill
- `two-ply`：s20260927 t=0.50 / s771014 t=0.65 → kill

**两处要独立核的**：
1. **裁决用的是哪个指标？** 上面那串 t 是**总得分**的 t（重尾）。而 `notes/PROTOCOL.md:200`
   明写「判据用**低方差指标**（名次分、胡次数、胜率），**不用重尾的每手分**」。
   ⇒ 若协调者用总得分 t 下裁决，等于用错指标。本脚本**同时报两套**（总得分 / 名次分 / 胡次数）。
2. **跨种子合并**：逐种子看容易被单种子噪声带偏，报 pooled 与符号一致性。

用法：.venv/bin/python agent/verify/verify_coordinator_verdicts.py
"""

from __future__ import annotations

import glob
import math
import pathlib
import re

LOGS = pathlib.Path(__file__).resolve().parents[2] / "data" / "experiments" / "logs"

ARMS = {
    "blocks-gate3": ["blocks-gate3-vs-v3-*.log"],
    "shape-gate3": ["shape-gate3-vs-v3-*.log"],
    "two-ply": ["two-ply-vs-v3-*.log"],
    "two-ply-only": ["two-ply-only-vs-v3-*.log"],
    "shape-blocks": ["shape-blocks-vs-v3-*.log"],
}

LINE = re.compile(
    r"^\s*(总得分|名次分|白板数|胡次数|番数总和)\s+均值\s+([+\-][\d.]+)\s+标准误\s+([\d.]+)\s+t\s+([+\-][\d.]+)"
)


def parse(path: pathlib.Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line)
        if m:
            out[m.group(1)] = {"mean": float(m.group(2)), "se": float(m.group(3)), "t": float(m.group(4))}
    return out


def main() -> int:
    print("C11 独立复算：协调者本轮裁决（metric 口径核对 + 跨种子合并）")
    print("=" * 84)
    print("PROTOCOL.md:200 ⇒ 判据用**低方差指标**（名次分/胡次数），**不用重尾的每手分（总得分）**。")
    print("=" * 84)
    for arm, pats in ARMS.items():
        files: list[pathlib.Path] = []
        for pat in pats:
            files += sorted(LOGS.glob(pat))
        if not files:
            continue
        print(f"\n── {arm} ──")
        pool: dict[str, list[tuple[float, float]]] = {}
        for f in files:
            seed = f.name.rsplit("-seed", 1)[-1].replace(".log", "")
            d = parse(f)
            if not d:
                print(f"  [{seed}] 解析失败")
                continue
            pool.setdefault("名次分", []).append((d["名次分"]["mean"], d["名次分"]["se"]))
            pool.setdefault("总得分", []).append((d["总得分"]["mean"], d["总得分"]["se"]))
            print(
                f"  [{seed}]  总得分 t={d['总得分']['t']:+.2f}  |  "
                f"名次分 t={d['名次分']['t']:+.2f}  |  胡次数 t={d['胡次数']['t']:+.2f}"
            )
        for metric, vals in pool.items():
            n = len(vals)
            if n == 0:
                continue
            # 逆方差加权合并
            wsum = sum(1.0 / (se**2) for _, se in vals if se > 0)
            msum = sum(m / (se**2) for m, se in vals if se > 0)
            pooled = msum / wsum if wsum else float("nan")
            pooled_se = math.sqrt(1.0 / wsum) if wsum else float("nan")
            pt = pooled / pooled_se if pooled_se else float("nan")
            signs = sum(1 for m, _ in vals if m > 0)
            print(
                f"    合并[{metric}] n={n} pooled_mean={pooled:+.4f} pooled_t={pt:+.2f} "
                f"同向={signs}/{n}"
            )
    print("\n" + "=" * 84)
    print("判读要点：若某臂的『名次分/胡次数』合并 t 明显异于协调者引用的总得分 t，")
    print("说明裁决用错了指标（总得分是 PROTOCOL 明确点名不用的重尾量）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
