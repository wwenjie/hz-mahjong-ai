#!/usr/bin/env python3
"""独立复算 A 的「基线漂移」核对（16:10 委托）。

A 的断言：`ab_test` 在 same_field 下，baseline 侧是共享的 `[heuristic]*4` 那一跑，
**跨不同 treatment 应逐位相同** —— 同一 seed + 同一场数下，不同 treatment 的
baseline 四旋转总得分应完全一致。

本脚本**只读日志文本**，自己解析「旋转 N ... baseline ...」行，不 import A 的工具。
输出：按 (seed, matches) 分组的 baseline 向量，并标出组内是否逐位一致。

用法：.venv/bin/python agent/verify/baseline_drift.py [--logs data/experiments/logs]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TOPLINE = re.compile(r"treatment=(\S+)\s+baseline=(\S+)\s+场上另三座=(\S+)\s+场数\s+(\d+)\s+每场\s+(\d+)\s+局\s+种子\s+(\d+)")
ROT = re.compile(r"旋转\s+(\d+)（[^）]*）\s+treatment\s+总得分\s+(-?\d+)\s+baseline\s+(-?\d+)")


def parse(path: Path):
    top = None
    rots = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TOPLINE.search(line)
        if m:
            top = {
                "treatment": m.group(1),
                "baseline": m.group(2),
                "field": m.group(3),
                "matches": int(m.group(4)),
                "seed": int(m.group(6)),
            }
        m2 = ROT.search(line)
        if m2:
            rots[int(m2.group(1))] = int(m2.group(3))  # baseline 总得分
    return top, rots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/experiments/logs")
    args = ap.parse_args()
    root = Path(args.logs)

    groups: dict[tuple, list] = {}
    for p in sorted(root.glob("*.log")):
        top, rots = parse(p)
        if not top or not rots:
            continue
        if top["baseline"] != "heuristic":
            continue
        key = (top["seed"], top["matches"], top["field"])
        vec = tuple(rots.get(i) for i in range(4))
        groups.setdefault(key, []).append((p.name, top["treatment"], vec))

    print("== baseline 向量按 (seed, matches, field) 分组 ===")
    for key, items in sorted(groups.items()):
        seed, matches, field = key
        vecs = {v for _, _, v in items}
        flag = "逐位一致" if len(vecs) == 1 else "!!! 不一致 !!!"
        print(f"\nseed={seed} matches={matches} field={field}  [{flag}]  n={len(items)}")
        for name, tr, v in items:
            print(f"   {tr:20s} {v}   ({name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
