#!/usr/bin/env python3
"""参数搜索驱动（B' 2026-10-07，用户指令）。

坐标下降粗扫：一次只动一个参数，找出敏感的 3-5 个，再细搜。

用法：
    # 在算力机上跑（E 为主力）
    python tools/param_search.py --phase coarse --machines mj-e --jobs 56

    # 或本地跑（慢）
    python tools/param_search.py --phase coarse --jobs 4
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

# 可搜参数（policy.py _score_discard 的权重/阈值）
# 格式：(参数名, 基准值, 搜索步长, 下界, 上界)
PARAM_GRID = [
    ("w_shanten",    1.0, 0.3, 0.3, 2.0),   # 向听惩罚权重
    ("w_blocks",     1.0, 0.3, 0.3, 2.0),   # 骨架权重
    ("w_pair",       1.0, 0.3, 0.3, 2.0),   # 对子价值
    ("w_meld",       1.0, 0.3, 0.3, 2.0),   # 副露价值
    ("w_route",      1.0, 0.3, 0.3, 2.0),   # 路线价值
    ("w_feed",       1.0, 0.3, 0.3, 2.0),   # 喂牌惩罚
    ("w_god",        1.0, 0.3, 0.3, 2.0),   # 财神留手
    ("piao_threshold", 1.3, 0.3, 0.5, 2.5), # 飘阈值缩放
    ("ukeire_preselect", 5, 2, 1, 10),      # 进张预选数
    ("ukeire_max_shanten", 3, 1, 1, 5),     # 进张最大向听
]

def run_ab_test(treatment: str, seed: int, jobs: int, field: str = "self", matches: int = 20) -> dict:
    """跑一组 A/B，返回结果。"""
    cmd = [
        sys.executable, "tools/ab_test.py",
        "--treatment", treatment,
        "--baseline", "v5",
        "--matches", str(matches),
        "--seed", str(seed),
        "--jobs", str(jobs),
        "--field", field,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    # 解析输出
    result = {"treatment": treatment, "seed": seed, "elapsed_s": elapsed, "rc": r.returncode}
    for line in r.stdout.splitlines():
        if "名次分" in line and "t" in line:
            # 解析 t 值
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "t":
                    result["t"] = float(parts[i+1])
        if "胡率" in line and "vs" in line:
            result["hu_line"] = line.strip()
    return result

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["coarse", "fine"], default="coarse")
    ap.add_argument("--jobs", type=int, default=56)
    ap.add_argument("--field", default="self", help="对弈场地（self=对自己, botlike=对bot模型）")
    ap.add_argument("--matches", type=int, default=20, help="每场局数（粗扫 20，细搜 40）")
    ap.add_argument("--out", default="agent/out/param_search.jsonl")
    args = ap.parse_args()

    print(f"参数搜索 phase={args.phase} jobs={args.jobs} field={args.field} matches={args.matches}", flush=True)

    # 坐标下降粗扫：一次只动一个参数
    results = []
    for name, base, step, lo, hi in PARAM_GRID:
        for direction, delta in [("+", step), ("-", -step)]:
            val = base + delta
            if val < lo or val > hi:
                continue
            # 构造参数化 decider 名（需要 policy.py 支持）
            treatment = f"v5-param-{name}{direction}{abs(delta)}"
            print(f"  搜 {name} {direction}{abs(delta)} = {val:.2f}...", flush=True)
            # TODO: 这里需要 policy.py 支持参数化 decider 注册
            # 暂时跳过，等明天和 A 对齐参数化方案
            results.append({"param": name, "direction": direction, "value": val, "status": "skipped"})

    # 写结果
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"结果已写入 {args.out}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
