#!/usr/bin/env python3
"""跨种子聚合队列对照（只读 `notes/experiments.json`，不重跑、不发平台请求）。

为什么需要它：`notes/STATUS.md` 逐 job 报 `mean ± se`，但**每个 job 只有一个种子**。
把单种子的 `+1.9` 当成结论是错的——B 的噪声底（`verify/noise_floor.py`）实测
**两臂差 SD ≈ 1.33%**。本脚本把同一 treatment 的多个种子按**逆方差加权**合并，
给出合并均值、合并 se、t、以及符号一致性，再与噪声底对照。

判据（AGENTS.md §6.3）：合并 |t| < 2 或符号不一致 → **不得说「有提升」**。

用法: nice -n 19 uv run python agent/verify/queue_agg.py [--noise-floor 1.33]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS = ROOT / "notes" / "experiments.json"

# 与 A/B 主线相关的组，按 treatment 归类
METRICS = ("总得分", "名次分", "胡次数", "番数总和", "白板数")


def combine(rows):
    """逆方差加权合并。rows: list of (mean, se)。返回 (mean, se, k)。"""
    usable = [(m, s) for m, s in rows if s is not None and s > 0]
    if not usable:
        return None
    wsum = sum(1.0 / (s * s) for _, s in usable)
    mean = sum(m / (s * s) for m, s in usable) / wsum
    se = math.sqrt(1.0 / wsum)
    return mean, se, len(usable)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-floor", type=float, default=1.33,
                    help="真机噪声底背景值：两臂差胜率 SD%%。仅作参考打印；本表判读用 |t|>=1.96（分数尺度），不与它直接比较")
    args = ap.parse_args()

    doc = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    jobs = doc.get("jobs") or []

    # 去嵌套子样本：同一 (treatment, field, seed) 下若有多个 matches，只保留**最大**的那个。
    # 依据 A 14:05：`run_match` 随机源 = seed*100003 + index → 同种子的 40 场是 120 场的
    # **严格子集**，不能当独立复现累加（否则虚高功效与 t 值）。
    # key 必须含 `field`：`--field meld-equal` 与不带 field 是**不同实验**，不可合并。
    def _label(j):
        f = j.get("field")
        return f"{j.get('treatment')}@{f}" if f else str(j.get("treatment"))

    best = {}
    for j in jobs:
        if j.get("status") != "done":
            continue
        seeds = tuple((j.get("result") or {}).get("seeds") or j.get("seeds") or ())
        key = (_label(j), seeds)
        cur = best.get(key)
        if cur is None or (j.get("matches") or 0) > (cur.get("matches") or 0):
            best[key] = j
    kept = list(best.values())
    dropped = [j["id"] for j in jobs
               if j.get("status") == "done" and j not in kept
               and j.get("treatment")]

    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    matches_by_group = collections.defaultdict(set)
    seeds_by_group = collections.defaultdict(set)
    for j in kept:
        t = _label(j)
        m = (j.get("result") or {}).get("metrics") or {}
        if not m:
            continue
        if j.get("matches"):
            matches_by_group[t].add(j["matches"])
        for s in (j.get("result") or {}).get("seeds") or j.get("seeds") or []:
            seeds_by_group[t].add(s)
        for k in METRICS:
            if k in m and isinstance(m[k], dict):
                groups[t][k].append((m[k].get("mean"), m[k].get("se")))

    if dropped:
        print(f"[去嵌套] 丢弃 {len(dropped)} 个同 (treatment,field,seed) 的较小规模 job：{', '.join(sorted(dropped))}")

    print(f"[背景] 真机两臂差胜率 SD（B 实测）= {args.noise_floor:.2f}% —— **胜率尺度**，仅参考")
    print("判据：|t|>=1.96 且符号一致 → 过门（t 为跨种子逆方差合并，**总得分=分数尺度**）")
    print(f"{'treatment@field':22s} {'种子':>5} {'matches':>9} {'总得分(合并)':>20} {'t':>7} {'符号':>5} 判读")
    print("-" * 88)
    rows = []
    for t, md in groups.items():
        if "总得分" not in md:
            continue
        c = combine(md["总得分"])
        if c is None:
            continue
        mean, se, k = c
        tstat = mean / se if se else float("nan")
        signs = [1 if m > 0 else -1 for m, _ in md["总得分"] if m is not None]
        consistent = len(set(signs)) == 1
        if abs(tstat) >= 1.96 and consistent and abs(tstat) < 2.5:
            verdict = "边缘(过门，需更多种子)"
        elif abs(tstat) >= 1.96 and consistent:
            verdict = "过门"
        elif abs(tstat) >= 1.96 and not consistent:
            verdict = "显著但符号不一致"
        else:
            verdict = "未过门"
        print(f"{t:22s} {len(seeds_by_group[t]):>5} {str(sorted(matches_by_group[t])):>9} "
              f"{mean:+8.3f}±{se:.3f}  {tstat:>+7.2f} {'一致' if consistent else '不一致':>4} {verdict}")
        rows.append((t, mean, se, tstat, verdict))

    print()
    print("=== 判读 ===")
    ok = [r for r in rows if r[4] == "过门"]
    edge = [r for r in rows if r[4].startswith("边缘")]
    if not ok and not edge:
        print("无任何 treatment 在 |t|>=1.96 且符号一致下过门 → 现阶段不得写「有提升」。")
    else:
        for t, mean, se, tstat, _ in ok:
            print(f"  {t}: {mean:+.3f}±{se:.3f} (t={tstat:+.2f})")
    for t, mean, se, tstat, _ in edge:
        print(f"  [边缘] {t}: {mean:+.3f}±{se:.3f} (t={tstat:+.2f}) — 刚过 1.96，加种子复核前不采信")
    print()
    print("注意：合并只做**跨种子**；同一种子内的配对结构已由 ab_test 的 se 反映。")
    print("matches 数不同（40/120/200）时逆方差权重已按各自 se 处理，未强行等权。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
