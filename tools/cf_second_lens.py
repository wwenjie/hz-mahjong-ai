"""条件对拍产物的**第二口径**分析：把「别人本来就要胡」这个稀释项剥掉。

**为什么需要**（用户 2026-10-10 14:44 的批评，成立）：
「对拍的测试 case 并没有复原当时 4 家的场景……如果复原了场景，也有可能本身某一家原本就
马上要胡牌了，不管我怎么打都不改变结果。」——**前半句我承认是局限**（对拍只复原了
「四家起手 + 真实牌墙 + 到触发点为止的真实事件」，**触发点之后三家由 `--opponents` 决策器
模拟**，不是真机对手）；**后半句可以直接量**：`diff == 0` 的点里，有多少是「两分支都没人胡 /
都是同一家胡」= 该局结果与我的决策**无关**。

本工具给出四个口径（全部在同一批点上）：
1. **原始净分差**（旧口径，被锁定局稀释）；
2. **只看"我们胡没胡"** 的配对 2×2（McNemar）：`b` = 基线胡/处理不胡，`c` = 基线不胡/处理胡；
   这是「我们的决策到底有没有改变**我们自己的结果**」的直接检验，**不受别人胡不胡稀释**；
3. **只看两分支结果不同的点**（`diff != 0` 或 winner 变了）的净分差；
4. **锁定率**：两分支「我们都没胡 且 都是同一家/无人胡」的占比 = 我的决策与该局结果无关的比例。

用法::
    .venv/bin/python tools/cf_second_lens.py agent/out/trigger-points/cf-tbcover-slack0-local.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math


def wilson_like(mean: float, se: float) -> str:
    return f"{mean:+.3f} ± {1.96 * se:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="条件对拍第二口径（剥掉锁定局）")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--metric", default="diff")
    args = ap.parse_args()

    for path in args.paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        ok = [r for r in rows if r.get("ok")]
        live = [r for r in ok if r["baseline"].get("picked") != r["treatment"].get("picked")]
        print(f"\n===== {path} =====")
        print(f"总点 {len(rows)}  可用 {len(ok)}  **触发（两分支实选不同）{len(live)}**")

        for label, subset in (("全部可用点", ok), ("只触发点", live)):
            diffs = [r["diff"] for r in subset]
            n = len(diffs)
            if n < 2:
                continue
            mean = sum(diffs) / n
            se = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1) / n)
            # 锁定率
            locked = sum(
                1 for r in subset
                if not r["baseline"]["win"] and not r["treatment"]["win"]
                and r["baseline"]["score"] == r["treatment"]["score"]
            )
            zero = sum(1 for r in subset if r["diff"] == 0)
            # McNemar：我们的决策是否改变"我们胡没胡"
            b = sum(1 for r in subset if r["baseline"]["win"] and not r["treatment"]["win"])
            c = sum(1 for r in subset if not r["baseline"]["win"] and r["treatment"]["win"])
            # 两分支我们胡的次数
            wb = sum(1 for r in subset if r["baseline"]["win"])
            wt = sum(1 for r in subset if r["treatment"]["win"])
            chi = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
            print(f"  [{label}] n={n}")
            print(f"    净分差 {wilson_like(mean, se)}  t {mean / se if se else 0:+.2f}")
            print(f"    锁定局（两分支同分且我们都没胡）{locked}（{locked / n:.1%}）  分差恰为 0 的 {zero}（{zero / n:.1%}）")
            print(f"    **我们胡**：基线 {wb} / 处理 {wt} ⇒ Δ {wt - wb:+d}"
                  f"；McNemar b={b} c={c} χ²={chi:.2f}"
                  f"（{'显著' if chi > 3.84 else '不显著'}）")
        # 只看两分支结果不同的点
        diffpts = [r for r in live if r["diff"] != 0]
        if len(diffpts) > 1:
            diffs = [r["diff"] for r in diffpts]
            n = len(diffs)
            mean = sum(diffs) / n
            se = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1) / n)
            print(f"  [只看分差≠0 的点] n={n}  净分差 {wilson_like(mean, se)}  t {mean / se if se else 0:+.2f}")

        by_shanten = collections.defaultdict(list)
        for r in live:
            by_shanten[r.get("current_shanten")].append(r["diff"])
        print("  按最小向听（只触发点）：")
        for shanten in sorted(by_shanten, key=lambda x: (x is None, x)):
            diffs = by_shanten[shanten]
            n = len(diffs)
            if n < 2:
                continue
            mean = sum(diffs) / n
            se = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1) / n)
            wins = sum(1 for r in live if r.get("current_shanten") == shanten
                       and r["treatment"]["win"]) - sum(
                1 for r in live if r.get("current_shanten") == shanten and r["baseline"]["win"])
            print(f"    向听 {shanten}: n={n} 净分差 {mean:+.3f} t {mean / se if se else 0:+.2f}"
                  f"  Δ我们胡 {wins:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
