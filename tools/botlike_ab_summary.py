#!/usr/bin/env python3
"""bot-like A/B 读数汇总（A 03:48③ 预登记判据）。

等 6 条 `--field botlike` A/B 落地后，一键汇总：
- 每场名次分 >0 且 t≥2 ⇒ 「场地伪影」成立、该轴按新场地重审
- ≤0 或 t<2（n=2 只作方向）⇒ 维持关闭
- |t|<1.2 关闭；1.2≤|t|<2 补到 6 种子

判据来自 A 03:48③（写进 THREAD 的预登记）。

用法：
    uv run python tools/botlike_ab_summary.py [--dir agent/out] [--pattern "ab-test-*-vs-v5"]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


def load_ab_result(path: str) -> dict | None:
    """从 ab_test 产物提取关键指标。"""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return {
        "path": path,
        "treatment": d.get("treatment"),
        "baseline": d.get("baseline"),
        "field": d.get("field"),
        "seed": d.get("seed"),
        "n_matches": d.get("n_matches"),
        "rank_points": d.get("rank_points"),  # 每场名次分
        "t_stat": d.get("t_stat"),  # t 值
        "win_rate": d.get("win_rate"),
        "avg_rank": d.get("avg_rank"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="agent/out", help="ab_test 产物目录")
    ap.add_argument("--pattern", default="ab-test-*-vs-v5", help="文件名模式")
    ap.add_argument("--out", default="agent/out/botlike-ab-summary.txt")
    args = ap.parse_args()

    # 找所有 botlike 场的 A/B 产物
    pattern = os.path.join(args.dir, f"{args.pattern}*.json")
    files = sorted(glob.glob(pattern))
    
    results = []
    for f in files:
        r = load_ab_result(f)
        if r and r.get("field") == "botlike":
            results.append(r)
    
    if not results:
        print(f"未找到 botlike 场的 A/B 产物（{pattern}）", flush=True)
        return 1
    
    # 按 treatment 分组
    by_treatment = defaultdict(list)
    for r in results:
        by_treatment[r["treatment"]].append(r)
    
    lines = [
        "=" * 72,
        "bot-like A/B 读数汇总（A 03:48③ 预登记判据）",
        "判据：每场名次分 >0 且 t≥2 ⇒ 场地伪影成立、该轴按新场地重审",
        "      ≤0 或 t<2（n=2 只作方向）⇒ 维持关闭",
        "      |t|<1.2 关闭；1.2≤|t|<2 补到 6 种子",
        "=" * 72,
        "",
    ]
    
    for treatment, rs in sorted(by_treatment.items()):
        lines.append(f"【{treatment}】（n={len(rs)} 种子）")
        lines.append(f"{'seed':<12}{'名次分':>10}{'t值':>8}{'胜率':>10}{'均名次':>10}")
        
        rank_points = []
        t_stats = []
        for r in sorted(rs, key=lambda x: x.get("seed", 0)):
            rp = r.get("rank_points", "N/A")
            ts = r.get("t_stat", "N/A")
            wr = r.get("win_rate", "N/A")
            ar = r.get("avg_rank", "N/A")
            lines.append(f"{r.get('seed', 'N/A'):<12}{rp:>10}{ts:>8}{wr:>10}{ar:>10}")
            if isinstance(rp, (int, float)):
                rank_points.append(rp)
            if isinstance(ts, (int, float)):
                t_stats.append(ts)
        
        # 合并判读
        if rank_points and t_stats:
            avg_rp = sum(rank_points) / len(rank_points)
            # 多个 seed 的 t 值合并（简化：取平均）
            avg_t = sum(t_stats) / len(t_stats)
            
            lines.append(f"\n合并：均名次分 {avg_rp:+.2f}，均 t 值 {avg_t:+.2f}")
            
            # 预登记判据
            if avg_rp > 0 and avg_t >= 2:
                verdict = "✅ 显著转正 ⇒ 「场地伪影」成立，该轴按新场地重审"
            elif abs(avg_t) < 1.2:
                verdict = "❌ |t|<1.2 ⇒ 关闭"
            elif 1.2 <= abs(avg_t) < 2:
                verdict = "⚠️ 1.2≤|t|<2 ⇒ 补到 6 种子"
            else:
                verdict = "❌ 未显著转正 ⇒ 维持关闭"
            lines.append(f"判读：{verdict}")
        
        lines.append("")
    
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
