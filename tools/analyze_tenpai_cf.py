"""类 A / 类 B 强制对拍的读表：按 **类 × 巡目 × 财神** 分层。

读 `tools/trigger_counterfactual.py --mode discard --force-tile` 的逐点产出，
判据是**「强制牌 ≠ 基线实际打的牌」**（discard 模式的自动汇总判据是「拆对子」，与本题无关）。

预登记（THREAD 2026-10-09 11:25）：
- 类 A：若理论的巡目依赖成立，**早巡（1-6）差值应 ≥0、晚巡（11+）应显著为负**（符号随巡目翻转）；
- 类 B：若「听口种数」有独立价值，强制打「种数最多」应为**正**；否则判「不值得做」。

用法::
    .venv/bin/python tools/analyze_tenpai_cf.py agent/out/trigger-points/cf-tenpai.jsonl
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

Z_ALPHA = 1.959964
Z_POWER_80 = 0.8416212


def stat(values: list[float]) -> tuple[int, float, float, float]:
    n = len(values)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return n, mean, se, (mean / se if se else 0.0)


def line(name: str, values: list[float]) -> None:
    n, m, se, t = stat(values)
    if n < 2:
        print(f"  {name:<22} N={n:<5} 样本不足")
        return
    verdict = ("**强制分支更好**" if t > Z_ALPHA else
               ("**基线更好**" if t < -Z_ALPHA else "不显著"))
    print(f"  {name:<22} N={n:<5} {m:+8.3f}  se {se:6.3f}  t {t:+6.2f}   {verdict}")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "agent/out/trigger-points/cf-tenpai.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    usable = [
        r for r in rows
        if r.get("baseline") and r.get("treatment")
        and r["baseline"].get("picked") != r["treatment"].get("picked")
    ]
    print(f"文件 {path.name}：总 {len(rows)} 行，**强制牌与基线不同** {len(usable)} 行")
    if len(usable) < 2:
        print("样本不足")
        return 1

    def klass_of(row: dict) -> str:
        raw = str(row.get("klass", ""))
        return "A早听窄" if raw.startswith("A") else ("B种数" if raw.startswith("B") else "其他")

    for klass in ("A早听窄", "B种数", "其他"):
        group = [r for r in usable if klass_of(r) == klass]
        if not group:
            continue
        print(f"\n【{klass}】总体")
        line("全部", [float(r["diff"]) for r in group])
        buckets: dict[str, list[float]] = collections.defaultdict(list)
        for r in group:
            buckets[str(r.get("turn_bucket") or f"巡{r.get('turn')}")].append(float(r["diff"]))
        print(f"  按巡目：")
        for name in ("1-6巡", "7-10巡", "11巡+"):
            if buckets.get(name):
                line(name, buckets[name])
        gods: dict[str, list[float]] = collections.defaultdict(list)
        for r in group:
            gods[f"{min(int(r.get('god_n', 0)), 2)}财神"].append(float(r["diff"]))
        print(f"  按财神：")
        for name in sorted(gods):
            line(name, gods[name])
        if klass == "A早听窄":
            narrows: dict[str, list[float]] = collections.defaultdict(list)
            for r in group:
                narrows[f"听口{min(int(r.get('narrow_copies', 0)), 5)}张"].append(float(r["diff"]))
            print("  按听口张数：")
            for name in sorted(narrows):
                line(name, narrows[name])
        if klass == "B种数":
            gains: dict[str, list[float]] = collections.defaultdict(list)
            for r in group:
                delta = int(r.get("kind_best_kinds", 0)) - int(r.get("narrow_kinds", 0))
                gains[f"种数+{min(delta, 3)}"].append(float(r["diff"]))
            print("  按种数增益：")
            for name in sorted(gains):
                line(name, gains[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
