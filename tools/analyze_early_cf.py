"""「早听 vs 好听」强制对拍的读表（修正汇总口径）。

**为什么需要单独读表**：`trigger_counterfactual.py` 的 discard 模式把 `triggered` 定义成
「基线拆对、处理不拆对」——那是给「拆对子」那类问题用的判据；本题（强制打另一张牌）
的判据应是**「强制牌与基线实际打的牌不同」**，故离线重算（逐点数据里 `baseline.picked`
与 `treatment.picked` 都在，不必重跑）。

用法::
    .venv/bin/python tools/analyze_early_cf.py agent/out/trigger-points/cf-early-A-20261009.jsonl
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


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "agent/out/trigger-points/cf-early-A-20261009.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = [
        r for r in rows
        if r.get("baseline") and r.get("treatment")
        and r["baseline"].get("picked") != r["treatment"].get("picked")
    ]
    print(f"文件 {path.name}：总 {len(rows)} 行，其中**强制牌与基线不同** = {len(usable)} 行")
    if len(usable) < 2:
        print("样本不足")
        return 1
    n, mean, se, t = stat([float(r["diff"]) for r in usable])
    mde = (Z_ALPHA + Z_POWER_80) * se
    print(f"\n**EV(晚一步宽听) − EV(早一步窄听) = {mean:+.3f} 净分/触发**  se {se:.3f}  t {t:+.2f}  "
          f"95%CI [{mean - Z_ALPHA * se:+.3f}, {mean + Z_ALPHA * se:+.3f}]  MDE(80%) {mde:.3f}")
    print("判读：负 ⇒ 早一步窄听更好（v5 现状对）；正 ⇒ 晚一步宽听更好（该做这条跨向听权衡）")

    def table(title: str, key) -> None:
        groups: dict[str, list[float]] = collections.defaultdict(list)
        for row in usable:
            groups[key(row)].append(float(row["diff"]))
        print(f"\n【{title}】")
        for name in sorted(groups):
            n2, m2, s2, t2 = stat(groups[name])
            verdict = "样本不足" if n2 < 2 else (
                "**宽听更好**" if t2 > Z_ALPHA else ("**窄听更好**" if t2 < -Z_ALPHA else "不显著")
            )
            print(f"  {name:<12} N={n2:<5} {m2:+8.3f}  se {s2:6.3f}  t {t2:+6.2f}   {verdict}")

    table("按向听", lambda r: f"向听{r.get('current_shanten')}")
    table("按财神数", lambda r: f"{min(int(r.get('god_n', 0)), 2)}财神")
    table("按进张比", lambda r: "1.5-2×" if float(r.get("ratio", 0)) < 2
          else ("2-3×" if float(r.get("ratio", 0)) < 3 else "3×+"))
    table("按是否字牌对", lambda r: "字牌对" if int(r.get("pair_tile", -1)) >= 27 else "数牌对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
