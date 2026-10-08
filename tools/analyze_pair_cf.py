"""对拍逐点结果的分层分析（回答「该不该拆这对」）。

读 `tools/trigger_counterfactual.py --out` 产出的 JSONL，只取**确实触发**的点
（`triggered=True`），按可解释的维度分层报告 `保留 − 拆`（`diff`）的均值/se/t：

- `对子数`：手牌里「恰好 2 张」的牌种数（1 / 2 / 3+）——七对路线的代理
- `向听`：0（听牌，其实是**选听口**）/ 1-2（成型）/ 3+（散牌）
- `财神数`：0 / 1 / ≥2
- `拆的是否字牌`：字牌（东/南/西/北/中/发）对 vs 数牌对

用法::
    .venv/bin/python tools/analyze_pair_cf.py agent/out/trigger-points/cf-pairs.jsonl
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

Z_ALPHA = 1.959964
Z_POWER_80 = 0.8416212
HONORS = set(range(27, 34))


def stats(values: list[float]) -> tuple[int, float, float, float]:
    n = len(values)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return n, mean, se, mean / se if se else 0.0


def pairs_count(hand: list[int]) -> int:
    return sum(1 for amount in hand if amount == 2)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "agent/out/trigger-points/cf-pairs.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trig = [r for r in rows if r.get("triggered")]
    print(f"文件 {path.name}：总 {len(rows)} 点，其中**确实触发** {len(trig)} 点"
          f"（其余 {len(rows) - len(trig)} 为两分支同决策/异常/排除）")
    if len(trig) < 2:
        print("触发点不足，无法分层")
        return 1

    n, mean, se, t = stats([float(r["diff"]) for r in trig])
    mde = (Z_ALPHA + Z_POWER_80) * se
    print(f"\n总体：保留对子 − 拆对子 = {mean:+.3f} 净分/触发  se {se:.3f}  t {t:+.2f}  "
          f"95%CI [{mean - Z_ALPHA * se:+.3f}, {mean + Z_ALPHA * se:+.3f}]  MDE(80%) {mde:.3f}")
    print("（正 = 保留更好；负 = 拆更好）")

    def table(title: str, key) -> None:
        groups: dict[str, list[float]] = collections.defaultdict(list)
        for row in trig:
            groups[key(row)].append(float(row["diff"]))
        print(f"\n【{title}】")
        print(f"  {'分层':<16}{'N':>6}{'均值':>10}{'se':>9}{'t':>8}   判读")
        for name in sorted(groups):
            n2, m, s, tt = stats(groups[name])
            verdict = "样本不足" if n2 < 2 else ("保留更好" if tt > Z_ALPHA else ("拆更好" if tt < -Z_ALPHA else "不显著"))
            print(f"  {name:<16}{n2:>6}{m:>10.3f}{s:>9.3f}{tt:>8.2f}   {verdict}")

    table("按向听", lambda r: "0(选听口)" if r.get("current_shanten") == 0
          else ("1(成型)" if r.get("current_shanten") == 1
                else ("2" if r.get("current_shanten") == 2 else "3+散牌")))
    table("按对子数", lambda r: (lambda c: f"{min(c, 4)}对" if c < 4 else "4+对(七对近)")(
        pairs_count(r.get("hand_counts") or [0] * 34)))
    table("按财神数", lambda r: f"{min(int(r.get('god_n', 0)), 2)}财神"
          if int(r.get("god_n", 0)) < 2 else "2+财神")
    table("拆的对是否字牌", lambda r: "字牌对" if int(r.get("pair_tile", -1)) in HONORS else "数牌对")
    table("按副露数", lambda r: f"{len(r.get('melds') or [])}副露")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
