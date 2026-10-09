"""在「放宽闸门」触发点上做**条件搜索**：是否存在「吃是赚的」子群（供条件化判据）。

背景：无脑放宽 = −0.790/触发（t −4.43，定向臂，4,786 点）⇒ 关闭。但**机会→吃 转化率我们只有
22%、强 bot 44%** ⇒ 要问的是「**哪些被拒的机会本不该拒**」。本脚本在既有点上搜条件，
不重跑对拍：分层维度都来自逐点数据。

分层维度：
- 财神数 / 向听 / 吃的是否含字牌 / 是否已有副露
- **被选吃法在候选里的形质排名**（从 `cand_pick` 的牌码反查 `options[*].ukeire_copies`）

判读：只有 |t|≥2 且为正、且 N≥60 的子群才值得做条件化判据；否则本轴整体关闭。
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import tiles as T  # noqa: E402

HONOR = 27
MIN_N = 60


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
    if n < MIN_N:
        return
    verdict = "**吃更好**" if t > 1.96 else ("**不吃更好**" if t < -1.96 else "不显著")
    print(f"  {name:<30} N={n:<5} {m:+8.3f}  se {se:6.3f}  t {t:+6.2f}   {verdict}")


def parse_chosen(row: dict) -> list[int]:
    match = re.search(r"chi:(\S+)", str(row.get("cand_pick", "")))
    if not match:
        return []
    out = []
    for code in match.group(1).split("+"):
        try:
            out.append(T.parse(code))
        except Exception:  # noqa: BLE001
            pass
    return out


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "agent/out/trigger-points/cf-eqchi3.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    usable = [r for r in rows if r.get("triggered") and r.get("baseline") and r.get("treatment")]
    diffs = [float(r["diff"]) for r in usable]
    n, m, se, t = stat(diffs)
    print(f"文件 {path.name}：可用点 {len(usable)}；总体 {m:+.3f}（se {se:.3f}、t {t:+.2f}）")

    buckets: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in usable:
        value = float(row["diff"])
        god = min(int(row.get("god_n", 0)), 2)
        shanten = int(row.get("current_shanten", -1))
        chosen = parse_chosen(row)
        options = list(row.get("options") or [])
        shapes = sorted((o.get("ukeire_copies") or 0) for o in options)
        honor = any(tile >= HONOR for tile in chosen)
        buckets["财神"][f"{god}财神"].append(value)
        buckets["向听"][f"向听{shanten}"].append(value)
        buckets["字牌"][("含字牌" if honor else "全数牌")].append(value)
        buckets["副露"]["已有副露" if (row.get("melds") or []) else "无副露"].append(value)
        if god >= 1 and shanten == 1:
            buckets["组合"]["财神≥1&向听1"].append(value)
        for option in options:
            if sorted(option.get("tiles") or []) == sorted(chosen):
                value_shape = option.get("ukeire_copies") or 0
                rank = 1 + sum(1 for s in shapes if s > value_shape)
                buckets["形质排名"][f"排名第{min(rank, 3)}"].append(value)
                break

    for title in ("财神", "向听", "字牌", "副露", "组合", "形质排名"):
        if not buckets[title]:
            continue
        print(f"\n【{title}】")
        for name in sorted(buckets[title]):
            line(name, buckets[title][name])
    print(f"\n（只报 N≥{MIN_N} 的子群；未见「吃更好」显著项 ⇒ 该轴无可用条件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
