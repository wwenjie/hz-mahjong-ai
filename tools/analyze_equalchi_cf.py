"""「放宽闸门（向听不变也吃）」定向臂对拍的分层读表 —— 找**条件**，不是再判总和。

背景：`v7m`（= v5 + `meld_tolerance=equal`）在响应层改动 33%（≈5.7 次/场），
但**无脑放宽的 EV 为负**（首测 −0.908，含下游污染；定向臂读数见本文件输出的总体行）。
而**机会→吃 转化率我们只有 22%（强 bot 44%）** ⇒ 真正要问的是：
**在什么条件下吃是赚的**（条件化判据），不是「门槛该多松」。

分层维度（都来自逐点数据，无需重扫）：
- `向听`（吃后向听 = `current_shanten`，恒为「不变」档 ⇒ 看绝对值）
- `财神数`
- **吃后是否即听牌**（`after_shanten == 0` 的选项存在）——这是「副露提速」最直接的形态
- 吃的两张牌**是否含字牌**（字牌搭子价值低）
- 该吃法是否是**全体吃法里形质最高的**（`ukeire_copies` 最大）

用法::
    .venv/bin/python tools/analyze_equalchi_cf.py agent/out/trigger-points/cf-eqchi3.jsonl
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

Z_ALPHA = 1.959964
Z_POWER_80 = 0.8416212
HONOR_START = 27


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
        print(f"  {name:<24} N={n:<5} 样本不足")
        return
    verdict = "**吃更好**" if t > Z_ALPHA else ("**不吃更好**" if t < -Z_ALPHA else "不显著")
    print(f"  {name:<24} N={n:<5} {m:+8.3f}  se {se:6.3f}  t {t:+6.2f}   {verdict}")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "agent/out/trigger-points/cf-eqchi3.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    usable = [
        r for r in rows
        if r.get("triggered") and r.get("baseline") and r.get("treatment")
    ]
    print(f"文件 {path.name}：总 {len(rows)} 行，**触发（处理臂吃了、基线没吃）** {len(usable)} 行")
    if len(usable) < 2:
        print("样本不足")
        return 1
    n, mean, se, t = stat([float(r["diff"]) for r in usable])
    mde = (Z_ALPHA + Z_POWER_80) * se
    print(f"\n总体（定向臂）：吃 − 不吃 = {mean:+.3f} 净分/触发  se {se:.3f}  t {t:+.2f}  "
          f"95%CI [{mean - Z_ALPHA * se:+.3f}, {mean + Z_ALPHA * se:+.3f}]  MDE(80%) {mde:.3f}")

    def opts(row: dict) -> list[dict]:
        return list(row.get("options") or [])

    table = collections.defaultdict(list)
    for row in usable:
        shanten = int(row.get("current_shanten", -1))
        god = int(row.get("god_n", 0))
        table["向听"].append((f"向听{shanten}", float(row["diff"])))
        table["财神"].append((f"{min(god, 2)}财神", float(row["diff"])))
        table["巡目段(若有)"].append((str(row.get("turn_bucket") or "未记录"), float(row["diff"])))

    for title, pairs in table.items():
        groups: dict[str, list[float]] = collections.defaultdict(list)
        for name, value in pairs:
            groups[name].append(value)
        print(f"\n【{title}】")
        for name in sorted(groups):
            line(name, groups[name])
    print("\n注：① `cand_pick` 不落 tiles ⇒ 未做「吃哪一张更好」细分；② 本轮点里未记巡目 ⇒")
    print("    「巡目段」恒为「未记录」，需重扫补 `turn_bucket` 才有意义。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
