#!/usr/bin/env python
"""bot-like 场 A/B 判读（A 2026-10-06 03:48 预登记判据）。

**判据**（照抄 A 03:48③，预登记，不可事后改）：
- 某旧臂在 bot-like 场里**每场名次分 >0 且 t≥2** ⇒ **「场地伪影」假说成立** ⇒ 该轴按新场地重审
- **≤0 或 t<2**（n=2 只作方向）⇒ **维持关闭**
- 升级规则沿用 A 11:35：|t|<1.2 关闭；1.2≤|t|<2 补到 6 种子
- **⚠️ 谨慎条款**：本场对手是模型近似（77.8% 一致率）⇒ **正中要谨慎，显著转正才有说服力**

**注意**：已完成 job 的 metrics 里只有「名次分」（逐局名次之和），A 判据里的「每场名次分」
是「按整场累计总得分给四家排 +3/+1/−1/−3」——两者口径不同（ab_test.py 输出注释明确写了不可混用）。
本脚本两个都报，判读用**每场名次分**（若有）；若该键缺失，退用「名次分」并在输出里标注。

用法：`.venv/bin/python agent/verify/botlike_ab_verdict.py`
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "notes" / "experiments.json"

# A 03:48③ 登记的三条臂（在「对自己」场里测平或测负的旧轴）
ARMS = ["v5-piao05", "v7m", "v5-standing-feed"]


def main() -> int:
    d = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    jobs = d if isinstance(d, list) else d.get("jobs", [])

    # 按臂分组：{arm: {seed: job}}
    by_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    for j in jobs:
        treatment = j.get("treatment", "")
        # botlike 场的 job 有顶层 "field": "botlike" 字段
        if treatment in ARMS and j.get("field") == "botlike":
            for s in j.get("seeds", []):
                by_arm[treatment][s] = j

    print("== bot-like 场 A/B 判读（A 03:48 预登记判据）==\n")
    any_pending = False
    for arm in ARMS:
        seeds = by_arm.get(arm, {})
        if not seeds:
            print(f"### {arm}: **无 botlike 场 job**（未入队或未跑）\n")
            any_pending = True
            continue
        done = {s: j for s, j in seeds.items() if j.get("status") == "done" and (j.get("result") or {}).get("metrics")}
        running = [s for s, j in seeds.items() if j.get("status") == "running"]
        pending = [s for s, j in seeds.items() if j.get("status") == "pending"]
        print(f"### {arm}: done={sorted(done)} running={running} pending={pending}")
        if not done:
            any_pending = True
            print()
            continue

        # 逐种子读「每场名次分」（优先）或「名次分」（退）
        means, ts = [], []
        used_key = None
        for s, j in sorted(done.items()):
            m = j["result"]["metrics"]
            for key in ("每场名次分", "名次分"):
                if key in m:
                    used_key = key
                    means.append(m[key]["mean"])
                    ts.append(m[key]["t"])
                    print(f"  seed {s}: {key} mean={m[key]['mean']:+.3f} t={m[key]['t']:+.2f}")
                    break
        if not means:
            print("  ⚠️ metrics 里既无「每场名次分」也无「名次分」\n")
            continue
        if used_key == "名次分":
            print("  ⚠️ 判读用「名次分」（逐局口径），非 A 判据的「每场名次分」——结论需 A 复核口径")

        # 判读（A 03:48③）
        pos = sum(1 for x in means if x > 0)
        sig = sum(1 for t in ts if abs(t) >= 2)
        weak = sum(1 for t in ts if 1.2 <= abs(t) < 2)
        verdict = (
            "✅ **场地伪影成立**（每场名次分 >0 且 t≥2）⇒ 该轴按新场地重审"
            if pos == len(means) and sig == len(ts)
            else ("🟡 **弱信号**（1.2≤|t|<2）⇒ 按升级规则补到 6 种子" if weak and pos > 0
                  else "❌ **维持关闭**（名次分 ≤0 或 t<2）")
        )
        print(f"  ⇒ **判读**: {verdict}\n")

    if any_pending:
        print("== 有臂未跑完，判读仅覆盖已完成部分 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
