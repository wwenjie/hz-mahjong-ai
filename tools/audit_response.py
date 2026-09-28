"""响应窗口审计：把「碰/吃/杠为什么 pass」变成可数的分布（只读日志）。

**为什么做这个面**：我们副露 0.59/局 vs 对手 1.09/局，但闸门放宽的三种配置实测**一致为负**，
所以「副露少」本身不是缺陷。可从没看过**多出来的那些 `pass` 是为什么**。
日志里每条决策都写了 `reason`，那是最直接的证据面，而且完全免费。

**要查的是「静默退化」**：若 `_shanten_after_meld` 之类的路径静默返回 None，
闸门看起来是「严格」、实际是「算不出来就 pass」——两者在 `reason` 上会不一样。

**首次结论（2026-09-28，120 文件 / 140730 条响应决策）**：

| 分类 | 占比 | 判读 |
|---|---|---|
| `不副露：向听 N 无改善` | 88.3% | **严格闸门的正常产物**（设计如此） |
| 无 `reason` 且候选只有 `pass` | 5.2% | 本就不在窗口内（`responding_seats` 不含本人），无害 |
| `保留七对路线：不吃不碰` | 2.3% | 路由判断，正常 |
| 实际副露（吃/碰/杠） | 4.2% | 决策生效 |
| 无 `reason` 但有真选项 | 0.5% | **全部是 `first-legal` 对照臂**（按设计不写 reason），无害 |

**⇒ 这个面没有静默退化。**「我们副露少」被如实解释为严格闸门 + 七对路线，
不存在「算不出来就 pass」的隐藏路径。

用法::

    uv run python tools/audit_response.py --limit 200
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

PHASES = ("response_peng", "response_chi", "response_gang")
# 对照臂按设计取第一个合法动作、不写 reason；把它单列，避免误判成退化。
CONTROL_ARMS = ("first-legal",)


def normalize(reason: str) -> str:
    """把 reason 里的数字与牌名挖掉，只留模板——模板才能数出分布。"""
    text = re.sub(r"向听 \d+→\d+", "向听 a→b", reason)
    text = re.sub(r"向听 \d+", "向听 a", text)
    text = re.sub(r"[0-9]?[东南西北中发白wtb]+", "T", text)
    return text[:60]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="响应窗口决策审计")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--limit", type=int, default=200, help="最多读几个房间日志")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(str(Path(args.log_dir) / "a_*.jsonl")))[: args.limit]
    by_phase: dict[str, collections.Counter] = {p: collections.Counter() for p in PHASES}
    reasons: collections.Counter = collections.Counter()
    choices: collections.Counter = collections.Counter()
    silent: collections.Counter = collections.Counter()  # (有无真选项, 档位)
    total = 0
    for path in files:
        for line in Path(path).read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 —— 单行坏日志不中断整次审计
                continue
            if row.get("event") != "decision.made":
                continue
            phase = str(row.get("phase") or "")
            if phase not in PHASES:
                continue
            total += 1
            reason = str(row.get("reason") or "")
            decider = str(row.get("decider") or "")
            by_phase[phase][(reason.split("：")[0].split(" ")[0]) or "(空)"] += 1
            reasons[normalize(reason)] += 1
            choices[str(row.get("choice") or "").split(":")[0]] += 1
            if not reason:
                kinds = {str(x).split(":")[0] for x in (row.get("candidates") or ())}
                silent[("有真选项" if kinds - {"pass"} else "只有 pass",
                        "对照臂" if decider in CONTROL_ARMS else decider)] += 1

    print(f"文件 {len(files)} 个；响应窗口决策 {total} 条")
    print("\n== 按阶段：reason 前缀分布 ==")
    for phase in PHASES:
        if by_phase[phase]:
            print(f"  {phase:16s} {dict(by_phase[phase])}")
    print(f"\n== 选择分布 ==\n  {dict(choices)}")
    print("\n== 无 reason 的细分（这是「静默退化」唯一可能出现的地方）==")
    for key, count in silent.most_common():
        print(f"  {count:6d}  {key}")
    print("\n== reason 模板 top 10 ==")
    for text, count in reasons.most_common(10):
        print(f"  {count:6d}  {text}")
    print("\n读法：`不副露：向听 a 无改善` 是严格闸门的正常产物；"
          "真正要警惕的是「**有真选项却无 reason**」那一格，且必须先排除对照臂。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
