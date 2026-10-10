"""C 的独立机制复核：复现三个报障点的 v7 决策，dump 主分 argmax 与破平层走向。

只读 `src/`（不 import B' 的测量/回放代码的结论），用与 `tools/replay_report.py` 相同的
`_situation_from_view` 逆映射重建 `Situation`（这是快照格式的规范逆映射，非 B' 测量件）。

输出每点：
- 主分 argmax（按 `DiscardScore.total`）与其 `describe()`
- v7 实选（含破平层/闸门后的最终）
- 破平层是否覆盖了主分 argmax（`tied_full` / `tiebreak` / `tiebreak_narrowed` / `god_wait_boost` …）
- 候选面（每张的 total 与分项）

用法：
    .venv/bin/python agent/verify/mc_mechanism_probe.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import tiles as T  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

from replay_report import _situation_from_view  # noqa: E402

REPORTS = {
    "seq21": "webapp/reports/report_20261010_120431_seq21.json",
    "seq23": "webapp/reports/report_20261010_120618_seq23.json",
    "seq53": "webapp/reports/report_20261010_130453_seq53.json",
}
CLAIMS = {"seq21": "7b", "seq23": "3b", "seq53": "4b"}


def main() -> int:
    for tag, path in REPORTS.items():
        doc = json.load(open(os.path.join(_ROOT, path), encoding="utf-8"))
        view = doc["decision"]["situation"]
        sit = _situation_from_view(view)
        dec = make_decider("v7", Mode.QUALIFIER)
        actions = tuple(legal_actions(sit))
        chosen = dec.choose(sit, actions, budget_ms=1800)
        # 复算主分 argmax（独立于 choose 内部）
        cands = [a for a in actions if a.kind == DISCARD]
        scored = sorted(
            ((dec._score_discard(sit, a), a) for a in cands),
            key=lambda iv: iv[0].total, reverse=True,
        )
        print("=" * 78)
        print(f"{tag}  快照={doc['created_at']}  向听(主分argmax)={scored[0][0].shanten}")
        print(f"  手牌: {' '.join(view['my_hand'])}   摸到={view.get('drawn_tile')}")
        print(f"  v7 实选: {T.to_code(chosen.tile)}   用户主张: {CLAIMS[tag]}")
        print(f"  reason: {dec.last_reason}")
        print(f"  主分 argmax: {scored[0][1].describe()}")
        print("  候选面（按 total 降序）:")
        for sc, a in scored:
            mark = "  <== v7" if a.tile == chosen.tile else ("  <== 主张" if T.to_code(a.tile) == CLAIMS[tag] else "")
            print(f"    {a.describe():<48} total={sc.total:+.2f} 向听={sc.shanten} "
                  f"喂牌={sc.feed_cost:.2f} 财神={sc.god_penalty:.0f} 路线={sc.route}{mark}")
        print(f"  last_detail keys: {sorted(dec.last_detail.keys())}")
        for k in ("tied_full", "tiebreak", "tiebreak_narrowed", "preselect",
                  "god_wait_boost", "goodshape_pick", "tiebreak_timeout",
                  "wait_copies_failed", "piao_blocked_by"):
            if k in dec.last_detail:
                print(f"    {k} = {dec.last_detail[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
