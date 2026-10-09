"""响应窗口空干预门：两档位在**真机吃/碰窗口**上的决策分歧率。

**为什么需要它**（2026-10-08 23:40，B' 车道）：`tools/divergence_gate.py` 只比**出牌层**
（`tile_discarded` 点）的分歧率，对只改「吃/碰闸门」的臂（`meld_chi_best` /
`meld_chi_tiebreak` / `meld_tolerance`）**天生盲**——这些臂在出牌层可以 100% 同，
在响应层却完全不同（`v7m-keepchi` vs `v5` 出牌层实测 0/300，但响应层应有分歧）。

本工具在真机**响应窗口**（吃/碰）上逐点比较两档位的 `choose()`，给出分歧率 +
分歧形态分布（pass↔吃、吃法↔吃法、pass↔碰…），用于 A 23:40 要求的
「响应窗口口径的空干预门」：分歧率 <5% ⇒ 判空干预、不进 A/B。

用法::

    .venv/bin/python tools/divergence_gate_resp.py --arms v5,v7m-keepchi --rooms 40
    .venv/bin/python tools/divergence_gate_resp.py --arms v5,v7m-keepchi --phase chi --limit 500
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import CHI, PASS, PENG, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_RESPONSE_CHI, PHASE_RESPONSE_PENG  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"

PHASES = {
    "chi": PHASE_RESPONSE_CHI,
    "peng": PHASE_RESPONSE_PENG,
}


def _kind(action) -> str:
    if action is None:
        return "none"
    return {CHI: "chi", PENG: "peng", PASS: "pass"}.get(action.kind, str(action.kind))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="响应窗口空干预门（吃/碰）")
    ap.add_argument("--arms", default="v5,v7m-keepchi", help="恰好两个档位名，逗号分隔")
    ap.add_argument("--rooms", type=int, default=40, help="抽样房数")
    ap.add_argument("--limit", type=int, default=0, help="最多比多少个响应决策点，0=不限")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--phase", choices=("both", *PHASES), default="both")
    ap.add_argument("--examples", type=int, default=5, help="打印几个分歧例子")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    if len(names) != 2:
        raise SystemExit("--arms 需要恰好两个档位名")
    deciders = [make_decider(n, Mode.QUALIFIER) for n in names]

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))

    want = list(PHASES) if args.phase == "both" else [args.phase]
    phase_const = {k: PHASES[k] for k in want}

    same = differ = skipped = 0
    forms: collections.Counter = collections.Counter()
    per_phase: collections.Counter = collections.Counter()
    examples: list[str] = []

    for path in files:
        if args.limit and same + differ >= args.limit:
            break
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for block_no, (state, events) in enumerate(replay.iter_rounds(doc)):
            for ev_index, event in enumerate(events):
                if args.limit and same + differ >= args.limit:
                    break
                if (event.get("type") != "tile_discarded" or event.get("seat") == mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                offered = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if offered is None:
                    replay.apply_event(state, event)
                    continue
                for tag, phase in phase_const.items():
                    try:
                        sit = state.situation_for(
                            mine, phase=phase, offered=offered, responding=(mine,)
                        )
                        acts = [a for a in legal_actions(sit)
                                if a.kind in (CHI if tag == "chi" else PENG, PASS)]
                    except Exception:  # noqa: BLE001
                        continue
                    real = [a for a in acts if a.kind != PASS]
                    if not real:
                        continue
                    # 只统计「引擎真认为我方在该窗口」（吃仅下家；碰任意）
                    if tag == "chi" and (int(event.get("seat", -1)) + 1) % 4 != mine:
                        continue
                    per_phase[tag] += 1
                    picks = [d.choose(sit, acts, budget_ms=2000) for d in deciders]
                    if any(p is None for p in picks):
                        skipped += 1
                        continue
                    if _kind(picks[0]) == _kind(picks[1]) and (
                        picks[0].kind == PASS or list(picks[0].tiles) == list(picks[1].tiles)
                    ):
                        same += 1
                    else:
                        differ += 1
                        forms[f"{_kind(picks[0])}→{_kind(picks[1])}"] += 1
                        if len(examples) < args.examples:
                            examples.append(
                                f"    [{tag}] block{block_no} ev{ev_index} offered="
                                f"{tiles.to_code(offered)}  {names[0]}={picks[0].describe()} "
                                f"vs {names[1]}={picks[1].describe()}"
                            )
                replay.apply_event(state, event)

    total = same + differ
    if not total:
        print("无样本")
        return 1
    rate = differ / total
    print(f"响应窗口门：{names[0]} vs {names[1]}，比较了 {total} 个响应决策点"
          f"（{'/'.join(f'{k}={v}' for k, v in per_phase.items())}）")
    print(f"  相同 {same}（{same / total:.1%}）  **不同 {differ}（{rate:.1%}）**")
    if skipped:
        print(f"  跳过 {skipped} 个（有档位没给出决策）")
    for form, count in forms.most_common():
        print(f"    分歧形态 {form}: {count}")
    for line in examples:
        print(line)
    print(f"\n判据：分歧率 <5% ⇒ 判**空干预**、不进 A/B；当前 {rate:.1%} ⇒ "
          f"{'**过门**' if rate >= 0.05 else '**判空干预**'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
