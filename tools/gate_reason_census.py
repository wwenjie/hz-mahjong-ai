"""闸门原因普查：真机响应窗口上，v5 **为什么**放行/否决副露（按 `last_detail['meld']['gate']`）。

**为什么需要**（2026-10-09 15:20）：
- 到听速度差距稳定存在（闲-闲每巡慢 0.10→0.21 向听，庄-庄同样），而**副露数 0.65 vs 1.23**；
- 但「放宽闸门」两条轴都已关闭（吃 −0.790、碰 −0.090）⇒ v5 严格档**已经吃掉所有「向听真降」
  的吃碰**，对手多出来的副露只能来自**被我们以别的理由拒掉**的窗口；
- 闸门里有一条「**七对路线承诺 ⇒ 不吃不碰**」（`_is_pair_route`，`pair_route_pairs=5`），
  以及「`equal-not-allowed`（严格档不许向听不变副露）」⇒ 本工具量这两类各占多少。
- 数据来源：今晚新加的**响应层 `meld` trace**（`last_detail['meld']['gate']`）——这正是那条日志的第一个用途。

用法::
    .venv/bin/python tools/gate_reason_census.py --rooms 400
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
from majiang.rules.action import PASS, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_RESPONSE_CHI, PHASE_RESPONSE_PENG  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="闸门原因普查（v5 为什么否决副露）")
    ap.add_argument("--rooms", type=int, default=400)
    ap.add_argument("--arm", default="v5")
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args(argv)

    dec = make_decider(args.arm, Mode.QUALIFIER)
    pool = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(pool) > args.rooms:
        pool = sorted(random.sample(pool, args.rooms))

    gates: collections.Counter = collections.Counter()
    rejects: collections.Counter = collections.Counter()
    decided = collections.Counter()
    windows = 0
    for path in pool:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for state, events in replay.iter_rounds(doc):
            for event in events:
                if (event.get("type") != "tile_discarded" or event.get("seat") == mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                offered = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if offered is None:
                    replay.apply_event(state, event)
                    continue
                for phase in (PHASE_RESPONSE_PENG, PHASE_RESPONSE_CHI):
                    try:
                        situation = state.situation_for(
                            mine, phase=phase, offered=offered, responding=(mine,)
                        )
                        actions = legal_actions(situation)
                    except Exception:  # noqa: BLE001
                        continue
                    if not any(a.kind != PASS for a in actions):
                        continue
                    windows += 1
                    dec.choose(situation, actions, budget_ms=2000)
                    detail = (dec.last_detail or {}).get("meld") or {}
                    gates[str(detail.get("gate", "(无 trace)"))] += 1
                    for option in detail.get("options") or []:
                        if option.get("reject"):
                            rejects[str(option["reject"])] += 1
                    decided["吃掉/碰掉" if any(
                        a.kind != PASS for a in actions) and dec.last_reason.startswith("副露")
                        else "pass"] += 1
                    break
                replay.apply_event(state, event)

    total = max(1, windows)
    print(f"档位 {args.arm}；响应窗口 {windows}（我方能被问到吃/碰的窗口）")
    print(f"\n【闸门（`detail['meld']['gate']`）】")
    for key, value in gates.most_common():
        print(f"  {key:<24} {value:>6}  {value / total:6.1%}")
    print(f"\n【逐候选被否原因（`options[*].reject`）】")
    for key, value in rejects.most_common():
        print(f"  {key:<24} {value:>6}")
    print(f"\n【最终决定】")
    for key, value in decided.most_common():
        print(f"  {key:<24} {value:>6}  {value / total:6.1%}")
    print("\n判读：`pair-route` 占比高 ⇒ 七对路线承诺在大量拒掉副露（提速的最大嫌疑）；")
    print("      `equal-not-allowed` 占比高 ⇒ 严格档不许「向听不变」的副露（已验证放宽会亏/中性）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
