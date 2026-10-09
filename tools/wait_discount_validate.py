"""验证「被捏牌折扣」是否有信息：折扣后的选听**更接近真值最优**吗？

背景（2026-10-09 16:10，`tools/wait_truth.py`）：
- 我们的听口张数口径 `est = 4 − 可见` 相对「墙里实剩」**中位高估 10 倍**；
- 听口有 **30%** 捏在对手暗手；
- 把排序键换成「墙里实剩」会让 **36.3%** 的听牌决策改选 ⇒ **非均匀**、剂量大。
- 但**统一乘子（王牌沉底比例）对排序零效应**，所以只有「**按牌种非均匀的折扣**」才有意义。

本脚本用**静态吸引力代理** `risk.visible_need`（中张 1.0 / 幺九 0.6 / 字牌 0.4）做折扣，
量「按该折扣选听」在**真值（墙里实剩）**上是否更优：
- `loss_raw`  = `truth[真值最优] − truth[按原估计最优]`
- `loss_disc` = `truth[真值最优] − truth[按折扣估计最优]`（折扣 `mult = 1 − k × need`）
⇒ 若 `loss_disc` 显著小于 `loss_raw`，则折扣有信息、值得做成机制件；否则关掉这条。

用法::
    .venv/bin/python tools/wait_discount_validate.py --rooms 300
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import shanten as sh  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules import win as win_mod  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy import risk as risk_mod  # noqa: E402
from majiang.strategy.policy import Mode, _wait_copies  # noqa: E402

OUR = "u_a7f7c67bb14a"
SEATS = 4
KS = (0.25, 0.5, 0.75)


def hands_from_start(span):
    codes_list = list(getattr(span, "start_hands", ()) or ())
    if len(codes_list) != SEATS or any(c is None for c in codes_list):
        return None
    hands = [[0] * tiles.TILE_KINDS for _ in range(SEATS)]
    for seat, codes in enumerate(codes_list):
        for code in codes:
            hands[seat][tiles.parse(code)] += 1
    return hands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="被捏牌折扣的验证")
    ap.add_argument("--rooms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args(argv)

    pool = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(pool) > args.rooms:
        pool = sorted(random.sample(pool, args.rooms))

    points = 0
    losses_raw: list[float] = []
    losses_disc: dict[float, list[float]] = {k: [] for k in KS}
    losses_oracle = 0.0
    changed: dict[float, int] = {k: 0 for k in KS}
    agree_truth: list[int] = []

    for path in pool:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        spans = replay.round_spans(doc)
        for index, (state, events) in enumerate(replay.iter_rounds(doc)):
            hands = hands_from_start(spans[index])
            if hands is None:
                continue
            events = list(events)
            future = [
                replay._tile_of(e.get("tile"))  # noqa: SLF001
                for e in events if e.get("type") == "tile_drawn"
            ]
            draws_done = 0
            melds = [0] * SEATS
            for event in events:
                kind = event.get("type")
                seat = event.get("seat")
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                data = event.get("data") or {}
                if not isinstance(seat, int):
                    continue
                if kind == "tile_drawn" and tile is not None:
                    hands[seat][tile] += 1
                    draws_done += 1
                elif kind == "tile_discarded" and tile is not None:
                    if hands[seat][tile] > 0:
                        hands[seat][tile] -= 1
                    if seat == mine and sum(hands[mine]) == 13 - 3 * melds[mine]:
                        try:
                            situation = state.situation_for(mine, phase=PHASE_DRAW)
                            actions = legal_actions(situation)
                        except Exception:  # noqa: BLE001
                            situation = None
                        if situation is not None:
                            visible = sh.visible_counts(
                                situation.hand.counts,
                                [m.tiles for m in situation.all_melds],
                                situation.discards,
                            )
                            rest = future[draws_done:]
                            rows = []
                            for candidate in [a for a in actions if a.kind == DISCARD]:
                                after = list(situation.hand.counts)
                                if after[candidate.tile] <= 0:
                                    continue
                                after[candidate.tile] -= 1
                                try:
                                    if sh.shanten(after, situation.hand.meld_count) != 0:
                                        continue
                                    waits = win_mod.winning_draws(after, situation.hand.meld_count)
                                except Exception:  # noqa: BLE001
                                    continue
                                est = _wait_copies(
                                    after, situation.hand.meld_count, visible, candidate.tile
                                )
                                if est is None or not waits:
                                    continue
                                truth = sum(rest.count(w) for w in waits)
                                need = sum(risk_mod.visible_need(w) for w in waits) / len(waits)
                                rows.append((candidate.tile, est, truth, need))
                            if len(rows) >= 2:
                                points += 1
                                best_truth = max(rows, key=lambda r: r[2])[2]
                                pick_raw = max(rows, key=lambda r: r[1])
                                losses_raw.append(best_truth - pick_raw[2])
                                losses_oracle += 0.0
                                agree_truth.append(1 if pick_raw[2] == best_truth else 0)
                                for k in KS:
                                    pick_disc = max(rows, key=lambda r: r[1] * (1.0 - k * r[3]))
                                    losses_disc[k].append(best_truth - pick_disc[2])
                                    if pick_disc[0] != pick_raw[0]:
                                        changed[k] += 1
                elif kind in ("chi", "peng", "gang", "minggang", "ming_gang"):
                    melds[seat] += 1
                    if kind == "chi":
                        run = [
                            t for t in (replay._tile_of(c) for c in (data.get("tiles") or ()))  # noqa: SLF001
                            if t is not None
                        ]
                        for t in [x for x in run if x != tile]:
                            if hands[seat][t] > 0:
                                hands[seat][t] -= 1
                    elif kind == "peng" and tile is not None:
                        for _ in range(2):
                            if hands[seat][tile] > 0:
                                hands[seat][tile] -= 1
                replay.apply_event(state, event)

    if not points:
        print("无样本")
        return 1
    print(f"可比听牌点 {points}")
    print(f"  按原估计选听：平均放弃墙内听口 {statistics.mean(losses_raw):.2f} 张；"
          f"「选中真值最优」的比例 {statistics.mean(agree_truth):.1%}")
    for k in KS:
        print(f"  折扣 k={k}: 平均放弃 {statistics.mean(losses_disc[k]):.2f} 张；"
              f"相对原估计改选 {changed[k] / points:.1%}")
    print("\n判读：折扣后「平均放弃」明显小于原估计 ⇒ 折扣有信息（值得做成机制件）；")
    print("      若两者接近 ⇒ 该代理无效，这条轴应关掉（改用行为类代理：对手弃牌/副露的牌种分布）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
