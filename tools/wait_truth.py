"""听口张数的**估计 vs 真值**：我们的「4 − 可见」有多偏，且**会不会改变选听**。

**为什么先做这个**（2026-10-09 15:50，B' 文献简报的第 3 条）：
简报建议「白板折扣修正的进张真值」。但先分清：
- **王牌沉底 / 白板折扣这类「统一乘子」修正，对候选排序是零效应**（所有候选同乘一个系数）
  ⇒ 做了也白做；
- 真正**非均匀**的只有「**对手暗手持有**」那一项（可见张数只覆盖弃牌与副露，暗手不可见）。
所以在写任何修正前，先量三件事：
  1. `est = 4 − 可见`（我们现在的口径）与 `truth_wall`（该听口在**剩余摸牌序列**里还剩几张）；
  2. 该听口当前被**对手暗手**捏住几张（`held`）；
  3. **关键**：排序键从 `est` 换成 `truth_wall`，我们选的听口**会不会变**（不变 ⇒ 该轴无剂量）。

口径：四家精确手牌（`start_hands` + 事件流）；剩余摸牌序列 = 该局后续 `tile_drawn` 的牌序。

用法::
    .venv/bin/python tools/wait_truth.py --rooms 400
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
from majiang.strategy.policy import Mode, _wait_copies  # noqa: E402

OUR = "u_a7f7c67bb14a"
SEATS = 4


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
    ap = argparse.ArgumentParser(description="听口张数：估计 vs 真值")
    ap.add_argument("--rooms", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args(argv)

    pool = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(pool) > args.rooms:
        pool = sorted(random.sample(pool, args.rooms))

    points = 0
    rank_changed = 0
    ratios: list[float] = []
    held_share: list[float] = []
    by_turn: dict[int, list[float]] = collections.defaultdict(list)
    examples: list[str] = []

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
            span = spans[index]
            hands = hands_from_start(span)
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
                            options = []
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
                                if est is None:
                                    continue
                                truth = sum(rest.count(w) for w in waits)
                                held = sum(
                                    hands[other][w]
                                    for other in range(SEATS) if other != mine for w in waits
                                )
                                options.append((candidate.tile, est, truth, held))
                            if len(options) >= 2:
                                points += 1
                                best_est = max(options, key=lambda x: x[1])
                                best_truth = max(options, key=lambda x: x[2])
                                ratios.append(best_est[2] / max(1, best_est[1]))
                                held_share.append(best_est[3] / max(1, best_est[1] + best_est[3]))
                                by_turn[draws_done // 4].append(best_est[2] / max(1, best_est[1]))
                                if best_est[0] != best_truth[0]:
                                    rank_changed += 1
                                    if len(examples) < 5:
                                        counts_now = situation.hand.counts
                                        hand_str = "".join(
                                            tiles.to_codes(
                                                [t for t in range(tiles.TILE_KINDS)
                                                 for _ in range(counts_now[t])]
                                            )
                                        )
                                        examples.append(
                                            f"    手={hand_str} 巡{draws_done // 4} | "
                                            f"估计最优={tiles.to_code(best_est[0])}(est {best_est[1]}) "
                                            f"真值最优={tiles.to_code(best_truth[0])}"
                                            f"(墙真值 {best_truth[2]}、被捏 {best_truth[3]})"
                                        )
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

    print(f"可比较的听牌决策点 {points}（同一手牌有 ≥2 个听牌候选）")
    if points:
        print(f"  估计/真值 比值：中位 {statistics.median(ratios):.2f}  均值 {statistics.mean(ratios):.2f}"
              f"  （<1 = 我们高估了墙里还剩的听口）")
        print(f"  听口被「对手暗手」捏住的占比：中位 {statistics.median(held_share):.1%}")
        print(f"  **排序改变（估计最优 ≠ 真值最优）= {rank_changed} / {points} = "
              f"{rank_changed / points:.1%}**")
        print("  按巡目（估计/真值 中位）：")
        for turn in sorted(by_turn):
            if len(by_turn[turn]) >= 5:
                print(f"    巡{turn}: {statistics.median(by_turn[turn]):.2f}（n={len(by_turn[turn])}）")
    for line in examples:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
