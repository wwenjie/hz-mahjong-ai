#!/usr/bin/env python3
"""agent-c 独立实测：候选面截断的排序键到底改不改候选集？

背景：我 11:22 的复核 ② 建议「`ukeire_order="blocks"` 或先按形质取候选面再截断」。
新 agent-b 11:50 指出：`blocks` 只改排序键、不改截断时机，`top2(total序)` 与
`top2(blocks序)` 只差约 1.8% ⇒ **`blocks` 不是这条的修法**，真修法是「先算进张再截断」。
本探针**独立复现**这个判断，口径自建，不 import 报告的任何统计。

只读真机事件流（`data/auto_sessions/*/events/*.json`），对外零请求。

用法: nice -n 19 uv run python agent/verify/probe_candidate_truncation.py [--rooms 40]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.rules.situation import PHASE_DRAW
from majiang.sim import replay
from majiang.strategy.policy import DISCARD, Action, HeuristicDecider, PolicyConfig

OUR = "u_a7f7c67bb14a"


def candidates(sit) -> list[int]:
    return [t for t in range(34) if sit.hand.counts[t] > 0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=40)
    ap.add_argument("--limit-points", type=int, default=80, help="最多对多少个截断生效点算精确进张")
    args = ap.parse_args(argv)

    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[: args.rooms]

    # ---- 两个配置：只差 ukeire_order（sort key），用于对比候选集是否改变 ----
    cfg_total = PolicyConfig(tiebreak="exact-ukeire", ukeire_order="total")
    cfg_blocks = PolicyConfig(tiebreak="exact-ukeire", ukeire_order="blocks")

    n_dec = 0
    n_tied3 = 0  # 截断真正生效（同向听并列 >=3 且向听<=1）
    same_top2 = 0  # top2(total) == top2(blocks)
    set_same_top2 = 0  # 作为集合相同
    ukeire_pts = 0
    cut_total = 0  # ukeire 最优张 不在 top2(total)
    cut_blocks = 0  # ukeire 最优张 不在 top2(blocks)
    room_ok = 0

    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        room_ok += 1
        mine = ids.index(OUR)

        for event, state in replay.iter_before_each_event(doc):
            if event.get("type") != replay.DISCARDED or event.get("seat") != mine:
                continue
            if not state.opened or (event.get("data") or {}).get("catch_play"):
                continue
            try:
                sit = state.situation_for(mine, phase=PHASE_DRAW)
            except Exception:
                continue
            n_dec += 1
            cand = candidates(sit)
            decider = HeuristicDecider(cfg_total)
            scores = sorted(
                (decider._score_discard(sit, Action(DISCARD, tile=t)) for t in cand),
                key=lambda s: s.total,
                reverse=True,
            )
            if not scores:
                continue
            top_sh = scores[0].shanten
            tied = [s for s in scores if s.shanten == top_sh]
            if top_sh > 1 or len(tied) < 3:
                continue
            n_tied3 += 1
            top2_total = [s.tile for s in tied[:2]]
            by_blocks = sorted(tied, key=lambda s: -s.blocks)
            top2_blocks = [s.tile for s in by_blocks[:2]]
            if top2_total == top2_blocks:
                same_top2 += 1
            if set(top2_total) == set(top2_blocks):
                set_same_top2 += 1

            # ---- 精确进张：对每个并列候选算 ukeire ----
            if ukeire_pts >= args.limit_points:
                continue
            visible = sm.visible_counts(
                sit.hand.counts,
                [m.tiles for m in sit.all_melds],
                sit.discards,
            )
            vals: dict[int, int] = {}
            memo: dict = {}
            for s in tied:
                counts = list(sit.hand.counts)
                counts[s.tile] -= 1
                try:
                    entries = sm.ukeire(counts, sit.hand.meld_count, visible=visible, memo=memo)
                except Exception:
                    continue
                vals[s.tile] = sum(c for _, c in entries)
            if not vals:
                continue
            best_tile = max(vals, key=lambda t: vals[t])
            ukeire_pts += 1
            if best_tile not in top2_total:
                cut_total += 1
            if best_tile not in top2_blocks:
                cut_blocks += 1

    print("=" * 60)
    print(f"房数（有效）={room_ok} · 出牌决策点={n_dec}")
    print(f"截断生效点（同向听并列>=3 且向听<=1）={n_tied3}")
    if n_tied3:
        print(f"  top2(total序) == top2(blocks序)（逐位）: {same_top2}/{n_tied3} = {same_top2/n_tied3:.1%}")
        print(f"  top2(total序) == top2(blocks序)（作集合）: {set_same_top2}/{n_tied3} = {set_same_top2/n_tied3:.1%}")
    print(f"精确进张测算点={ukeire_pts}")
    if ukeire_pts:
        print(f"  ukeire 最优张 **不在** top2(total序): {cut_total}/{ukeire_pts} = {cut_total/ukeire_pts:.1%}")
        print(f"  ukeire 最优张 **不在** top2(blocks序): {cut_blocks}/{ukeire_pts} = {cut_blocks/ukeire_pts:.1%}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
