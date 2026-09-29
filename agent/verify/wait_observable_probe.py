#!/usr/bin/env python3
"""agent-c 独立实测（第三项，修正版）：**只用公开信息**能否复现「计入对手暗手」后的出牌？

背景与两处坑：
- 决策影响已测（`wait_choice_impact_probe.py`）：计入对手暗手会改变 **27.4%** 的听牌出牌。
- 但上一版探针的「可观测估计」把分母写成了与现有口径 V 相同 ⇒ 必然同值 ⇒ **那个 72.6% 是假数**，
  不是「公开信息不可预测」的证据。本探针重做，用**三个真正不同的可观测估计量**对照全信息 T。

可观测估计量（都只用公开信息 + 自己的手牌 + 各家暗手**张数**，不含暗手构成）：
  ① V（现有口径）：Σ max(0, 4 − seen)。基线。
  ② 按比例摊派：E[opp_w] = H × unseen[w] / unseen_total，H=三家暗手总张数（可精确知道）。
     T_prop = Σ max(0, 4 − seen[w] − E[opp_w])
  ③ 壁式保守：若 seen[w] ≥ 3 视为对手几乎不可能持有（opp_w=0），否则同 V。
     T_kabe = Σ max(0, 4 − seen[w] − (0 if seen[w]>=3 else 0))  → 即 V 的特例；改用
     「邻居壁」：若 seen[w−1] 或 seen[w+1] 已达 4，则该面不可成顺 ⇒ 折减权重。

判据：各估计量的 argmax 与全信息 T 的 argmax 一致率。

只读真机事件流，对外零请求。
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
import glob
import json
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.rules import tiles, win
from majiang.rules.situation import PHASE_DRAW
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"
CPK = tiles.COPIES_PER_KIND


def _kinds(counts) -> list[int]:
    return [t for t in range(tiles.TILE_KINDS) if counts[t] > 0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=60)
    args = ap.parse_args(argv)
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[: args.rooms]

    n = 0
    same_v = same_prop = same_kabe = 0
    # 只统计「V 与 T 不同」的子集：这才是估计量真正要抢救的部分
    n_diff = 0
    same_prop_on_diff = same_kabe_on_diff = 0

    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
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
            counts14 = list(sit.hand.counts)
            seen = sm.visible_counts(counts14, [m.tiles for m in sit.all_melds], sit.discards)
            opp = [0] * tiles.TILE_KINDS
            H = 0
            for s in range(4):
                if s == mine:
                    continue
                for i, amount in enumerate(state.seats[s].hand):
                    opp[i] += amount
                H += sum(state.seats[s].hand)

            cands = []
            for tile in _kinds(counts14):
                after = list(counts14)
                after[tile] -= 1
                try:
                    waits = win.winning_draws(after, sit.hand.meld_count)
                except ValueError:
                    continue
                if not waits:
                    continue
                rem = list(seen)
                rem[tile] = max(0, rem[tile] - 1)
                unseen_total = sum(max(0, CPK - rem[w]) for w in range(tiles.TILE_KINDS))
                v = sum(max(0, CPK - rem[w]) for w in waits)
                t = sum(max(0, CPK - rem[w] - opp[w]) for w in waits)
                # ② 按比例摊派
                prop = 0.0
                for w in waits:
                    avail = max(0, CPK - rem[w])
                    e_opp = H * (avail / unseen_total) if unseen_total else 0.0
                    prop += max(0.0, avail - e_opp)
                # ③ 壁式：邻居已达 4 ⇒ 该牌不可成顺/刻，权重折半（保守）
                kabe = 0.0
                for w in waits:
                    avail = max(0, CPK - rem[w])
                    if avail == 0:
                        continue
                    blocked = False
                    for d in (-2, -1, 1, 2):
                        nb = w + d
                        if 0 <= nb < tiles.TILE_KINDS and tiles.is_number(w) and tiles.is_number(nb):
                            if tiles.suit(nb) == tiles.suit(w) and max(0, CPK - rem[nb]) == 0:
                                blocked = True
                    kabe += avail * (0.5 if blocked else 1.0)
                cands.append((tile, v, t, prop, kabe, waits))
            if len(cands) < 2:
                continue
            # 需要「候选导致不同听口」，否则比较无意义
            if len({c[5] for c in cands}) < 2:
                continue
            n += 1
            bt = max(cands, key=lambda c: (c[2], -c[0]))[0]
            bv = max(cands, key=lambda c: (c[1], -c[0]))[0]
            bp = max(cands, key=lambda c: (c[3], -c[0]))[0]
            bk = max(cands, key=lambda c: (c[4], -c[0]))[0]
            same_v += bv == bt
            same_prop += bp == bt
            same_kabe += bk == bt
            if bv != bt:
                n_diff += 1
                same_prop_on_diff += bp == bt
                same_kabe_on_diff += bk == bt

    print("=" * 66)
    print(f"房数={len(files)} · 听牌出牌点（≥2 候选、听口不同）={n}")
    if n:
        print(f"① 现有口径 V        vs 全信息 T：{same_v}/{n} = {same_v / n:.1%}")
        print(f"② 比例摊派估计       vs 全信息 T：{same_prop}/{n} = {same_prop / n:.1%}")
        print(f"③ 壁式折减估计       vs 全信息 T：{same_kabe}/{n} = {same_kabe / n:.1%}")
    if n_diff:
        print(f"   （V 与 T 不同的子集 n={n_diff}：② 抢救回 {same_prop_on_diff}/{n_diff} = "
              f"{same_prop_on_diff / n_diff:.1%}；③ 抢救回 {same_kabe_on_diff}/{n_diff} = "
              f"{same_kabe_on_diff / n_diff:.1%}）")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
