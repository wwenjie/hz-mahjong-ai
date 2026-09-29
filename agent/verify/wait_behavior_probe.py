#!/usr/bin/env python3
"""agent-c 独立实测（第四项，决定项）：**用对手行为特征**估计「暗手占了多少张」，能否复现全信息出牌？

前三项：① 现有口径高估可用听口 ~1.8×（V=11.64 vs T=6.53）；
② 计入对手暗手会改变 **27.4%** 的听牌出牌；③ 按比例摊派 / 壁式折减只救回 **1.6%**。
第 ③ 项失败的原因：它们是**无信息**的估计（均匀摊派 = 对 V 做等比缩放，argmax 不变）。

本项用**真正的读牌信号**（公开信息）：
  - 对手弃牌的花色结构 ⇒ 某花色弃得少 ⇒ 他们更可能在持该花色；
  - 牌种级：某牌种从未被任何人（含我们）打出过、又不在我们手里 ⇒ 更可能被持有。
用「热花色」权重把未见张数摊派给对手，再与全信息 T 的 argmax 对照。

合规：估计量只用公开信息；对手暗手**仅作离线标签**，不入任何决策路径。

用法: nice -n 19 uv run python agent/verify/wait_behavior_probe.py --rooms 60
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.rules import tiles, win
from majiang.rules.situation import PHASE_DRAW
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"
CPK = tiles.COPIES_PER_KIND
SUITS = 4  # 万/筒/条/字


def _kinds(counts) -> list[int]:
    return [t for t in range(tiles.TILE_KINDS) if counts[t] > 0]


def _suit(t: int) -> int:
    return tiles.suit(t) if tiles.is_number(t) else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=60)
    args = ap.parse_args(argv)
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[: args.rooms]

    n = 0
    same_v = same_prop = same_beh = 0
    n_diff = 0
    rec_prop = rec_beh = rec_beh2 = 0
    beh_wins = beh_loses = 0
    gain_beh: list[int] = []

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

            # ---- 行为信号：对手弃牌的花色结构（只用公开信息）----
            opp_disc_by_suit = [0] * SUITS
            opp_disc_total = 0
            for s in range(4):
                if s == mine:
                    continue
                for d in state.seats[s].discards:
                    opp_disc_by_suit[_suit(d)] += 1
                    opp_disc_total += 1
            # Laplace：弃得越少 ⇒ 越可能在持该花色 ⇒ 权重越高
            suit_rate = [(opp_disc_by_suit[k] + 1.0) / (opp_disc_total + SUITS) for k in range(SUITS)]
            suit_weight = [1.0 / r for r in suit_rate]

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
                unseen = [max(0, CPK - rem[w]) for w in range(tiles.TILE_KINDS)]
                unseen_total = sum(unseen)
                v = sum(unseen[w] for w in waits)
                t = sum(max(0, unseen[w] - opp[w]) for w in waits)
                # ② 按比例摊派（与 probe3 同）
                prop = 0.0
                for w in waits:
                    e = H * (unseen[w] / unseen_total) if unseen_total else 0.0
                    prop += max(0.0, unseen[w] - e)
                # ④ 行为加权摊派
                wsum = sum(unseen[z] * suit_weight[_suit(z)] for z in range(tiles.TILE_KINDS))
                beh = 0.0
                for w in waits:
                    share = (unseen[w] * suit_weight[_suit(w)] / wsum) if wsum else 0.0
                    e = H * share
                    beh += max(0.0, unseen[w] - e)
                cands.append((tile, v, t, prop, beh, waits))
            if len(cands) < 2 or len({c[5] for c in cands}) < 2:
                continue
            n += 1
            bt = max(cands, key=lambda c: (c[2], -c[0]))[0]
            bv = max(cands, key=lambda c: (c[1], -c[0]))[0]
            bp = max(cands, key=lambda c: (c[3], -c[0]))[0]
            bb = max(cands, key=lambda c: (c[4], -c[0]))[0]
            same_v += bv == bt
            same_prop += bp == bt
            same_beh += bb == bt
            if bv != bt:
                n_diff += 1
                rec_prop += bp == bt
                rec_beh += bb == bt
                if bb == bt:
                    beh_wins += 1
                else:
                    beh_loses += 1

    print("=" * 68)
    print(f"房数={len(files)} · 听牌出牌点（≥2 候选、听口不同）={n}")
    if n:
        print(f"① 现有口径 V        vs 全信息 T：{same_v}/{n} = {same_v / n:.1%}")
        print(f"② 按比例摊派（无信息）vs 全信息 T：{same_prop}/{n} = {same_prop / n:.1%}")
        print(f"④ 行为加权摊派（读牌）vs 全信息 T：{same_beh}/{n} = {same_beh / n:.1%}")
    if n_diff:
        print(f"   V≠T 子集 n={n_diff}：② 救回 {rec_prop}/{n_diff}={rec_prop / n_diff:.1%}；"
              f"④ 救回 {rec_beh}/{n_diff}={rec_beh / n_diff:.1%}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
