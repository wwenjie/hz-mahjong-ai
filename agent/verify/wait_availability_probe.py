#!/usr/bin/env python3
"""agent-c 独立实测：我们的「听口剩余张数」估计被对手暗手高估了多少？

动机（用户 14:08 指出）：读对手的牌**不是**为了防点炮（本变体没有点炮），
而是为了算准**我们自己还摸不摸得到**——三家都在抢同一张时，那张牌不在墙里、
在别人手上。例：三家各持 1w2w 听 3w，我们持 2w3w 听 1w ⇒ 1w 近乎死口。

本探针用**离线全信息回放**测量两种口径的差（对手暗手**仅作离线标签**，不入决策路径）：
  V = 现有口径：Σ max(0, 4 − 已见)，已见 = 自己暗手 + 四家弃牌 + 四家副露
  T = 真实可用：Σ max(0, 4 − 已见 − 对手暗手)
再报「V>0 但 T=0」（看着能摸、其实全在别人手里）的比例，与逐张听口的死口率。

只读真机事件流（`data/auto_sessions/*/events/*.json`），对外零请求。

用法: nice -n 19 uv run python agent/verify/wait_availability_probe.py --rooms 60
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.rules import tiles, win
from majiang.rules.situation import PHASE_DRAW
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"
CPK = tiles.COPIES_PER_KIND


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=60)
    args = ap.parse_args(argv)
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[: args.rooms]

    n_pts = 0
    n_T0 = 0          # 真实可用 = 0（整条听口已死）
    n_trap = 0        # V>0 但 T=0（看着能摸、其实全在别人手里）
    diffs: list[int] = []
    vs: list[int] = []
    ts: list[int] = []
    ratios: list[float] = []
    total_waits = 0
    dead_waits = 0

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
                tile = tiles.parse(str(event.get("tile")))
            except Exception:
                continue
            counts14 = list(sit.hand.counts)
            if not (0 <= tile < tiles.TILE_KINDS) or counts14[tile] <= 0:
                continue
            visible = sm.visible_counts(
                counts14, [m.tiles for m in sit.all_melds], sit.discards
            )
            after = list(counts14)
            after[tile] -= 1
            try:
                waits = win.winning_draws(after, sit.hand.meld_count)
            except ValueError:
                continue
            if not waits:
                continue
            rem = list(visible)
            rem[tile] = max(0, rem[tile] - 1)
            opp = [0] * tiles.TILE_KINDS
            for s in range(4):
                if s == mine:
                    continue
                for i, amount in enumerate(state.seats[s].hand):
                    opp[i] += amount
            v_sum = sum(max(0, CPK - rem[w]) for w in waits)
            t_sum = sum(max(0, CPK - rem[w] - opp[w]) for w in waits)
            n_pts += 1
            vs.append(v_sum)
            ts.append(t_sum)
            diffs.append(v_sum - t_sum)
            if v_sum > 0:
                ratios.append(t_sum / v_sum)
            if t_sum == 0:
                n_T0 += 1
            if v_sum > 0 and t_sum == 0:
                n_trap += 1
            for w in waits:
                total_waits += 1
                if max(0, CPK - rem[w] - opp[w]) <= 0:
                    dead_waits += 1

    print("=" * 64)
    print(f"房数={len(files)} · 我方听牌出牌点={n_pts}")
    if not n_pts:
        return 0
    print(f"听口张数  现有口径 V：均值 {statistics.mean(vs):.3f}  中位 {statistics.median(vs):.1f}")
    print(f"听口张数  真实可用 T：均值 {statistics.mean(ts):.3f}  中位 {statistics.median(ts):.1f}")
    print(f"高估量 V−T：均值 {statistics.mean(diffs):.3f} 中位 {statistics.median(diffs):.1f} "
          f"| 逐点差>0 占 {sum(1 for d in diffs if d > 0) / n_pts:.1%}")
    print(f"比值 T/V：中位 {statistics.median(ratios):.3f}（=1 表示现有口径从不低估）/ "
          f"均值 {statistics.mean(ratios):.3f}")
    print(f"真实可用 T=0（整条听口已死）：{n_T0}/{n_pts} = {n_T0 / n_pts:.1%}")
    print(f"「看着能摸、其实全在别人手里」(V>0 且 T=0)：{n_trap}/{n_pts} = {n_trap / n_pts:.1%}")
    print(f"逐张听口：全死（4 张全被对手暗手占）{dead_waits}/{total_waits} = "
          f"{dead_waits / total_waits:.1%}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
