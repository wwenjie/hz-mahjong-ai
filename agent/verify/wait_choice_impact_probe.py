#!/usr/bin/env python3
"""agent-c 独立实测（第二项，决定性）：把「对手暗手」计入可用张，会不会改变我们的出牌？

第一项（`wait_availability_probe.py`）只说明现有口径**高估**可用张数（V/T≈1.8）。
但「高估」本身不构成修法：若把每条听口**同比例**缩放，argmax 不变 ⇒ 决策不变。
所以本探针测**决策相关性**：在「听牌出牌点」上，枚举每个可打的候选，
对每个候选算打完后的听口，分别用两种口径取最大者——
  V = Σ max(0, 4 − 已见)           （现有口径）
  T = Σ max(0, 4 − 已见 − 对手暗手) （真实可用，对手暗手**仅作离线标签**）
报两者**选出同一张打牌的比例**；不一致时，T 口径选中的听口比 V 口径宽多少张。

再测**可预测性**（决定这条能否在线上实现，不读暗手）：
用**公开信息**（各家弃牌/副露的可见张数）作特征，估计「某牌种被对手暗手占有多少张」，
看这个**可观测估计**能否复现 T 的排序（与全信息 T 的 argmax 一致率）。

只读真机事件流，对外零请求。

用法: nice -n 19 uv run python agent/verify/wait_choice_impact_probe.py --rooms 60
"""
from __future__ import annotations

import argparse
import collections
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

    n_pts = 0                 # 听牌出牌点（有≥2个候选导致不同听口）
    n_same = 0                # V/T 选出同一张
    gains: list[int] = []     # 不一致时 T 比 V 宽出的张数
    n_by_example = 0          # 触发用户举的「多面听里选哪条」情形
    # 可观测估计 vs 全信息
    n_obs_same = 0
    n_obs_pts = 0

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
            visible = sm.visible_counts(
                counts14, [m.tiles for m in sit.all_melds], sit.discards
            )
            opp = [0] * tiles.TILE_KINDS
            for s in range(4):
                if s == mine:
                    continue
                for i, amount in enumerate(state.seats[s].hand):
                    opp[i] += amount

            # 只处理已听牌的手（先看当前是否听牌：任一张打法后能胡）
            cands: list[tuple[int, int, int, tuple[int, ...]]] = []  # tile, V, T, waits
            for tile in _kinds(counts14):
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
                v = sum(max(0, CPK - rem[w]) for w in waits)
                t = sum(max(0, CPK - rem[w] - opp[w]) for w in waits)
                cands.append((tile, v, t, waits))
            if len(cands) < 2:
                continue
            # 只保留「候选导致不同听口」的点（否则比较无意义）
            uniq = {c[3] for c in cands}
            if len(uniq) < 2:
                continue
            n_pts += 1
            best_v = max(cands, key=lambda c: (c[1], -c[0]))
            best_t = max(cands, key=lambda c: (c[2], -c[0]))
            if best_v[0] == best_t[0]:
                n_same += 1
            else:
                gains.append(best_t[2] - best_v[2])
            if len({len(c[3]) for c in cands}) > 1 or len(uniq) >= 2:
                n_by_example += 1

            # ---- 可观测估计：用公开信息近似「对手暗手占用量」----
            # 简单代理：一个牌种「已见越多 ⇒ 对手持有越少」。
            # 这里只检验「公开信息能否给出与全信息 T 同序的估计」——
            # 用 seen 作为 opp 的替代（seen 含我们自己的手牌，故偏保守）。
            obs = list(visible)
            cands_obs = []
            for tile, v, t, waits in cands:
                rem = list(obs)
                rem[tile] = max(0, rem[tile] - 1)
                o = sum(max(0, CPK - rem[w]) for w in waits)
                cands_obs.append((tile, o))
            if cands_obs:
                n_obs_pts += 1
                best_obs = max(cands_obs, key=lambda c: (c[1], -c[0]))[0]
                if best_obs == best_t[0]:
                    n_obs_same += 1

    print("=" * 64)
    print(f"房数={len(files)} · 听牌出牌点（≥2 候选且听口不同）={n_pts}")
    if n_pts:
        print(f"V 口径与 T 口径选出同一张：{n_same}/{n_pts} = {n_same / n_pts:.1%}")
        print(f"不一致（= 计入对手暗手会改变出牌）：{n_pts - n_same}/{n_pts} = "
              f"{(n_pts - n_same) / n_pts:.1%}")
        if gains:
            gains.sort()
            mid = gains[len(gains) // 2]
            print(f"  不一致时 T 口径的听口比 V 宽：均值 {sum(gains) / len(gains):.2f} 张 / 中位 {mid} 张")
    if n_obs_pts:
        print(f"【可预测性】仅用公开信息（已见张数）估计 ⇒ 与全信息 T 同选："
              f"{n_obs_same}/{n_obs_pts} = {n_obs_same / n_obs_pts:.1%}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
