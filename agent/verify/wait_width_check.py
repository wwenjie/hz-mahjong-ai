#!/usr/bin/env python3
"""独立复算「听牌时听口可见张数」按财神数分层（只读、离线、零平台请求）。

目的：A 的 `tools/analyze_wait_ceiling.py` / `analyze_god_conversion.py` 是**同一作者**
自建的仪器，而「听口宽度」是他当前 P0 的唯一依据。我用自己的重实现独立复算同一张表，
**不 import 他的工具、不 import 他的测量代码**（AGENTS.md §6「仪表要对拍」）。

我的口径（显式写出，与 A 的可能有细节差）：
  - 时点：**我们的出牌之后**，且该时点我方 `best_shanten(...)==0`（听牌）。
  - 可见张数 = Σ_{胡牌张 t} (4 − seen[t])，其中
    seen[t] = 我方暗手张数 + **四家副露吃碰杠的牌面** + **四家已打出牌**（含我方刚打的那张）。
  - 财神分层 = 该时点我方暗手里的财神数。
  - 剔除抓打圈强制的出牌（无法识别时不下结论——本脚本不识别，故统计全量并在结论标注）。

复用 `agent/verify/baotou_decompose.py` 的已对拍重建器（22,833 局自检 100%）。
用法: nice -n 19 uv run python agent/verify/wait_width_check.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent" / "verify"))

from majiang.rules import tiles  # noqa: E402
from majiang.rules import shanten as shantenmod  # noqa: E402
from majiang.rules import win as winmod  # noqa: E402

import baotou_decompose as B  # noqa: E402

GOD = B.GOD


def replay_wait_widths(start_hands, events, our):
    """返回我方每个听牌出牌点的 (财神数, 可见张数)。"""
    hands = {i: [0] * tiles.TILE_KINDS for i in range(4)}
    meld_tiles = {i: [0] * tiles.TILE_KINDS for i in range(4)}
    melds = {0: 0, 1: 0, 2: 0, 3: 0}
    discards = [0] * tiles.TILE_KINDS
    for i, codes in enumerate(start_hands or []):
        for c in codes:
            hands[i][tiles.parse(c)] += 1
    out = []
    for e in events:
        t = e.get("type")
        s = e.get("seat")
        if s is None or not (0 <= s < 4):
            continue
        if t == "tile_drawn":
            hands[s][tiles.parse(e["tile"])] += 1
        elif t == "tile_discarded":
            tile = tiles.parse(e["tile"])
            hands[s][tile] -= 1
            discards[tile] += 1
            if s == our:
                # 出牌后时点：手牌是 13-3m 的等待型 → 用 shanten（不是 best_shanten，
                # 后者期望 14-3m 的摸牌后手牌，在此处会越界报错——我踩过这个坑）。
                try:
                    if shantenmod.shanten(hands[s], melds[s]) == 0:
                        seen = [0] * tiles.TILE_KINDS
                        for k in range(tiles.TILE_KINDS):
                            seen[k] = (hands[s][k] + meld_tiles[0][k] + meld_tiles[1][k]
                                       + meld_tiles[2][k] + meld_tiles[3][k] + discards[k])
                        wins = winmod.winning_draws(hands[s], melds[s])
                        vis = sum(max(0, tiles.COPIES_PER_KIND - seen[k]) for k in wins)
                        out.append((hands[s][GOD], vis))
                except Exception:
                    pass
        elif t == "peng":
            tile = tiles.parse(e["tile"])
            hands[s][tile] -= 2
            meld_tiles[s][tile] += 3
            melds[s] += 1
        elif t == "gang":
            kind = str((e.get("data") or {}).get("kind") or "ming")
            tile = tiles.parse(e["tile"])
            if kind == "an":
                hands[s][tile] -= 4
                meld_tiles[s][tile] += 4
                melds[s] += 1
            elif kind == "bu":
                hands[s][tile] -= 1
                meld_tiles[s][tile] += 1
            else:
                hands[s][tile] -= 3
                meld_tiles[s][tile] += 3
                melds[s] += 1
        elif t == "chi":
            data = e.get("data") or {}
            called = tiles.parse(e["tile"]) if e.get("tile") else None
            for c in (data.get("tiles") or []):
                cc = tiles.parse(c)
                meld_tiles[s][cc] += 1
                if cc == called:
                    continue
                hands[s][cc] -= 1
            melds[s] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    paths = sorted(glob.glob(str(ROOT / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.limit:
        paths = paths[:args.limit]
    print(f"扫描 {len(paths)} 个文件（每文件 8 局）")

    bucket = {0: [], 1: [], 2: []}  # god -> [可见张数]
    n_points = 0
    for p in paths:
        try:
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        seats = doc.get("seats") or []
        our = next((i for i, s in enumerate(seats) if s.get("user_id") == B.OUR_UID), None)
        if our is None:
            continue
        for rn, r in B.group_rounds(doc).items():
            if not r["start_hands"]:
                continue
            try:
                for g, vis in replay_wait_widths(r["start_hands"], r["events"], our):
                    bucket[min(g, 2)].append(vis)
                    n_points += 1
            except Exception:
                continue

    print(f"我方听牌出牌点 {n_points}")
    print("\n=== 我方听牌时的听口可见张数（按财神数分层）===")
    labels = {0: "0 张", 1: "1 张", 2: ">=2 张"}
    for g in (0, 1, 2):
        v = bucket[g]
        if not v:
            print(f"  {labels[g]}: 无样本")
            continue
        mean = sum(v) / len(v)
        sv = sorted(v)
        p25 = sv[int(0.25 * len(sv))]
        print(f"  {labels[g]:>6}: n={len(v):5d}  均值 {mean:6.2f}  中位 {sv[len(sv)//2]:5.1f}  p25 {p25}")
    print("\n对照 A 的仪器（analyze_god_conversion.py，400 文件）：")
    print("  0 张 7.64 / 1 张 11.73 / >=2 张 23.09（我方）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
