#!/usr/bin/env python3
"""爆头缺口分解·第二层：控制变量以定因果（只读、离线、零平台请求）。

第一层（`baotou_decompose.py`）已知：
  P(爆头|胡) 我们 15.45% vs 对手 22.10%（22,833 局，仪表自检 100%）。
本层回答 A 的互斥二分：差异是**留财神**还是**结构（自然面子/副露）**造成的？

三个控制测试：
  T1 同 god 桶内比爆头率 → 若控制住财神差距消失，则①「留财神」是主因；
     若仍差，则财神数不是解释。
  T2 同 god 桶内比**副露数**与**自然面子进度**（`natural_shanten`）。
  T3 **到听时刻**的 god 分布（首次成听牌型那一刻），对比胡牌时刻。

复用 `agent/verify/baotou_decompose.py` 的重建器（已 100% 自检）。
用法: nice -n 19 uv run python agent/verify/baotou_control.py
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
from majiang.rules.hand import Hand  # noqa: E402

import baotou_decompose as B  # noqa: E402

GOD = B.GOD


def first_waiting_gods(start_hands, events, our):
    """逐事件推进四家暗手，返回「首次成听牌型」时的 {seat: god数}。

    口径：只在**摸牌后**（打前）判定；**听牌的真判据是 `best_shanten(...) == 0`**，
    不是 `Hand.is_waiting_shape`（后者只查张数==13-3m，是尺寸谓词，我踩过这个坑）。
    """
    hands = {i: [0] * tiles.TILE_KINDS for i in range(4)}
    melds = {0: 0, 1: 0, 2: 0, 3: 0}
    for i, codes in enumerate(start_hands or []):
        for c in codes:
            hands[i][tiles.parse(c)] += 1
    first = {}
    for e in events:
        t = e.get("type")
        s = e.get("seat")
        if s is None or not (0 <= s < 4):
            continue
        if t == "tile_drawn":
            hands[s][tiles.parse(e["tile"])] += 1
            if s not in first:
                try:
                    if shantenmod.best_shanten(hands[s], melds[s]) == 0:
                        first[s] = hands[s][GOD]
                except Exception:
                    pass
        elif t == "tile_discarded":
            hands[s][tiles.parse(e["tile"])] -= 1
        elif t == "peng":
            hands[s][tiles.parse(e["tile"])] -= 2
            melds[s] += 1
        elif t == "gang":
            kind = str((e.get("data") or {}).get("kind") or "ming")
            tile = tiles.parse(e["tile"])
            if kind == "an":
                hands[s][tile] -= 4
                melds[s] += 1
            elif kind == "bu":
                hands[s][tile] -= 1
            else:
                hands[s][tile] -= 3
                melds[s] += 1
        elif t == "chi":
            data = e.get("data") or {}
            called = tiles.parse(e["tile"]) if e.get("tile") else None
            for c in (data.get("tiles") or []):
                cc = tiles.parse(c)
                if cc == called:
                    continue
                hands[s][cc] -= 1
            melds[s] += 1
    return first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3-limit", type=int, default=300,
                    help="T3（到听时刻 god 分布）只取前 N 个文件——它要对每次摸牌算 best_shanten，全量代价过高")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(ROOT / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    print(f"扫描 {len(paths)} 个小局文件")

    rows = []
    wait_gods = {"us": collections.Counter(), "opp": collections.Counter()}
    t3_files = 0
    for idx, p in enumerate(paths):
        try:
            our, out = B.analyze_file(p)
        except Exception:
            continue
        if not out:
            continue
        if idx < args.t3_limit:
            try:
                doc = json.loads(Path(p).read_text(encoding="utf-8"))
                rounds = B.group_rounds(doc)
                t3_files += 1
                for rn, r in rounds.items():
                    if not r["start_hands"]:
                        continue
                    first = first_waiting_gods(r["start_hands"], r["events"], our)
                    for seat, g in first.items():
                        side = "us" if seat == our else "opp"
                        wait_gods[side][g] += 1
            except Exception:
                pass
        rows.extend(out)

    print(f"有效胡牌局 {len(rows)}")
    checked = [r for r in rows if r["self_check_ok"]]

    # T1/T2：同 god 桶内，爆头率 + 副露数
    print("\n=== T1/T2 同 god 桶内：爆头率 与 副露数（us vs opp）===")
    print(f"{'god':>4} {'我们n':>6} {'我们爆头':>8} {'我们副露':>8} {'对手n':>6} {'对手爆头':>8} {'对手副露':>8}")
    for g in range(5):
        us = [r for r in checked if r["side"] == "us" and r["gods_at_win"] == g]
        op = [r for r in checked if r["side"] == "opp" and r["gods_at_win"] == g]
        if len(us) < 20 or len(op) < 20:
            continue
        ub = sum(1 for r in us if r["is_baotou"]) / len(us)
        ob = sum(1 for r in op if r["is_baotou"]) / len(op)
        um = sum(r["n_melds"] for r in us) / len(us)
        om = sum(r["n_melds"] for r in op) / len(op)
        print(f"{g:>4} {len(us):>6} {ub:>8.1%} {um:>8.2f} {len(op):>6} {ob:>8.1%} {om:>8.2f}")

    # T3：到听时刻 vs 胡牌时刻的 god 分布
    print("\n=== T3 到听时刻 god 分布（首次成听牌型）===")
    print(f"（抽样：前 {t3_files} 个文件，因需逐摸牌算 shanten）")
    for side in ("us", "opp"):
        c = wait_gods[side]
        tot = sum(c.values()) or 1
        dist = "  ".join(f"{k}:{v/tot:.1%}" for k, v in sorted(c.items()))
        print(f"  {side:4s} n={tot:5d}  {dist}")

    print("\n=== 胡牌时刻 god 分布（对照）===")
    for side in ("us", "opp"):
        c = collections.Counter(r["gods_at_win"] for r in checked if r["side"] == side)
        tot = sum(c.values()) or 1
        dist = "  ".join(f"{k}:{v/tot:.1%}" for k, v in sorted(c.items()))
        print(f"  {side:4s} n={tot:5d}  {dist}")

    # T2b：控制副露数，看爆头率差是否缩小
    print("\n=== T2b 同副露数内：爆头率（us vs opp）===")
    for m in range(4):
        us = [r for r in checked if r["side"] == "us" and r["n_melds"] == m]
        op = [r for r in checked if r["side"] == "opp" and r["n_melds"] == m]
        if len(us) < 20 or len(op) < 20:
            continue
        ub = sum(1 for r in us if r["is_baotou"]) / len(us)
        ob = sum(1 for r in op if r["is_baotou"]) / len(op)
        print(f"  副露 {m}: 我们 {ub:.1%} (n={len(us)})   对手 {ob:.1%} (n={len(op)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
