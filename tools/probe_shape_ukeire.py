"""中段度量改动的**离线机制验证**：新口径选的牌，进张是否更多？（只读）

**为什么需要它**：自对弈 A/B 一晚只能跑一两条、且与 RL 线争 CPU，而「机制量优先」
要求我们先拿到机制证据。这条提供**几分钟级**的离线机制门：

对真机局面，取「同最小向听且新旧口径**分歧**」的决策点，比较两个选择
打完之后各自的**精确进张张数**（`shanten.ukeire` 的剩余张数之和，扣可见牌）。
判据：均值 > 0 ⇒ 形质项确实在做牌效；≈0 或 <0 ⇒ 先调权重，不要拿 A/B 正号覆盖。

**判据以「中位 + 胜负比」为主，均值仅供参考**（实测方差极大：2 文件时均值 **−9.25**、
10 文件时 **+2.67**、40 文件时 **+1.48**，均值会被离群点主导）。

**最新结果（2026-09-29，40 文件 / 153 个分歧点）**：
均值 **+1.48** / 中位 **+0** / p75 **+4** / 极值 [−32, +39]；
**新更好 76 / 旧更好 34 / 持平 43（2.2:1）**。
⇒ **形质项在多数分歧点上更优，但不是压倒性**——所以仍需自对弈机制门（到听摸序）与 A/B 佐证。

用法::

    uv run python tools/probe_shape_ukeire.py --limit 10
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.rules import tiles
from majiang.sim import replay
from majiang.strategy.risk import visible_need

SEATS = 4
OUR = "u_a7f7c67bb14a"


def ukeire_copies(counts: list[int], melds: int, vis: list[int], memo: dict) -> int:
    return sum(copy for _, copy in sm.ukeire(counts, melds, visible=vis, memo=memo))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="中段度量改动的离线机制验证")
    parser.add_argument("--limit", type=int, default=10, help="读几个房文件（每房 10 个）")
    parser.add_argument("--new-only", action="store_true", help="只测新口径（对旧口径的增益）")
    args = parser.parse_args(argv)

    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[: args.limit]
    gains: list[int] = []
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}", file=sys.stderr)
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != SEATS or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for state, events in replay.iter_rounds(doc):
            for e in events:
                if not (
                    e.get("type") == "tile_discarded"
                    and e.get("seat") == mine
                    and state.opened
                    and not (e.get("data") or {}).get("catch_play")
                ):
                    replay.apply_event(state, e)
                    continue
                hand = list(state.seats[mine].hand)
                if sum(hand) % 3 == 2:
                    melds = len(state.seats[mine].melds)
                    rows = []
                    for drop in range(tiles.TILE_KINDS):
                        if hand[drop] <= 0:
                            continue
                        after = list(hand)
                        after[drop] -= 1
                        try:
                            value = sm.shanten_any(after, melds)
                        except Exception:  # noqa: BLE001
                            continue
                        blocks = sm.quick_blocks(after)
                        rows.append((value, 2 * blocks[0] + blocks[1],
                                     sm.shape_value(after, melds), drop))
                    if len(rows) >= 2:
                        best = min(r[0] for r in rows)
                        cands = [r for r in rows if r[0] == best]
                        if len(cands) >= 2:
                            pick_old = max(cands, key=lambda r: r[1] - 3.0 * visible_need(r[3]))
                            pick_new = max(cands, key=lambda r: r[2] - 3.0 * visible_need(r[3]))
                            if pick_old[3] != pick_new[3]:
                                vis = sm.visible_counts(
                                    hand,
                                    [m.tiles for s in state.seats for m in s.melds],
                                    [list(s.discards) for s in state.seats],
                                )
                                memo: dict = {}

                                def after_of(drop: int) -> list[int]:
                                    out = list(hand)
                                    out[drop] -= 1
                                    return out

                                gains.append(
                                    ukeire_copies(after_of(pick_new[3]), melds, vis, memo)
                                    - ukeire_copies(after_of(pick_old[3]), melds, vis, memo)
                                )
                replay.apply_event(state, e)

    if not gains:
        print("无分歧样本")
        return 1
    gains.sort()
    mean = sum(gains) / len(gains)
    median = gains[len(gains) // 2]
    win = sum(1 for g in gains if g > 0)
    lose = sum(1 for g in gains if g < 0)
    print(f"分歧决策点 {len(gains)} 个（文件 {len(files)}）")
    print(f"  新口径 − 旧口径 的精确进张：**均值 {mean:+.2f} / 中位 {median:+d} 张**")
    print(f"  分位 p25 {gains[len(gains) // 4]:+d} / p75 {gains[3 * len(gains) // 4]:+d}"
          f" / 极值 [{gains[0]:+d}, {gains[-1]:+d}]")
    print(f"  新更好 {win} / 旧更好 {lose} / 持平 {len(gains) - win - lose}")
    print("\n**判据以「中位 + 胜负比」为主，均值仅供参考**——"
          "实测方差极大（2 文件时均值 −9.25、10 文件时 +2.67），均值会被离群点主导。"
          "\n中位 >0 且胜负比 >2:1 ⇒ 形质项确实在做牌效；否则先调权重。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
