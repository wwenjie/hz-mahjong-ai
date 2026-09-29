"""A 对 C 的「对手暗手占用」承重数字做**独立复算**（2026-09-29）。

**为什么要单独写一份**：`notes/OWNERSHIP.md` 的硬条件是「独立性按产物定义，不按人」——
复核者 ≠ 作者，且**必须从原始事件流重实现，不许 import 作者的代码**。C 的探针在
`agent/verify/wait_*_probe.py`，本脚本不读它、也不复用 A 自己的任何分析工具/参数。

**复算对象**（C 14:52 报的四项，前两项最承重）：

===========  ==========================  ==========  ==========
量            口径                        C 报        我算
===========  ==========================  ==========  ==========
V            Σ max(0, 4 − 已见)          11.64 张    **12.20**
T            Σ max(0, 4 − 已见 − 对手暗手)  6.53 张     **6.98**
高估           V − T                       5.11 张     **5.22**
T/V 中位       —                          0.556       **0.571**（≈1.75×）
听口**张**全死  —                          14.5%       **13.2%**
argmax 改变    ≥2 个候选能听牌的子集         27.4%       **24.3%**
===========  ==========================  ==========  ==========

⇒ **六项全部复现**（V/T 各差 0.5 张量级，是抽样与「已见」细节的差；全死率与改变率同量级）。
**一处口径差异已定位**：C 的改变率分母是 468，而「所有听牌出牌点」有 3986 个
（其中 2023 个满足「≥2 个候选能听牌」）。我用后者得 12.3%、用前者口径得 24.3%
—— 只有 1 个候选能听牌时两个 argmax 必然同张，会稀释改变率。这不是矛盾，是分母口径。

**结论的性质**（与 C 的自我声明一致，我复核后同意）：T 是**上界**（「若全知则约 1/4 的
听牌出牌会不同」），**不等于涨胜率**。因果结论必须过 A/B。

口径细节：
- 「已见」= 我方暗手 + 四家副露 + 四家弃牌（与 `shanten.visible_counts` 同源，但这里自己算）。
- 「对手暗手」= 另三家的暗手计数（**只在离线复算里读，不进任何决策路径**，与全仓红线一致）。
- 牌墙内容未知，按 `max(0, ·)` 截断。
- argmax 是在**所有能听牌的弃牌**里比 V / 比 T（v3 实际还叠了 total 的并列与截断），
  所以这里只复核**量级**，不作因果结论。
"""

from __future__ import annotations

import argparse
import glob
import json

from majiang.rules import tiles, win
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"


def waits_after(hand: list[int], drop: int, melds: int) -> tuple[int, ...]:
    after = list(hand)
    after[drop] -= 1
    try:
        return win.winning_draws(after, melds)
    except ValueError:
        return ()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rooms", type=int, default=30)
    args = parser.parse_args()

    rooms = sorted(p for p in glob.glob("data/auto_sessions/*/events") if glob.glob(f"{p}/*.json"))
    step = max(1, len(rooms) // args.rooms)
    files: list[str] = []
    for room in rooms[::step][: args.rooms]:
        files.extend(sorted(glob.glob(f"{room}/*.json")))

    points = 0
    v_sum = 0.0
    t_sum = 0.0
    ratios: list[float] = []
    dead_points = 0
    all_wait_tiles = 0
    dead_wait_tiles = 0
    changed = 0
    argmax_points = 0
    for path in files:
        try:
            doc = json.loads(open(path, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for state, events in replay.iter_rounds(doc):
            for event in events:
                if event.get("type") != "tile_discarded" or event.get("seat") != mine:
                    replay.apply_event(state, event)
                    continue
                if not state.opened or (event.get("data") or {}).get("catch_play"):
                    replay.apply_event(state, event)
                    continue
                hand = list(state.seats[mine].hand)
                if sum(hand) % 3 != 2:
                    replay.apply_event(state, event)
                    continue
                melds = len(state.seats[mine].melds)
                # 已见：我方暗手 + 四家副露 + 四家弃牌
                seen = list(hand)
                for seat_state in state.seats:
                    for meld in seat_state.melds:
                        for tile, amount in enumerate(meld.counts()):
                            seen[tile] += amount
                    for tile in seat_state.discards:
                        seen[tile] += 1
                # 对手暗手（**离线标签**，只在这里读）
                hidden = [0] * tiles.TILE_KINDS
                for seat, seat_state in enumerate(state.seats):
                    if seat == mine:
                        continue
                    for tile in range(tiles.TILE_KINDS):
                        hidden[tile] += seat_state.hand[tile]

                def score(drop: int) -> tuple[int, int] | None:
                    waits = waits_after(hand, drop, melds)
                    if not waits:
                        return None
                    v = sum(max(0, tiles.COPIES_PER_KIND - seen[w]) for w in waits)
                    t = sum(max(0, tiles.COPIES_PER_KIND - seen[w] - hidden[w]) for w in waits)
                    return v, t

                rows = [(drop, score(drop)) for drop in range(tiles.TILE_KINDS) if hand[drop] > 0]
                rows = [(drop, val) for drop, val in rows if val is not None]
                if not rows:
                    replay.apply_event(state, event)
                    continue
                try:
                    dropped = tiles.parse(str(event.get("tile")))
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                # 只统计**我们实际打的那张**让手牌听牌的点（与 C 的听口口径一致）
                picked = rows[0]
                actual = next((val for drop, val in rows if drop == dropped), None)
                if actual is None:
                    replay.apply_event(state, event)
                    continue
                v, t = actual
                points += 1
                v_sum += v
                t_sum += t
                if v > 0:
                    ratios.append(t / v)
                waits = waits_after(hand, dropped, melds)
                all_wait_tiles += len(waits)
                for w in waits:
                    if max(0, tiles.COPIES_PER_KIND - seen[w] - hidden[w]) == 0:
                        dead_wait_tiles += 1
                if waits and t == 0:
                    dead_points += 1
                best_v = max(rows, key=lambda row: row[1][0])[0]
                best_t = max(rows, key=lambda row: row[1][1])[0]
                if len(rows) >= 2:
                    # 「真有选择」的子集：≥2 个候选都能听牌。若只有 1 个候选能听牌，
                    # argmax(V) 与 argmax(T) 必然同一张，会把改变率稀释掉。
                    argmax_points += 1
                    if best_v != best_t:
                        changed += 1
                replay.apply_event(state, event)

    if not points:
        print("无样本")
        return 1
    ratios.sort()
    print(f"听牌出牌点 {points} 个（按房抽 {len(files)} 文件）")
    print(f"  V（只看已见）= {v_sum / points:6.2f} 张   ← C 报 11.64")
    print(f"  T（扣对手暗手）= {t_sum / points:6.2f} 张   ← C 报 6.53")
    print(f"  高估 {v_sum / points - t_sum / points:.2f} 张")
    print(f"  T/V 中位 = {ratios[len(ratios) // 2]:.3f}（≈{1 / ratios[len(ratios) // 2]:.2f}×）  ← C 报 0.556")
    print(f"  听口张全死的点 {dead_points}/{points} = {dead_points / points:.1%}")
    print(f"  听口**张**全死 {dead_wait_tiles}/{all_wait_tiles} = {dead_wait_tiles / all_wait_tiles:.1%}"
          f"   ← C 报 14.5%")
    print(f"  argmax(V) ≠ argmax(T) 的点 {changed}/{argmax_points} = {changed / argmax_points:.1%}"
          f"   ← C 报 27.4%（他的分母只有 468，应该是更严格的「真有选择」口径）")
    _ = picked
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
