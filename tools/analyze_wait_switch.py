"""换听机会量化：我们处于「听口很窄的听牌态」有多频繁，退回一向听能换到多少张。

**要回答的问题**：本平台**只能自摸**，所以听口剩余张数几乎等于胡牌概率
（听 2 张 vs 听 8 张差 4 倍）。而我们的出牌评分里向听权重是 10——
从 0 向听退回 1 向听等于 −10 分，**永远补不回来**，因此我们**从不**为了换好形而退回一向听。

但「听口窄时宁可退回一向听」在自摸制下有真实理论支持。改之前先量机会规模：
如果「窄听口 + 存在好得多的 1 向听替代」的情形很罕见，这个杠杆就不值得做。

**口径**：在**我们自己的出牌时刻**（`tile_discarded` 事件之前取局面），
对每个可打出的候选 `t` 算打完之后的精确向听：

- 打完是 **0 向听（听牌）** → 用 `win.winning_draws` 求听口，用 `visible_counts` 求剩余张数
- 打完是 **1 向听** → 用精确 `ukeire` 求「能推进到听牌的剩余张数」

记录实际打出的那一张的（向听, 张数），以及**所有 1 向听替代里最好的那个张数**。

**判决量**：`最好 1 向听替代的张数 / 实际听口的张数`。远大于 1 才说明「值得换」。

用法::

    uv run python tools/analyze_wait_switch.py --limit 60
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win as win_module
from majiang.rules.hand import Hand
from majiang.sim import replay

SEATS = 4
NARROW = 6  # 听口剩余张数低于这个值算「窄」


def copies_of(state: replay.ReplayState, seat: int, kinds: tuple[int, ...]) -> int:
    visible = shanten_module.visible_counts(
        list(state.seats[seat].hand),
        [meld.tiles for item in state.seats for meld in item.melds],
        [list(item.discards) for item in state.seats],
    )
    return sum(max(0, tiles.COPIES_PER_KIND - visible[tile]) for tile in kinds)


def scan(payload: dict, ours: str, waits: list[int], alts: list[tuple[int, int]],
         counts: Counter) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or ours not in ids:
        return
    mine = ids.index(ours)

    for state, events in replay.iter_rounds(payload):
        for event in events:
            if str(event.get("type")) != "tile_discarded" or event.get("seat") != mine:
                replay.apply_event(state, event)
                continue
            played = replay._tile_of(event.get("tile"))  # noqa: SLF001
            seat_state = state.seats[mine]
            hand = list(seat_state.hand)
            meld_count = len(seat_state.melds)
            if played is None or hand[played] <= 0:
                replay.apply_event(state, event)
                continue

            candidates = [
                tile for tile in range(tiles.TILE_KINDS) if hand[tile] > 0 and tile != tiles.GOD
            ]
            if len(candidates) < 2:
                replay.apply_event(state, event)
                continue
            visible = shanten_module.visible_counts(
                hand,
                [meld.tiles for item in state.seats for meld in item.melds],
                [list(item.discards) for item in state.seats],
            )
            actual_wait: int | None = None
            best_alt: int | None = None
            for tile in candidates:
                after = list(hand)
                after[tile] -= 1
                try:
                    value = shanten_module.shanten_any(after, meld_count)
                except Exception as exc:  # noqa: BLE001
                    counts[f"向听失败:{type(exc).__name__}"] += 1
                    continue
                if value == 0:
                    waits_kinds = win_module.winning_draws(after, meld_count)
                    copies = sum(
                        max(0, tiles.COPIES_PER_KIND - visible[kind]) for kind in waits_kinds
                    )
                    if tile == played:
                        actual_wait = copies
                elif value == 1:
                    # 精确进张（42–157 ms/次），只对 1 向听候选算
                    entries = shanten_module.ukeire(after, meld_count, visible=visible)
                    copies = sum(copy for _kind, copy in entries)
                    if best_alt is None or copies > best_alt:
                        best_alt = copies

            counts["我们的出牌时刻"] += 1
            if actual_wait is None:
                replay.apply_event(state, event)
                continue
            counts["实际打出后是听牌"] += 1
            waits.append(actual_wait)
            if best_alt is not None:
                alts.append((actual_wait, best_alt))
                counts["同时存在 1 向听替代"] += 1
            replay.apply_event(state, event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="换听机会量化")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.events))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有匹配到事件流", file=sys.stderr)
        return 1

    waits: list[int] = []
    alts: list[tuple[int, int]] = []
    counts: Counter = Counter()
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        try:
            scan(payload, args.ours, waits, alts, counts)
        except Exception as exc:  # noqa: BLE001 —— 带异常类型，避免裸 except 吞掉
            counts[f"跳过:{type(exc).__name__}"] += 1

    print(f"文件 {len(paths)} 个")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    if not waits:
        print("没有样本 —— 先检查「实际打出后是听牌」是否为 0")
        return 1

    waits.sort()
    print()
    print(f"实际听口的剩余张数：中位 {stats.median(waits):.0f} / 均值 {stats.mean(waits):.1f}")
    print(f"  窄（<{NARROW} 张）占比 {sum(1 for w in waits if w < NARROW) / len(waits):.1%}")
    for bound in (2, 4, 6, 8):
        share = sum(1 for w in waits if w < bound) / len(waits)
        print(f"  听口 <{bound} 张的占比 {share:.1%}")
    if not alts:
        print("没有可比样本（1 向听替代全为空）")
        return 0

    print()
    print(f"「窄听口 且 存在 1 向听替代」的样本 {len(alts)}")
    ratios = []
    for wait, alt in alts:
        if wait <= 0:
            continue
        ratios.append(alt / wait)
    if ratios:
        print(f"  改善倍数（最好1向听张数 / 当前听口张数）：中位 {stats.median(ratios):.2f} "
              f"/ 均值 {stats.mean(ratios):.1f}")
        for bound in (1.5, 2.0, 3.0):
            share = sum(1 for r in ratios if r >= bound) / len(ratios)
            print(f"  改善 ≥{bound}x 的占比 {share:.1%}")
    narrow = [(w, a) for w, a in alts if w < NARROW]
    if narrow:
        print(f"  其中**窄听口**（<{NARROW} 张）的 {len(narrow)} 例："
              f"当前听口中位 {stats.median([w for w, _ in narrow]):.0f} 张，"
              f"替代中位 {stats.median([a for _, a in narrow]):.0f} 张")
    print()
    print("判决口径：**改善倍数远大于 1 才值得为换形而退回一向听**。"
          "这只是机会规模，是否真划算要看配对自对弈（换听会牺牲当下的自摸机会）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
