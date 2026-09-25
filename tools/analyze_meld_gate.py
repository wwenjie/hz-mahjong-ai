"""诊断吃碰闸门：我们在真实对局里到底拒绝了多少副露，以及为什么拒（tasks.md 5.5）。

**动机**：真实数据显示我们每局副露 **0.59 次**，对手 **1.09 次**，而胡牌时已出牌手数完全一样
（7.8 vs 7.7）——差距不是「慢」，是「成牌机会少」。当前闸门有三道：

1. 手牌 ≥5 对（七对路线）→ **一律不吃不碰**
2. 副露后向听必须**严格下降**（``after < current``）
3. 抓打圈内不能吃碰明杠（规则层，`legal_actions` 已体现）

本工具不猜、不另写一套逻辑：它把真实对局里**我们的每一个碰/吃机会**重建成 ``Situation``，
用与线上**同一套** ``legal_actions`` 与策略方法算 ``current``/``after``，所以统计出来的就是
线上闸门的真实行为。它只回答「决策面有多少差异、差异在哪里」；**不做前进模拟**——
那要仿真器，见 ``tools/ab_test.py`` 的配对自对弈。

用法::

    uv run python tools/analyze_meld_gate.py --manifest notes/manifest-20260926.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules.action import CHI, PENG, chi_combinations, legal_actions
from majiang.rules.hand import Hand
from majiang.rules.melds import CHI_MAX_PER_HAND
from majiang.rules.melds import chi_count
from majiang.rules.situation import PHASE_RESPONSE_CHI, PHASE_RESPONSE_PENG
from majiang.sim import replay
from majiang.strategy.policy import HeuristicDecider, PolicyConfig

SEATS = 4
DISCARDED = "tile_discarded"
OPPORTUNITIES = "碰/吃机会"


def _could(state: replay.ReplayState, mine: int, discarder: int, tile: int) -> tuple[bool, bool]:
    """我们在这个响应窗口里能不能碰 / 能不能吃。财神不可被吃碰（规则强制）。"""
    if tile == tiles.GOD:
        return False, False
    hand = state.seats[mine].hand
    could_peng = hand[tile] >= 2
    could_chi = (
        (discarder + 1) % SEATS == mine
        and chi_count(state.seats[mine].melds) < CHI_MAX_PER_HAND
        and bool(chi_combinations(hand, tile))
    )
    return could_peng, could_chi


def _record(
    state: replay.ReplayState,
    mine: int,
    discarder: int,
    tile: int,
    decider: HeuristicDecider,
    stats: Counter,
    samples: list[str],
) -> None:
    could_peng, could_chi = _could(state, mine, discarder, tile)
    if not (could_peng or could_chi) or not state.opened:
        return
    stats[OPPORTUNITIES] += 1

    seat_state = state.seats[mine]
    hand = Hand.from_counts(seat_state.hand, seat_state.melds)
    try:
        current = shanten_module.shanten_any(hand.counts, hand.meld_count)
    except Exception:  # noqa: BLE001
        stats["向听计算失败"] += 1
        return

    pairs = sum(amount // 2 for amount in hand.counts if amount >= 2)
    stats[f"机会时的向听={current}"] += 1
    stats[f"机会时的对数={min(pairs, 7)}"] += 1
    stats[f"机会时已有副露={hand.meld_count}"] += 1

    if decider._is_pair_route(state.situation_for(mine)):  # noqa: SLF001
        stats["闸门①：被七对路线拦下"] += 1

    for label, kind, phase in (
        ("碰", PENG, PHASE_RESPONSE_PENG),
        ("吃", CHI, PHASE_RESPONSE_CHI),
    ):
        if (kind == PENG and not could_peng) or (kind == CHI and not could_chi):
            continue
        situation = state.situation_for(mine, phase=phase, offered=tile, responding=(mine,))
        actions = legal_actions(situation)
        if not any(action.kind == kind for action in actions):
            stats[f"{label}：规则层否决（抓打圈/上限）"] += 1
            continue
        after = decider._shanten_after_meld(situation, tile, kind)  # noqa: SLF001
        if after is None:
            stats[f"{label}：无法评估"] += 1
            continue
        delta = after - current
        # 交叉表：把「机会时的向听」带进 key，否则看不出被拒的到底是不是残局。
        # 0 向听时拒绝副露通常是**对的**（已经听牌，副露只会换掉听口），若被拒的
        # 大多是 0/1 向听，说明闸门没问题；若散在 2~4 向听，才是过紧。
        stats[f"{label} @向听{current} 变化{delta:+d}"] += 1
        if delta < 0:
            stats[f"{label}：闸门②放行"] += 1
        else:
            stats[f"{label}：闸门②拒绝（向听未降）"] += 1
            if len(samples) < 15:
                samples.append(
                    f"{label} 向听{current} 对数{pairs} 已有副露{hand.meld_count} "
                    f"牌={tiles.to_code(tile)}"
                )


def scan(
    payload: dict,
    ours: str,
    decider: HeuristicDecider,
    stats: Counter,
    samples: list[str],
) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != 4 or ours not in ids:
        return
    mine = ids.index(ours)

    for state, events in replay.iter_rounds(payload):
        for event in events:
            if str(event.get("type")) == DISCARDED:
                seat = event.get("seat")
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if isinstance(seat, int) and 0 <= seat < SEATS and seat != mine and tile is not None:
                    _record(state, mine, seat, tile, decider, stats, samples)
            replay.apply_event(state, event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="诊断吃碰闸门")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
    paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("清单为空", file=sys.stderr)
        return 1

    decider = HeuristicDecider(PolicyConfig())
    stats: Counter = Counter()
    samples: list[str] = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            scan(payload, args.ours, decider, stats, samples)
        except Exception as exc:  # noqa: BLE001
            stats[f"跳过:{type(exc).__name__}"] += 1

    print(f"文件 {len(paths)} 个，视角 {args.ours}")
    print()
    opportunities = stats[OPPORTUNITIES]
    print(f"{OPPORTUNITIES:<28} {opportunities:>8}")
    for key in sorted(
        (k for k in stats if k != OPPORTUNITIES), key=lambda k: (-stats[k], k)
    ):
        print(f"{key:<28} {stats[key]:>8}")
    if samples:
        print()
        print("被拒机会样例:")
        for line in samples:
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
