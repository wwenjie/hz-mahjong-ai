"""量化「弃胡求生」的代价（tasks.md 5.10 的实测校验）。

默认策略的 ``chase_baotou`` 会在**能自摸胡牌时故意不胡**，改为打出一张牌做成「听任意」
（爆头）以求番数翻倍。自对弈里它换到番 +27% / 胡牌率 −11%，但真机数据显示我们的番数
**并不比全场高**（1.268 vs 1.316），等于白丢胜率——本工具就是从真实对局里直接数：

1. 我们每次摸牌后**手牌已成胡牌型**的时刻有多少；
2. 其中当场胡掉的多少、**弃胡**的多少；
3. 弃胡的那些局最后怎么收场（自己后来胡了 / 别人胡了 / 流局），拿到的番数比当场胡
   少多少。

判断完全来自事件流，不读日志：摸牌后手牌成胡，若紧跟着的本人事件是 ``tile_discarded``
就是弃胡，若是 ``round_ended``（且赢家是我们）就是当场胡。

用法::

    uv run python tools/analyze_declined_wins.py
    uv run python tools/analyze_declined_wins.py --ours u_a7f7c67bb14a --min-round 4
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from majiang.rules import score as score_module
from majiang.rules import win
from majiang.rules.fan import compute_fan
from majiang.sim import replay

SEATS = 4
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"


def hand_after(state: replay.ReplayState, seat: int) -> list[int]:
    return list(state.seats[seat].hand)


def immediate_fan(state: replay.ReplayState, seat: int, drawn: int) -> int | None:
    """此刻自摸的番数（0 表示不成胡）。与 ``sim.round.settle_win`` 同口径。"""
    seat_state = state.seats[seat]
    waiting = list(seat_state.hand)
    if waiting[drawn] <= 0:
        return None
    waiting[drawn] -= 1
    try:
        result = compute_fan(
            waiting,
            drawn,
            len(seat_state.melds),
            chain_count=seat_state.chain_count,
            piao_count=seat_state.piao_count,
        )
    except Exception:  # noqa: BLE001 —— 单点判定失败不应中断整批
        return None
    return int(result.fan) if result.hu else None


def scan(payload: dict, ours: str, stats: Counter, examples: list[str], base_score: int) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != 4 or ours not in ids:
        return
    mine = ids.index(ours)
    official = {
        int(entry.get("round_no", 0) or 0): entry for entry in (payload.get("rounds") or [])
    }

    for state, events in replay.iter_rounds(payload):
        declined_at: int | None = None
        declined_fan: int | None = None
        declined_baotou = False
        for index, event in enumerate(events):
            kind = str(event.get("type"))
            seat = event.get("seat")
            if kind == DRAWN and seat == mine:
                drawn = replay._tile_of(event.get("tile"))  # noqa: SLF001 —— 同一模块内的解析
                if drawn is None:
                    continue
                replay.apply_event(state, event)
                if not win.is_winning_shape(hand_after(state, mine), len(state.seats[mine].melds)):
                    continue
                fan = immediate_fan(state, mine, drawn)
                if fan is None:
                    continue
                stats["能胡的摸牌时刻"] += 1
                # 紧跟着的本人事件决定「当场胡」还是「弃胡」
                following = next(
                    (
                        str(item.get("type"))
                        for item in events[index + 1 :]
                        if item.get("seat") == mine
                        or str(item.get("type")) in (replay.ROUND_ENDED, replay.GAME_ENDED)
                    ),
                    None,
                )
                if following == DISCARDED:
                    stats["弃胡"] += 1
                    declined_at = index
                    declined_fan = fan
                    declined_baotou = win.is_baotou(
                        hand_after(state, mine), len(state.seats[mine].melds)
                    )
                    if len(examples) < 12:
                        examples.append(
                            f"{payload.get('game_id')} 局{state.round_no} seq{event.get('seq')} "
                            f"弃掉 {fan} 番"
                        )
                else:
                    stats["当场胡"] += 1
                continue
            replay.apply_event(state, event)

        if declined_at is None:
            continue
        entry = official.get(state.round_no) or {}
        winner = None if entry.get("is_draw") else entry.get("winner")
        if winner is None:
            stats["弃胡后：流局"] += 1
        elif int(winner) == mine:
            got = int(entry.get("multiplier", 1) or 1)
            stats["弃胡后：自己还是胡了"] += 1
            stats["弃胡后：自己胡了的番数合计"] += got
            if declined_fan is not None:
                stats["弃胡让出的番数合计"] += declined_fan
            if got >= (declined_fan or 0) * 2:
                stats["  其中番数确实翻倍了"] += 1
        else:
            stats["弃胡后：别人胡了"] += 1
        if declined_baotou:
            stats["  弃胡时本已是爆头"] += 1

        # 反事实：当场胡会拿多少分 vs 实际拿到多少分。这是判据本身。
        scores = entry.get("scores") or []
        if len(scores) != 4 or declined_fan is None:
            stats["反事实无法计算"] += 1
            continue
        dealer = int(entry.get("dealer", state.dealer) or 0)
        if not 0 <= dealer < SEATS:
            dealer = state.dealer
        would = score_module.seat_deltas(
            declined_fan, base_score, winner_seat=mine, dealer_seat=dealer
        )[mine]
        actual = int(scores[mine])
        stats["反事实：当场胡的净分"] += would
        stats["反事实：弃胡实际净分"] += actual
        if actual > would:
            stats["  （弃胡更赚的局数）"] += 1
        elif actual < would:
            stats["  （弃胡更亏的局数）"] += 1
        else:
            stats["  （打平）"] += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="量化弃胡求爆头的代价")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--base-score", type=int, default=1, help="底分，自动房为 1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.events))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有匹配到事件流", file=sys.stderr)
        return 1

    stats: Counter = Counter()
    examples: list[str] = []
    games = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            scan(payload, args.ours, stats, examples, args.base_score)
        except Exception as exc:  # noqa: BLE001 —— 单局失败不应中断整批
            stats[f"跳过:{type(exc).__name__}"] += 1
            continue
        games += 1

    print(f"对局 {games} 局（{len(paths)} 个文件），视角 {args.ours}")
    print()
    for key in (
        "能胡的摸牌时刻",
        "当场胡",
        "弃胡",
        "  弃胡时本已是爆头",
        "弃胡后：自己还是胡了",
        "  其中番数确实翻倍了",
        "弃胡后：别人胡了",
        "弃胡后：流局",
        "弃胡让出的番数合计",
        "弃胡后：自己胡了的番数合计",
        "反事实：当场胡的净分",
        "反事实：弃胡实际净分",
        "  （弃胡更赚的局数）",
        "  （弃胡更亏的局数）",
        "  （打平）",
        "反事实无法计算",
    ):
        value = stats.get(key, 0)
        total = stats.get("能胡的摸牌时刻", 0)
        share = f"  ({value / total:5.1%} of 能胡时刻)" if total and key in ("当场胡", "弃胡") else ""
        print(f"{key:<26} {value:>8}{share}")
    skipped = {k: v for k, v in stats.items() if k.startswith("跳过")}
    if skipped:
        print()
        print("其他:", skipped)
    if examples:
        print()
        print("弃胡样例（前 12 例）:")
        for line in examples:
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
