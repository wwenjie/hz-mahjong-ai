"""从真实对局事件流测量各参与者的战力（tasks.md 6A.5 / 6A.4）。

回答一个具体问题：**我们和自己的对手比，差在哪里**。仅看总分很难判断，因为番数、
胜率、听牌速度三者会互相掩盖。本工具把每一条拆开，按身份聚合：

- ``hands`` / ``wins`` / ``win_rate`` —— 胡牌率（本平台只能自摸，等强时约 25%）
- ``avg_fan`` / ``>1番占比`` —— 番数是否吃得开
- ``score_per_hand`` —— 每手净分（平台第一裁决键）
- ``tenpai_rate`` —— **该玩家每次出牌时处于听牌（0 向听）的比例**

``tenpai_rate`` 是关键解释量：番数与胜率接近而总分偏低，通常意味着**成牌更慢**；
若听牌率相当而胜率偏低，则问题在终局（该胡不胡、或听了却没进张）。

听牌判定用 ``quick_shanten``（贪心近似）而非精确向听——量级快两个数量级，且对每个
参与者用同一把尺子，比较仍然有效。近似值只用于**横向对比**，不用于绝对判断。

用法::

    uv run python tools/measure_strength.py --min-hands 200
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

from majiang.rules.shanten import quick_shanten
from majiang.sim import replay

DRAW = "tile_discarded"


class Aggregate:
    __slots__ = ("hands", "wins", "score", "fan", "discard_moments", "tenpai_moments")

    def __init__(self) -> None:
        self.hands = 0
        self.wins = 0
        self.score = 0
        self.fan = 0
        self.discard_moments = 0
        self.tenpai_moments = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.hands if self.hands else 0.0

    @property
    def score_per_hand(self) -> float:
        return self.score / self.hands if self.hands else 0.0

    @property
    def avg_fan(self) -> float:
        return self.fan / self.wins if self.wins else 0.0

    @property
    def tenpai_rate(self) -> float:
        return self.tenpai_moments / self.discard_moments if self.discard_moments else 0.0


def is_tenpai(seat: replay.ReplaySeat) -> bool:
    """该座位此刻是否听牌（贪心近似，足够用于横向对比）。"""
    try:
        return quick_shanten(list(seat.hand), len(seat.melds)) <= 0
    except Exception:  # noqa: BLE001 —— 单点失败不应中断整批统计
        return False


def process(payload: dict, per_user: dict[str, Aggregate]) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != 4:
        return
    state = replay.from_payload(payload)
    # 出牌时刻的听牌率：只统计「明手牌数正常」的座位，避免把手牌不完整的时刻算进去
    for event in replay.all_events(payload):
        if str(event.get("type")) == DRAW:
            seat = event.get("seat")
            if isinstance(seat, int) and 0 <= seat < 4:
                bucket = per_user[ids[seat]]
                bucket.discard_moments += 1
                if is_tenpai(state.seats[seat]):
                    bucket.tenpai_moments += 1
        replay.apply_event(state, event)
    for result in payload.get("rounds") or []:
        scores = result.get("scores") or []
        if len(scores) != 4:
            continue
        for index, uid in enumerate(ids):
            bucket = per_user[uid]
            bucket.hands += 1
            bucket.score += int(scores[index])
        if result.get("is_draw"):
            continue
        winner = int(result.get("winner", -1))
        if 0 <= winner < 4:
            bucket = per_user[ids[winner]]
            bucket.wins += 1
            bucket.fan += int(result.get("multiplier", 1) or 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从真实对局测量各参与者战力")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument("--ours", default="u_a7f7c67bb14a", help="我们的 user_id，用于高亮")
    parser.add_argument("--min-hands", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 局（0 = 全部）")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.events))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有匹配到事件流", file=sys.stderr)
        return 1

    per_user: dict[str, Aggregate] = defaultdict(Aggregate)
    games = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            process(payload, per_user)
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        games += 1

    rows = [(uid, agg) for uid, agg in per_user.items() if agg.hands >= args.min_hands]
    rows.sort(key=lambda kv: -kv[1].score_per_hand)
    total_hands = sum(agg.hands for _, agg in rows) // 4 if rows else 0
    print(f"对局文件 {games} 个；参与人数 {len(per_user)}，其中手数 ≥{args.min_hands} 的 {len(rows)} 人")
    print(f"（手数一栏为各自的样本量；'听牌率' = 每次出牌时处于听牌的近似比例）")
    print()
    print(f"{'#':>3} {'身份':>7} {'手数':>6} {'胜率':>7} {'均番':>6} {'听牌率':>7} {'每手分':>8} {'总净分':>9}")
    for index, (uid, agg) in enumerate(rows, 1):
        tag = " ← 我们" if uid == args.ours else ""
        print(
            f"{index:>3} {uid[-6:]:>7} {agg.hands:>6} {agg.win_rate:>6.1%} "
            f"{agg.avg_fan:>6.3f} {agg.tenpai_rate:>6.1%} {agg.score_per_hand:>+8.3f} "
            f"{agg.score:>+9d}{tag}"
        )
    ours = per_user.get(args.ours)
    if ours is not None:
        field = Aggregate()
        for uid, agg in rows:
            if uid == args.ours:
                continue
            field.hands += agg.hands
            field.wins += agg.wins
            field.score += agg.score
            field.fan += agg.fan
            field.discard_moments += agg.discard_moments
            field.tenpai_moments += agg.tenpai_moments
        print()
        print(f"我们（{ours.hands} 手）vs 其余 {len(rows) - 1} 人（合计 {field.hands} 手）:")
        print(
            f"  胜率 {ours.win_rate:.1%} vs {field.win_rate:.1%}   "
            f"均番 {ours.avg_fan:.3f} vs {field.avg_fan:.3f}   "
            f"听牌率 {ours.tenpai_rate:.1%} vs {field.tenpai_rate:.1%}   "
            f"每手分 {ours.score_per_hand:+.3f} vs {field.score_per_hand:+.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
