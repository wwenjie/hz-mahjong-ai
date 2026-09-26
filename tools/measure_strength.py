"""从真实对局事件流测量各参与者战力（tasks.md 6A.5 / 6A.4）。

回答两个具体问题：**我们和自己的对手比差在哪**，以及**换档位是否真的变强**。

### 只看总分是不够的

分数分布极度重尾（番数可达 512），所以 150–300 手的选手其每手分标准误能到 ±1 以上，
排行榜上那些小样本名次大多是运气。本工具把总分拆成四个可比的量：

- ``hands`` / ``wins`` / ``win_rate`` —— 胡牌率（本平台只能自摸，等强时约 25%）
- ``avg_fan`` —— 番数是否吃得开（重尾，小样本同样不可靠）
- ``tenpai_rate`` —— **每次出牌时处于听牌的比例**。低方差、无量纲，用来区分
  「成牌慢」与「终局处理差」
- ``score_per_hand`` —— 每手净分（平台第一裁决键）

听牌判定用 ``quick_shanten``（贪心近似）而非精确向听——快两个数量级，且对每个参与者
用同一把尺子，横向比较仍然有效；只用于对比，不用于绝对判断。

### 交错 A/B

``--by-decider`` 会用 ``sessions.jsonl`` 把每个房间归到当时使用的决策器，然后只统计
**该房间只用过一个决策器**的部分（混用房间排除并计数）。因为轮换是**按会话交错**的
（而不是先跑完 A 再跑 B），对手组合与时间漂移在两个臂之间被随机化。

用法::

    uv run python tools/measure_strength.py --min-hands 200
    uv run python tools/measure_strength.py --by-decider
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

DRAW = "tile_discarded"  # 注意：常量名沿用旧代码，实际是**出牌**事件（见 is_tenpai 的口径）
ROUND_ENDED = "round_ended"
DEFAULT_LEDGER = "data/auto_sessions/sessions.jsonl"


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

    def merge(self, other: Aggregate) -> None:
        self.hands += other.hands
        self.wins += other.wins
        self.score += other.score
        self.fan += other.fan
        self.discard_moments += other.discard_moments
        self.tenpai_moments += other.tenpai_moments

    def line(self, label: str) -> str:
        return (
            f"{label:>20} 手 {self.hands:>5}  胜率 {self.win_rate:>6.1%}  "
            f"均番 {self.avg_fan:>5.3f}  听牌率 {self.tenpai_rate:>6.1%}  "
            f"每手 {self.score_per_hand:>+7.3f}  合计 {self.score:>+7d}"
        )


def is_tenpai(seat: replay.ReplaySeat) -> bool:
    try:
        return quick_shanten(list(seat.hand), len(seat.melds)) <= 0
    except Exception:  # noqa: BLE001 —— 单点失败不应中断整批统计
        return False


def process(
    payload: dict,
    per_user: dict[str, Aggregate],
    ours: str,
    by_group: dict[str, Aggregate],
    group: str | None,
) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != 4:
        return
    mine = ids.index(ours) if ours in ids else None
    ours_agg = by_group.setdefault(group, Aggregate()) if group is not None else None

    # **局数与流局必须取自事件流，不能取自顶层 ``rounds[]``。**
    # agent B 独立复核发现：``rounds[]`` 对中间流局的收录是**有损的**，且同一批数据里
    # 三种行为并存（994 文件全收 / 110 只收非流局 / 16 只收末位流局），与时间无关。
    # 用它当分母会漏掉约 192 条中间流局，使**有胡率虚高**（原报 99.5%，真值 97.4%），
    # 公平份额也随之虚高（24.9% → 24.36%）。``round_ended.data.draw`` 是唯一可靠来源。
    rounds_total = 0
    draws_total = 0
    for state, events in replay.iter_rounds(payload):
        rounds_total += 1
        for event in events:
            kind = str(event.get("type"))
            if kind == DRAW:
                seat = event.get("seat")
                if isinstance(seat, int) and 0 <= seat < 4:
                    tenpai = is_tenpai(state.seats[seat])
                    bucket = per_user[ids[seat]]
                    bucket.discard_moments += 1
                    bucket.tenpai_moments += int(tenpai)
                    if ours_agg is not None and seat == mine:
                        ours_agg.discard_moments += 1
                        ours_agg.tenpai_moments += int(tenpai)
            elif kind == ROUND_ENDED:
                data = event.get("data") or {}
                if data.get("draw"):
                    draws_total += 1
            replay.apply_event(state, event)

    summary = payload.get("rounds") or []
    for result in summary:
        scores = result.get("scores") or []
        if len(scores) != 4:
            continue
        for index, uid in enumerate(ids):
            bucket = per_user[uid]
            bucket.hands += 1
            bucket.score += int(scores[index])
        if ours_agg is not None and mine is not None:
            ours_agg.hands += 1
            ours_agg.score += int(scores[mine])
        if result.get("is_draw"):
            continue
        winner = int(result.get("winner", -1))
        if not 0 <= winner < 4:
            continue
        fan = int(result.get("multiplier", 1) or 1)
        per_user[ids[winner]].wins += 1
        per_user[ids[winner]].fan += fan
        if ours_agg is not None and winner == mine:
            ours_agg.wins += 1
            ours_agg.fan += fan

    # 摘要丢了中间流局 → 把缺的那几局补回**每一家**的手数分母。
    # 赢家与得分不受影响：被丢的都是流局（无赢家、四家 0 分）。
    missing = rounds_total - len(summary)
    if missing > 0:
        for bucket in per_user.values():
            bucket.hands += missing
        if ours_agg is not None:
            ours_agg.hands += missing
    return rounds_total, draws_total


def room_deciders(ledger: Path) -> dict[str, set[str]]:
    """房间 → 用过的决策器集合。"""
    mapping: dict[str, set[str]] = defaultdict(set)
    if not ledger.exists():
        return mapping
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        mapping[str(record.get("room_id", ""))].add(str(record.get("decider", "")))
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从真实对局测量各参与者战力")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument(
        "--manifest",
        default="",
        help="改用一份冻结的文件清单（每行一个路径，# 开头为注释）——双方比数必须用同一份",
    )
    parser.add_argument("--ours", default="u_a7f7c67bb14a", help="我们的 user_id，用于高亮")
    parser.add_argument("--min-hands", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（0 = 全部）")
    parser.add_argument("--by-decider", action="store_true", help="按会话所用决策器分组对比")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.events))
    if args.manifest:
        # 数据持续增长，只有冻结同一份清单，两边的数字才可比
        lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
        paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有匹配到事件流", file=sys.stderr)
        return 1

    deciders = room_deciders(Path(args.ledger)) if args.by_decider else {}
    per_user: dict[str, Aggregate] = defaultdict(Aggregate)
    by_group: dict[str, Aggregate] = defaultdict(Aggregate)
    mixed = 0
    games = 0
    rounds_all = 0
    draws_all = 0
    for path in paths:
        room = Path(path).parent.parent.name
        group: str | None = None
        if args.by_decider:
            used = deciders.get(room, set())
            if len(used) == 1:
                group = next(iter(used))
            else:
                # 该房间被多个决策器用过（会话中断后复用了同一房），归因不明故排除
                mixed += 1
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            seen, drawn = process(payload, per_user, args.ours, by_group, group)
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        games += 1
        rounds_all += seen
        draws_all += drawn

    rows = [(uid, agg) for uid, agg in per_user.items() if agg.hands >= args.min_hands]
    rows.sort(key=lambda kv: -kv[1].score_per_hand)
    if rounds_all:
        winner_rate = 1.0 - draws_all / rounds_all
        print(
            f"局数 {rounds_all}（流局 {draws_all} = {draws_all / rounds_all:.2%}）"
            f"  有胡率 {winner_rate:.2%}  公平份额 {winner_rate / 4:.2%}"
        )
        print("（局数与流局取自**事件流**的 round_ended.data.draw；顶层 rounds[] 对中间流局的"
              "收录是有损的，用它当分母会让有胡率虚高约 2 个百分点）")
    print(f"对局文件 {games} 个；参与人数 {len(per_user)}，其中手数 ≥{args.min_hands} 的 {len(rows)} 人")
    print("（'听牌率' = 每次出牌时处于听牌的近似比例；分值重尾，小样本的每手分不可靠）")
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
            if uid != args.ours:
                field.merge(agg)
        print()
        print(f"我们（{ours.hands} 手）vs 其余 {len(rows) - 1} 人（合计 {field.hands} 手）:")
        print("  " + ours.line("我们"))
        print("  " + field.line("其余"))

    if args.by_decider:
        print()
        print(f"按决策器分组（我们自己的战绩；混用房间已排除 {mixed} 个文件）:")
        if not by_group:
            print("  没有可归因的房间——检查 sessions.jsonl 是否记录了 decider")
        for name, agg in sorted(by_group.items(), key=lambda kv: -kv[1].score_per_hand):
            print("  " + agg.line(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
