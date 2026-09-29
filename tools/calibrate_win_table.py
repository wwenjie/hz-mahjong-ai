"""`P_win` 表的真机标定（B' 设计说明的门 2）。

**为什么需要它**：统一期望得分的第一项是 `P_win × E_pay`，而 `P_win` 查的是
`routes.WIN_RATE_MELD/PAIR` —— 那两张表是**自对弈标定**的（对手＝我们自己）。
B' 把它列为设计说明的**最大风险**：自对弈对手系统性偏弱于真机强 bot
（真机强 bot 胡率 29% vs 我们 20%），表可能整体偏高。
门 2 的判据是「**按房配对**核偏差 + **斜率 ∈ [0.8, 1.2]**」。

**口径**（每一步都用线上那条代码路径，不自己重推，避免口径漂移）：

- 取我们**实际打出**的那张牌之后的暗手向听 `s = shanten_any(hand − 打出牌, melds)`；
- 预测值直接调 `routes.win_probability(s, situation.table.draws_left)` —— `draws_left`
  取自 `state.situation_for(seat).table`，即决策器当时看到的那个数；
- 实际值 = 本局是否**我们**胡（`round_ended.data.scores` 的 argmax；`draw=true` 记 0）；
- 误差棒按**房**聚类（`notes/PROTOCOL.md` §7.2：一个房四家固定、约 80 手，
  按手抽样会把方差低估 6 倍）。

**这是一把仪器，不是一条臂**：它不动任何策略，只回答「表准不准」。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from statistics import mean, stdev

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.sim import replay
from majiang.strategy import routes

OUR = "u_a7f7c67bb14a"


def room_winner(doc: dict) -> dict[int, int]:
    """每局的赢家座位。

    **不能用顶层 `rounds[]`**：`notes/PROTOCOL.md` §7.1 记着它对**中途流局**的收录是有损的
    （三种行为并存），漏掉的局在我的统计里会变成「我们没胡」⇒ **正好制造一个向下的假偏差**，
    而那正是本仪器要判定的方向。所以只认事件流里必然存在的 `round_ended.data`：
    `scores` 是四家分差，赢家 = argmax（`draw=true` 记 −1）。
    """
    out: dict[int, int] = {}
    for entry in doc.get("rounds") or ():
        no = int(entry.get("round_no", 0) or 0)
        if entry.get("is_draw"):
            out[no] = -1
            continue
        winner = entry.get("winner")
        out[no] = int(winner) if winner is not None else -1
    return out


def winners_from_events(events) -> int:
    """从一局的事件里取赢家：`round_ended.data.scores` 的 argmax；流局记 −1。

    这是 `room_winner` 的**权威版本**（顶层 `rounds[]` 只用来交叉核对覆盖率）。
    """
    for event in events:
        if event.get("type") != "round_ended":
            continue
        data = event.get("data") or {}
        if data.get("draw"):
            return -1
        scores = [int(v) for v in (data.get("scores") or ())]
        if len(scores) != 4 or max(scores) <= 0:
            return -1
        return scores.index(max(scores))
    return -2


def main() -> int:
    parser = argparse.ArgumentParser(description="P_win 表的真机标定（门 2）")
    parser.add_argument("--rooms", type=int, default=30, help="按**房**抽样的房数")
    parser.add_argument("--limit", type=int, default=20000, help="最多看多少个决策点")
    parser.add_argument(
        "--correction",
        default="",
        help="按向听乘性修正，逗号分隔（例 `1.10,0.96,0.73,0.58,0.52,0.61`）。"
             "留空 = 用原始表。**重标后要复跑本仪器确认斜率回到 [0.8,1.2]**——"
             "这是让「重标」这个动作可被证伪的那一步，不能只改不打分。",
    )
    args = parser.parse_args()

    correction = tuple(float(x) for x in args.correction.split(",") if x.strip())
    rooms = sorted(p for p in glob.glob("data/auto_sessions/*/events") if glob.glob(f"{p}/*.json"))
    chosen = rooms[:: max(1, len(rooms) // args.rooms)][: args.rooms]
    files: list[str] = []
    for room in chosen:
        files.extend(sorted(glob.glob(f"{room}/*.json")))

    # per_room[room][shanten] = [(pred, actual), ...]
    per_room: dict[str, dict[int, list[tuple[float, float, int]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    seen = 0
    missing = mismatches = 0
    for path in files:
        if seen >= args.limit:
            break
        try:
            doc = json.loads(open(path, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        legacy = room_winner(doc)
        room_key = str(path).split("/")[-3]
        for state, events in replay.iter_rounds(doc):
            outcome = winners_from_events(events)
            if outcome == -2:
                continue  # 这一局没有 round_ended（事件流截断），整局跳过而不是记成没胡
            # 覆盖率交叉核对：顶层 rounds[] 与事件流对不上多少（PROTOCOL §7.1 的有损收录）
            if state.round_no not in legacy:
                missing += 1
            elif legacy[state.round_no] != outcome:
                mismatches += 1
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
                try:
                    dropped = tiles.parse(str(event.get("tile")))
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if hand[dropped] <= 0:
                    replay.apply_event(state, event)
                    continue
                after = list(hand)
                after[dropped] -= 1
                melds = len(state.seats[mine].melds)
                try:
                    s = shanten_module.shanten_any(after, melds)
                except shanten_module.ShantenError:
                    replay.apply_event(state, event)
                    continue
                situation = state.situation_for(mine)
                pred = routes.win_probability(s, situation.table.draws_left)
                if correction:
                    factor = correction[min(s, len(correction) - 1)]
                    pred = min(0.95, pred * factor)
                actual = 1 if outcome == mine else 0
                per_room[room_key][s].append((pred, actual, outcome))
                seen += 1
                replay.apply_event(state, event)

    if not seen:
        print("无样本")
        return 1

    buckets = sorted({s for room in per_room.values() for s in room})
    print(f"修正表 {correction or '（无，原始表）'}")
    print(f"决策点 {seen} 个 / 房 {len(per_room)} 个（按房抽样 {len(files)} 文件）")
    print(f"覆盖率交叉核对：顶层 `rounds[]` 缺该局 {missing} 次、与该局赢家不一致 {mismatches} 次"
          f"（PROTOCOL §7.1 记着它对中途流局收录有损 ⇒ 本仪器只认事件流的 `round_ended.data.scores`）\n")
    print(f"{'向听':>4} {'n':>7} {'预测均值':>9} {'实际均值':>9} {'差':>8} {'房数':>5} {'se':>7}")
    rows = []
    for s in buckets:
        preds: list[float] = []
        actuals: list[float] = []
        room_means: list[float] = []
        room_n = 0
        for room in per_room.values():
            items = room.get(s)
            if not items:
                continue
            room_n += 1
            preds.append(mean(p for p, _, _ in items))
            act = mean(a for _, a, _ in items)
            actuals.append(act)
            room_means.append(act)
        if not preds:
            continue
        pm, am = mean(preds), mean(actuals)
        se = stdev(room_means) / math.sqrt(len(room_means)) if len(room_means) > 1 else float("nan")
        rows.append((s, pm, am, room_n, se))
        print(
            f"{s:>4} {sum(len(per_room[r].get(s, [])) for r in per_room):>7} "
            f"{pm:>9.3f} {am:>9.3f} {am - pm:>+8.3f} {room_n:>5} {se:>7.3f}"
        )

    # 门 2：斜率（把「房-向听」当观测点做最小二乘）+ 偏置
    pts = [(pm, am) for _, pm, am, _, _ in rows if pm > 0]
    if len(pts) >= 3:
        mx, my = mean(p for p, _ in pts), mean(a for _, a in pts)
        var = sum((p - mx) ** 2 for p, _ in pts)
        slope = sum((p - mx) * (a - my) for p, a in pts) / var if var else float("nan")
        intercept = my - slope * mx
        print(f"\n=== 门 2 ===")
        print(f"  斜率（实际 ~ 预测）= **{slope:.3f}**（判据 ∈ [0.8, 1.2]）")
        print(f"  截距 {intercept:+.3f}")
        print(f"  加权总偏差：预测均值 {sum(p for p, _ in pts) / len(pts):.3f} vs "
              f"实际均值 {my:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
