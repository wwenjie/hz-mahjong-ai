"""局况分布诊断：**决策器现在看不到的那部分信息，到底有多少变化量**（B' Top1 的前置测量）。

**为什么先量再做**：`research/qualitative-leap-survey.md` 的 Top1 候选是「目标函数错位」——
比赛按首名率结算，而决策器优化素点期望、**且看不到比分**。
但「加一个局况项」值不值得投入，取决于**局况在真机上到底有多大变化量**：
若我们几乎总在同一档（例如永远中游），局况项的价值上限就很低；
若经常处于「大分差落后/领先且局数无多」，那它就是一个从未被用过的、真实存在的输入。

**数据来源**：事件流的每局 `round_ended.data.scores` 是四家分差 ⇒ 逐局累加即得到
「本局开始时四家的累计比分」——也就是决策器**本该看到却看不到**的那个局面。
（**不需要动平台**，全部离线可算。这也说明：接线之前就能先量价值。）

产物口径：
- `本局开始时的名次`：按累计比分在我们这一局**开始前**的排序；
- `与榜首差 / 与第 3 名差`：分差，用于判断「追/守」的方向；
- `剩余局数`：一场 8 局。

用法::

    uv run python tools/analyze_match_standing.py --rooms 60
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

from majiang.sim import replay

OUR = "u_a7f7c67bb14a"


def main() -> int:
    parser = argparse.ArgumentParser(description="局况（本局开始时的名次与分差）分布诊断")
    parser.add_argument("--rooms", type=int, default=60, help="按房抽样")
    args = parser.parse_args()

    rooms = sorted(p for p in glob.glob("data/auto_sessions/*/events") if glob.glob(f"{p}/*.json"))
    chosen = rooms[:: max(1, len(rooms) // args.rooms)][: args.rooms]
    files: list[str] = []
    for room in chosen:
        files.extend(sorted(glob.glob(f"{room}/*.json")))

    rank_before = collections.Counter()
    margin_before = collections.Counter()
    rounds_left_hist = collections.Counter()
    final_rank = collections.Counter()
    rounds_seen = 0
    # 每一局的 (该局开始前的名次, 该局开始前还剩几局, 该场终局名次, 该场总局数)
    # 这是局况项唯一需要的表：`P(终局首名 | 名次, 剩余局数)`
    transitions: list[tuple[int, int, int, int]] = []
    for path in files:
        try:
            doc = json.loads(open(path, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        cumulative = [0, 0, 0, 0]
        positions_in_order: list[int] = []
        for state, events in replay.iter_rounds(doc):
            end = next(
                (e.get("data") or {} for e in events if e.get("type") == "round_ended"), None
            )
            if end is None:
                continue
            # 本局**开始前**的局况（决策器在这个局面下的全部决策都看不到它）
            order = sorted(range(4), key=lambda s: -cumulative[s])
            position = order.index(mine) + 1
            leader = cumulative[order[0]]
            third = cumulative[order[2]]
            positions_in_order.append(position)
            rank_before[position] += 1
            if position == 1:
                margin_before["领先"] += 1
            elif leader - cumulative[mine] <= 20:
                margin_before["落后≤20"] += 1
            elif leader - cumulative[mine] <= 60:
                margin_before["落后21~60"] += 1
            else:
                margin_before["落后>60"] += 1
            rounds_seen += 1
            scores = end.get("scores") or []
            if len(scores) == 4:
                for index in range(4):
                    cumulative[index] += int(scores[index])
        total_rounds = len(positions_in_order)
        if not total_rounds:
            continue
        final_order = sorted(range(4), key=lambda s: -cumulative[s])
        final = final_order.index(mine) + 1
        final_rank[final] += 1
        rounds_left_hist[total_rounds] += 1
        for index, position in enumerate(positions_in_order):
            if index == 0:
                continue  # 第 1 局前的累计分全是 0 ⇒ 名次由座位序号决定，不是真局况
            transitions.append((position, total_rounds - index, final, total_rounds))
        _ = third

    if not rounds_seen:
        print("无样本")
        return 1

    print(f"局数 {rounds_seen}（按房抽 {len(chosen)} 房 / {len(files)} 文件）\n")
    print("① **本局开始时**我们的名次分布（决策器现在看不到这个量）")
    for position in (1, 2, 3, 4):
        share = rank_before[position] / rounds_seen
        bar = "█" * int(share * 40)
        print(f"   {position} 名 {rank_before[position]:6d}  {share:6.1%} {bar}")

    print("\n② 本局开始时的分差档位")
    for key in ("领先", "落后≤20", "落后21~60", "落后>60"):
        if margin_before[key]:
            print(f"   {key:8s} {margin_before[key]:6d}  {margin_before[key] / rounds_seen:6.1%}")

    print("\n③ 一场打完的**终局**名次")
    done = sum(final_rank.values())
    for position in (1, 2, 3, 4):
        print(f"   {position} 名 {final_rank[position]:6d}  {final_rank[position] / done:6.1%}")

    print("\n④ P(终局首名 | **本局开始时**的名次 × 剩余局数) —— 局况项需要的正是这张表")
    print("   （口径：对每一局，按它**开始前**的名次与剩余局数分桶，看该场最终有没有拿首名。")
    print("    **第 1 局除外**：那时累计分全是 0，名次其实由座位序号决定，不是真局况。）")
    print(f"   {'名次':>4} {'剩余局数':>8} {'样本':>6} {'P(终局首名)':>12}  {'P(终局垫底)':>12}")
    for position in (1, 2, 3, 4):
        for band, (lo, hi) in (("≤3", (0, 3)), ("4~6", (4, 6)), ("≥7", (7, 99))):
            picked = [
                (right, total)
                for (pos, left, right, total) in transitions
                if pos == position and lo <= left <= hi
            ]
            if not picked:
                continue
            n = len(picked)
            p_first = sum(1 for right, _ in picked if right == 1) / n
            p_last = sum(1 for right, _ in picked if right == 4) / n
            print(
                f"   {position:>4} {band:>8} {n:>6} {p_first:>12.1%}  {p_last:>12.1%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
