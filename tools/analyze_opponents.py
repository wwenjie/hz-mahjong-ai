"""对手画像：从自由匹配对战记录里看**我们输在哪、别人赢在哪**（只读台账）。

**为什么做**：此前的分析全是「我们 vs 对手均值」，把三个对手压成一个数。
但自由匹配房里是**具体某个 bot**——知道「谁在赢、他怎么赢」才能定位我们的缺口。

三层口径：
1. **人级台账**（`sessions.jsonl` 的 `per_user`）：每个 uid 的房数/手数/胡次数/得分，
   算出胡率与每手得分，排出一张**天梯**（含我们自己的位置）。
2. **房级胜负归因**：每房里谁第 1、我们第几、我们与第 1 名的分差；
   并且区分「我们输给谁」——是输给同一个人反复出现，还是分散。
3. **同房对手强度**：我们所在房的对手平均胡率 vs 全局平均 ⇒ 判断我们的样本是不是「刚好撞上强敌」。

限制（引用时必须写明）：
- 台账只有**结算后的**每 uid 得分/胡次数/手数，**没有番与副露**（那要去事件流里算，
  见 `analyze_god_conversion.py` / `analyze_tenpai_timing.py`）。
- uid 是匿名账号，无法知道背后是同一个人的不同号还是不同人。
- 房间数有限（v2 40 + v3 30），且**谁是庄、座次**已被平台随机化（我们座位分布 0/1/2/3 都有）。
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

LEDGER = Path("data/auto_sessions/sessions.jsonl")
OUR = "u_a7f7c67bb14a"


def load() -> list[dict]:
    rows = []
    for line in LEDGER.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if (doc.get("runtime") or {}).get("status") != "finished":
            continue
        if doc.get("per_user"):
            rows.append(doc)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对手画像（自由匹配台账）")
    parser.add_argument("--min-rooms", type=int, default=8, help="至少出现在几个房才排进天梯")
    args = parser.parse_args(argv)

    rooms = load()
    print(f"含 per_user 的已结算房：{len(rooms)}")

    per_uid: dict[str, dict] = collections.defaultdict(
        lambda: {"rooms": 0, "hands": 0, "wins": 0, "score": 0}
    )
    # **我们自己要按档位分开**：台账的 per_user 只认 uid，而我们同一个 uid 跑过多个档位
    # （冠军档 + `first-legal` 对照臂）。混在一起会把对照臂的弱成绩算进「我们的水平」，
    # 从而既低估自己也污染「对手均强」的比较。第一次跑就踩了：我们的行是 19.64%，
    # 而实际上对照臂（first-legal）的胡率远低于此。
    our_by_decider: dict[str, dict] = collections.defaultdict(
        lambda: {"rooms": 0, "hands": 0, "wins": 0, "score": 0}
    )
    winners: collections.Counter = collections.Counter()
    our_rank: collections.Counter = collections.Counter()
    our_gap_to_first: list[int] = []
    our_opp_strength: list[float] = []
    all_strength: list[float] = []

    for doc in rooms:
        per = doc["per_user"]
        scores = {uid: int(v.get("score") or 0) for uid, v in per.items()}
        first = max(scores, key=lambda u: scores[u]) if scores else None
        if first:
            winners[first] += 1
        for uid, v in per.items():
            cell = per_uid[uid]
            cell["rooms"] += 1
            cell["hands"] += int(v.get("hands") or 0)
            cell["wins"] += int(v.get("wins") or 0)
            cell["score"] += int(v.get("score") or 0)
            if uid == OUR:
                ours = our_by_decider[str(doc.get("decider") or "?")]
                ours["rooms"] += 1
                ours["hands"] += int(v.get("hands") or 0)
                ours["wins"] += int(v.get("wins") or 0)
                ours["score"] += int(v.get("score") or 0)
        rates = [
            (int(v.get("wins") or 0) / int(v["hands"]))
            for v in per.values()
            if int(v.get("hands") or 0) > 0
        ]
        if rates:
            all_strength.append(statistics.mean(rates))
        mine = per.get(OUR)
        if mine and int(mine.get("hands") or 0):
            others = [u for u in per if u != OUR]
            opp = [int(per[u].get("wins") or 0) / int(per[u]["hands"]) for u in others
                   if int(per[u].get("hands") or 0)]
            if opp:
                our_opp_strength.append(statistics.mean(opp))
            if first:
                our_gap_to_first.append(int(mine.get("score") or 0) - scores[first])
        if doc.get("our_rank"):
            our_rank[int(doc["our_rank"])] += 1

    print("\n== 天梯（按每手得分 = 总得分/手数）==")
    ranked = []
    for uid, c in per_uid.items():
        if c["rooms"] < args.min_rooms or not c["hands"]:
            continue
        ranked.append((c["score"] / c["hands"], uid, c))
    ranked.sort(reverse=True)
    print(f"{'#':>3s} {'uid':>16s} {'房':>5s} {'手':>6s} {'胡率':>7s} {'每手得分':>9s} {'总得分':>9s}")
    our_pos = None
    for i, (per_hand, uid, c) in enumerate(ranked, 1):
        tag = "  ← 我们" if uid == OUR else ""
        if uid == OUR:
            our_pos = i
        print(f"{i:>3d} {uid:>16s} {c['rooms']:>5d} {c['hands']:>6d} "
              f"{c['wins'] / c['hands']:>7.2%} {per_hand:>+9.3f} {c['score']:>+9d}{tag}")
    print(f"\n我们在这张天梯里排第 {our_pos}/{len(ranked)} 名（阈值：至少 {args.min_rooms} 房）")

    print("\n== 我们自己按档位分开（台账 per_user 只认 uid，混算会拖低我们的行）==")
    print(f"{'档位':>14s} {'房':>5s} {'手':>7s} {'胡率':>7s} {'每手得分':>9s}")
    for dec, c in sorted(our_by_decider.items(), key=lambda kv: -(kv[1]["score"] / max(1, kv[1]["hands"]))):
        if not c["hands"]:
            continue
        print(f"{dec:>14s} {c['rooms']:>5d} {c['hands']:>7d} "
              f"{c['wins'] / c['hands']:>7.2%} {c['score'] / c['hands']:>+9.3f}")

    print("\n== 谁常拿第 1 ==")
    for uid, n in winners.most_common(6):
        tag = "  ← 我们" if uid == OUR else ""
        print(f"  {n:3d} 次  {uid}{tag}")

    if our_rank:
        total = sum(our_rank.values())
        print("\n== 我们的名次分布 ==")
        for place in sorted(our_rank):
            print(f"  第 {place} 名: {our_rank[place]:3d} 次 ({our_rank[place] / total:.1%})")
    if our_gap_to_first:
        print(f"\n== 我们与同房第 1 名的分差 ==\n  均值 {statistics.mean(our_gap_to_first):+.1f}"
              f"  中位 {statistics.median(our_gap_to_first):+.1f}"
              f"  区间 [{min(our_gap_to_first)}, {max(our_gap_to_first)}]")
    if our_opp_strength and all_strength:
        print(f"\n== 样本难度自检 ==\n  我们所在房的对手平均胡率 {statistics.mean(our_opp_strength):.2%}"
              f"；全体房平均 {statistics.mean(all_strength):.2%}"
              f"（差 {statistics.mean(our_opp_strength) - statistics.mean(all_strength):+.2%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
