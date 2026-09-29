"""真机「自由对战」分时代对比：heuristic / v2 / v3（**观测对比，不是交错 A/B**）。

**为什么需要它**：用户会问「这个修复上线后真机涨了多少」。答案是**真机结构上分辨不了**，
所以必须把这条讲清楚，而不是拿自对弈的数字当答案。

实测（2026-09-29 18:35，376 场会话 / 21259 手）：

=======  ======  ========  =========  ==========  ==========
时代      场数     手数      合计胜率    名次分      每手均分
=======  ======  ========  =========  ==========  ==========
heuristic 231    12418      20.85%     3.104      −63.34
v2        42      3200      21.34%     3.098      −76.50
v3        72      5641      22.07%     3.099      −71.49
=======  ======  ========  =========  ==========  ==========

**v3 − v2（按房聚类，含时代混杂）**：胜率 **+1.15 个百分点（se 1.75pp, t 0.66）**、
名次分 +0.001（t 0.01）、每手均分 +5.01（t 0.14）—— **三项全部不显著**，
95%CI 甚至包含 −2.3pp。而 v3 的自对弈 A/B 是名次分 +0.235(t1.9)/+0.446(t3.8)。
⇒ **真机既没确认也没否认**：这个 n 下它只能分辨 ±3.4 个百分点，
而要分辨 5 个百分点需要约 22 小时采集（`notes/PROTOCOL.md` §7.2）。

**这不是 A/B 而是观测对比**：v2 与 v3 是先后两个时代，房间、对手池、时段都不同，
所以只能给方向与置信区间，**不能排除时代混杂**。交错 A/B 只能靠自对弈。

数据：`data/auto_sessions/sessions.jsonl`——每场会话一行，带 `decider`、`our_win_rate`
（本场胜率 = our_wins / hands，hands 恒为 80）、`our_rank`、`our_score`。

**为什么必须按房聚类**：`notes/PROTOCOL.md` §7.2 的教训——一个房的四家固定、约 80 手，
按文件/按手抽样都会低估方差（真机噪声基底实测是二项预期的 6 倍）。所以误差棒取
「每房一个值 → 房间标准差 / √房数」。

**这条为什么不是 A/B**：v2 与 v3 是**先后两个时代**，房间、对手池、时段都不同
⇒ 只能给「有/没有改善」的方向与置信区间，**不能排除时代混杂**。交错 A/B 只能靠自对弈。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean, stdev

ERAS = ("heuristic", "v2", "v3")


def load() -> list[dict]:
    return [json.loads(line) for line in open("data/auto_sessions/sessions.jsonl", encoding="utf-8") if line.strip()]


def cluster(rows: list[dict], field: str) -> tuple[float, float, int]:
    """按房聚类：返回 (均值, 标准误, 房数)。"""
    per_room: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        per_room[str(row.get("room_id"))].append(float(value))
    values = [mean(v) for v in per_room.values()]
    if len(values) < 2:
        return (mean(values) if values else 0.0, 0.0, len(values))
    return mean(values), stdev(values) / math.sqrt(len(values)), len(values)


def main() -> int:
    rows = load()
    stats = {}
    for era in ERAS:
        picked = [r for r in rows if str(r.get("decider")) == era]
        if not picked:
            continue
        hands = sum(int(r.get("hands") or 0) for r in picked)
        wins = sum(int(r.get("our_wins") or 0) for r in picked)
        stats[era] = picked
        print(f"=== {era}：{len(picked)} 场 / {hands} 手 / 合计胜率 {wins / hands:.2%}")
        for field in ("our_win_rate", "our_rank", "our_score", "our_average_fan"):
            m, se, rooms = cluster(picked, field)
            print(f"    {field:16s} 均值 {m:8.4f}  标准误 {se:6.4f}（按房聚类，房数 {rooms}）")
        span = (picked[0].get("started_at"), picked[-1].get("started_at"))
        print(f"    时段 {span[0]} → {span[1]}")

    if "v2" in stats and "v3" in stats:
        print("\n=== v3 − v2（观测差，含时代混杂，不是因果）")
        for field in ("our_win_rate", "our_rank", "our_score"):
            m2, se2, n2 = cluster(stats["v2"], field)
            m3, se3, n3 = cluster(stats["v3"], field)
            diff = m3 - m2
            se = math.sqrt(se2**2 + se3**2)
            t = diff / se if se else 0.0
            mde = 1.96 * se
            verdict = "显著" if abs(t) > 1.96 else "**不显著**"
            print(
                f"    {field:16s} {diff:+8.4f}  标准误 {se:6.4f}  t {t:+5.2f}  "
                f"95%CI [{diff - mde:+.4f}, {diff + mde:+.4f}]  {verdict}"
            )
        print(
            "\n读法：真机噪声基底 1.50 个百分点（胜率差 sd），是二项预期的 6 倍 ⇒ "
            "胜率的**每场**分辨力很弱；能分辨 5 个百分点需要 ~22 小时采集。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
