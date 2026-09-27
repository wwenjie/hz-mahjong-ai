"""按玩家看副露率：我们是不是异类（tasks.md 5.5 的因果问题）。

**要回答的问题**：agent B 测出对手在**出牌层面**跨房跨强度高度同质
（跨房留出 ≈ 同房留出）。但副露是**另一类决策**，而整站在这里差了一倍
（我们 0.591/局 vs 对手 1.093/局）。

如果副露率在玩家之间**离散度很大**且与胜率**同向**，那么「多副露」是能力差异的
一个真实维度，我们低就是缺口。如果离散度很小（大家差不多）、只有我们掉在外面，
那说明**我们是异类**，问题在我们自己的某条规则（吃碰闸门），而不是「我们不够会吃碰」。

输出：
1. 所有玩家（≥N 手）的 副露/局 分布（分位数），以及我们的位置
2. 副露率与胜率的**相关系数**（含球员级散点信息）
3. 按副露率高低分组看胜率——只看相关性斜率，不做因果断言（见文件末尾的警告）

用法::

    uv run python tools/analyze_meld_by_player.py --events 'data/auto_sessions/*/events/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

from majiang.sim import replay

SEATS = 4


def collect(paths: list[str], ours: str) -> dict[str, list[int]]:
    """玩家 → [手数, 胡次数, 局末副露总数]。"""
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
        if len(ids) != SEATS:
            continue
        official = {
            int(entry.get("round_no", 0) or 0): entry
            for entry in (payload.get("rounds") or [])
        }
        try:
            for state, events in replay.iter_rounds(payload):
                for event in events:
                    replay.apply_event(state, event)
                entry = official.get(state.round_no) or {}
                winner = None if entry.get("is_draw") else entry.get("winner")
                for seat, uid in enumerate(ids):
                    bucket = per[uid]
                    bucket[0] += 1
                    bucket[2] += len(state.seats[seat].melds)
                    if isinstance(winner, int) and winner == seat:
                        bucket[1] += 1
        except Exception:  # noqa: BLE001
            continue
    return per


def pearson(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = stats.mean(xs), stats.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return cov / denom if denom else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按玩家看副露率")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--min-hands", type=int, default=200)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.events))
    if not paths:
        print("没有匹配到事件流", file=sys.stderr)
        return 1
    per = collect(paths, args.ours)

    rows = []
    for uid, (hands, wins, melds) in per.items():
        if hands < args.min_hands:
            continue
        rows.append((uid, hands, wins / hands, melds / hands))
    if not rows:
        print("没有满足手数门槛的玩家", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r[3])

    rates = [r[3] for r in rows]
    ours_row = next((r for r in rows if r[0] == args.ours), None)
    print(f"玩家 {len(rows)} 人（手数 ≥{args.min_hands}）")
    if ours_row:
        rank = [r[0] for r in rows].index(args.ours) + 1
        print(f"我们的副露/局 {ours_row[3]:.3f}，在 {len(rows)} 人中排第 {rank}（从低到高）")
    print(f"全体副露/局：最低 {rates[0]:.3f} / 25% {stats.quantiles(rates, n=4)[0]:.3f} / "
          f"中位 {stats.median(rates):.3f} / 75% {stats.quantiles(rates, n=4)[2]:.3f} / "
          f"最高 {rates[-1]:.3f}")
    print()

    pairs = [(r[3], r[2]) for r in rows]
    r_value = pearson(pairs)
    print(f"副露率 与 胜率 的相关系数 r = {r_value:+.3f}（n={len(pairs)}）")
    # 分组看斜率（只看斜率，不做因果断言）
    mid = stats.median(rates)
    low = [r[2] for r in rows if r[3] <= mid]
    high = [r[2] for r in rows if r[3] > mid]
    print(f"副露率低于中位的一组：胜率均值 {stats.mean(low):.2%}（{len(low)} 人）")
    print(f"副露率高于中位的一组：胜率均值 {stats.mean(high):.2%}（{len(high)} 人）")
    print()
    print("结论口径警告：**这是相关性，不是因果**。高胜率玩家可能因为「领先时更愿意副露」"
          "而副露多，也可能因为「副露多而赢」。判因果只能靠配对自对弈（`tools/ab_test.py`）"
          "——本工具的作用是判断「我们是不是异类」以及「这个维度有没有区分度」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
