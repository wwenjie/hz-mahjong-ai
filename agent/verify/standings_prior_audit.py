#!/usr/bin/env python
"""C10 局况/名次 **先验审计**（只读真机事件流，零平台请求）。

**要解决的冲突（本仓两处先验互相矛盾）**：
- `src/majiang/rules/table.py:37`：「比赛按**首名率**结算（我们 6.2% vs 榜首 45.0%）」；
- `9386cce`（A 01:22 接受 B' 撤回）：「官方排序键是**总得分** ⇒ 逐手最大化期望已最优，
  首名率低是**得分能力**问题，不是目标错位」。

⇒ 协调者 10:55 的 P(首名) 设计如果是「换目标」，就必须先回答：**在「总得分为第一排序键」
的前提下，优化 P(首名) 与优化 E[总得分] 会不会给出不同策略？** 本仪器用真机数据量化这一点。

**数据来源**：每场事件流文件的 `rounds` 数组（8 局 × 四家净分，逐局累加）——离线可算，
不需要动平台。`rounds[k].scores[seat]` 是第 k+1 局该座位净得分。

**口径**：
- `本局开始时的累计分` = 前 k 局净分之和（第 1 局 = 全 0）；
- `名次` = 累计分降序排位（1..4，并列取最小名次）；
- `余局` = 8 − k；
- 终局首名 = 该场打完（8 局）后累计分最高者。

**四问**：
① 我们「本局开始时」的名次分布（我们大多落在哪一档）；
② `P(终局首名 | 名次 × 余局)`（复现，核对 30× 跨度）；
③ `P(终局首名 | 与榜首分差 × 余局)`——**非线性证据**（尾部是否对分差敏感）；
④ **天花板**：落在「余局 ≤3」这种尾部决策点的占比；以及我们的**单局均分 vs 场**——
    用来判「低首名率 = 能力不足（E-max 已最优）」还是「低首名率 = 姿态错（需要 P-first）」。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import statistics

OUR = "u_a7f7c67bb14a"


def rank_of(scores: list[int], i: int) -> int:
    """累计分降序名次（1=最高）；并列取最小名次。"""
    s = scores[i]
    return 1 + sum(1 for j in range(len(scores)) if j != i and scores[j] > s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(__file__).resolve().parents[2] / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.rooms:
        files = files[:: max(1, len(files) // args.rooms)][: args.rooms]

    rank_before = collections.Counter()
    final_rank = collections.Counter()
    # ② P(终局首名 | 名次, 余局)
    tbl: dict[tuple[int, int], list[int]] = collections.defaultdict(lambda: [0, 0])
    # ③ P(终局首名 | 分差档, 余局)
    mb: dict[tuple[str, int], list[int]] = collections.defaultdict(lambda: [0, 0])
    tail_pts = tail_first = 0
    person_rounds = 0
    n_rooms = 0
    # ④ 我方与场：单局均分
    our_round_sum = 0.0
    opp_round_sum = 0.0
    our_round_n = 0
    opp_round_n = 0

    def margin_band(d: int) -> str:
        if d >= 100:
            return "领先≥100"
        if d >= 30:
            return "领先30-99"
        if d >= 1:
            return "领先1-29"
        if d == 0:
            return "并列"
        if d >= -29:
            return "落后1-29"
        if d >= -99:
            return "落后30-99"
        return "落后≥100"

    for path in files:
        try:
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rounds = doc.get("rounds") or []
        seats = doc.get("seats") or []
        ids = [str(s.get("user_id", "")) for s in seats]
        if len(ids) != 4 or len(rounds) < 4 or OUR not in ids:
            continue
        me = ids.index(OUR)
        n_rooms += 1
        total_rounds = len(rounds)
        # 逐局累计
        cum = [0, 0, 0, 0]
        cum_hist: list[list[int]] = []
        for rd in rounds:
            sc = [int(x) for x in (rd.get("scores") or [0, 0, 0, 0])]
            cum = [a + b for a, b in zip(cum, sc)]
            cum_hist.append(list(cum))
        final_scores = cum_hist[-1]
        fin_r = rank_of(final_scores, me)
        final_rank[fin_r] += 1

        for k in range(total_rounds):
            before = [0, 0, 0, 0] if k == 0 else cum_hist[k - 1]
            rounds_left = total_rounds - k
            r = rank_of(before, me)
            person_rounds += 1
            rank_before[r] += 1
            is_first = 1 if fin_r == 1 else 0
            tbl[(r, rounds_left)][0] += 1
            tbl[(r, rounds_left)][1] += is_first
            margin = before[me] - max(before[j] for j in range(4) if j != me)
            mb[(margin_band(margin), rounds_left)][0] += 1
            mb[(margin_band(margin), rounds_left)][1] += is_first
            if rounds_left <= 3:
                tail_pts += 1
                tail_first += is_first
            # 单局均分：k>=1 才有该局得分（k 局是「第 k+1 局开始前」）
            if k >= 1:
                sc = [int(x) for x in (rounds[k - 1].get("scores") or [0, 0, 0, 0])]
                our_round_sum += sc[me]
                our_round_n += 1
                for j in range(4):
                    if j != me:
                        opp_round_sum += sc[j]
                        opp_round_n += 1

    print(f"扫描：{n_rooms} 场 × {len(files)} 文件（我方在场）")
    print(f"人-局决策点 = {person_rounds}\n")

    print("① 我们**本局开始时**的名次分布（决策器此前完全看不到这个量）")
    for r in range(1, 5):
        n = rank_before.get(r, 0)
        print(f"   {r} 名 {n:7d}  {n / person_rounds:6.1%} {'#' * int(n / person_rounds * 50)}")

    print("\n② 终局名次分布")
    tot = sum(final_rank.values())
    for r in range(1, 5):
        n = final_rank.get(r, 0)
        print(f"   {r} 名 {n:7d}  {n / tot:6.1%}")

    print("\n③ P(终局首名 | 名次 × 余局)  —— 复现核对")
    print(f"   {'名次':>4} {'余局':>4} {'样本':>6} {'P(首名)':>8}")
    for (r, rl), (n, f) in sorted(tbl.items()):
        if n >= 20:
            print(f"   {r:>4} {rl:>4} {n:>6} {f / n:>8.1%}")

    print("\n④ P(终局首名 | 与榜首分差档 × 余局)  —— 非线性证据")
    print(f"   {'分差档':>10} {'余局':>4} {'样本':>6} {'P(首名)':>8}")
    for (band, rl), (n, f) in sorted(mb.items()):
        if n >= 20:
            print(f"   {band:>10} {rl:>4} {n:>6} {f / n:>8.1%}")

    print("\n⑤ 天花板与归因")
    print(f"   尾部决策点（余局 ≤3）= {tail_pts} / {person_rounds} = {tail_pts / person_rounds:.1%}")
    print(f"   其中拿到终局首名 = {tail_first / tail_pts:.1%}" if tail_pts else "   n/a")
    print(f"   我方终局首名率 = {final_rank.get(1, 0) / tot:.1%}" if tot else "")
    if our_round_n and opp_round_n:
        print(f"   我方单局均分 = {our_round_sum / our_round_n:+.3f} / 对手 = {opp_round_sum / opp_round_n:+.3f}")
        print(f"   （若显著低于 0 ⇒ 每局都在输分 ⇒ 低首名率是**得分能力**问题，与 E-max 最优一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
