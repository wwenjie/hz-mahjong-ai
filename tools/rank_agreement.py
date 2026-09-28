"""排序对拍：我们的 ``total`` 定价 vs 真值表的实测定价（agent B 的真值表 v2）。

**它回答的问题**：`policy._score_discard` 的默认分支是一个**手调线性组合**
``total = -10×向听 + 骨架厚度 - 3×喂牌 - 财神罚``。这个组合的系数从来只是直觉，
没人验证过它的**隐含定价**是否与实测价格一致。如果一致，说明缺的只是维度
（喂牌/财神这种它没建模的量）；如果不一致，说明组合方式本身就错，那就该换成
**统一期望得分**（直接用真值表给每个候选定价）——这是 agent B 与我要判的分叉点。

**三个排序**（同一批位置，同一批候选牌）：

- ``A``：我们的 ``total``（原值，含喂牌与财神项）。
- ``A2``：我们的 ``total`` 去掉喂牌与财神项 = ``-10×向听 + 骨架厚度``。
  加这个是为了**定位偏离来源**：若 A2 对拍明显好于 A，说明是喂牌项的定价错了，
  而不是「组合方式」错了——两种情况的对策完全不同（补维度 vs 换框架）。
- ``B``：真值表期望分 = ``p_win × fan_avg``（乘番），另有 ``B_win`` = ``p_win``（不乘番）。
  两个都算：我在 THREAD 里请 B 裁「要不要乘番」，让数据先说话。

**口径与已知局限（不写清楚就会被误读）**：

1. 表的 ``p_win`` 是在**行为策略**（我们的历史对局）下统计的，所以它给候选牌定价时
   带一点自洽性：我们常去的状态，其价格被我们后续的打法影响过。桶足够粗，这个偏差可接受。
2. 表没有**喂牌维度**，所以 ``B`` 里没有与 ``-3×feed`` 对应的项。这不是失误，
   是表的盲区；A2 的存在就是为了把这一项单独隔离出来看。
3. 派的 ``us`` 列定价的是「我们自己」，所以只对 ``OUR_UID`` 那个座位做对拍。
4. 位置集刻意与 B 的表对齐：**抓打圈强制的出牌不算自由决策点**，跳过。

用法::

    uv run python tools/rank_agreement.py --limit 300
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from majiang.rules.action import DISCARD, Action  # noqa: E402
from majiang.rules.tiles import GOD, TILE_KINDS  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import HeuristicDecider  # noqa: E402

MANIFEST = Path("notes/manifest-20260926.txt")
TABLE = Path("verify/out/win_rate_table.json")
OUR_UID = "u_a7f7c67bb14a"
# 桶样本太小的格子不用：真值表本身给了 n，小于这个数就没有定价能力
MIN_CELL_N = 100


def load_table(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise SystemExit(f"缺 {path}——先跑 verify/win_rate_table.py（agent B 的产物）")
    return json.loads(path.read_text(encoding="utf-8"))


def price(
    table: dict[str, dict[str, float]], shb: int, godb: int, mb: int, ph: int, who: str = "us"
) -> tuple[float, float] | None:
    """返回 ``(期望分, 胜率)``；格子样本不足则 None（该位置整条作废，不猜）。"""
    cell = table.get(f"{who}|{shb}|{godb}|{mb}|{ph}")
    if not cell or int(cell.get("n", 0)) < MIN_CELL_N:
        return None
    return float(cell["p_win"]) * float(cell["fan_avg"]), float(cell["p_win"])


def argmax_set(values: dict[int, float]) -> set[int]:
    top = max(values.values())
    return {tile for tile, value in values.items() if value == top}


def positions(files: list[str], our_uid: str):
    """按「我们自己的自由出牌点」产出 ``(situation, 该座摸牌序号, 座位)``。"""
    for path in files:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        uids = [seat.get("user_id") for seat in doc.get("seats") or []]
        if our_uid not in uids:
            continue
        our = uids.index(our_uid)
        for state, events in replay.iter_rounds(doc):
            # **摸牌序号要自己数**：`state.draws` 统计的是全场（含别家），而真值表的阶段
            # 维度是「该座第几次摸牌」。且庄家首回合的第 14 张在 `start_hands` 里、
            # 没有 `tile_drawn` 事件，所以它的摸牌序号是 0——与 B 的表口径一致。
            draw_no = [0, 0, 0, 0]
            for event in events:
                etype = event.get("type")
                seat = event.get("seat")
                if etype == replay.DRAWN and isinstance(seat, int) and 0 <= seat < 4:
                    draw_no[seat] += 1
                data = event.get("data") or {}
                if (
                    etype == replay.DISCARDED
                    and seat == our
                    and state.opened
                    and not data.get("catch_play")  # 抓打圈强制出牌不是自由决策点
                ):
                    yield state, draw_no[our], our
                replay.apply_event(state, event)


def phase_of(draw_no: int) -> int:
    """与 ``verify/win_rate_table.py`` 的 ``bucket()`` 完全一致，不得各写一套。"""
    return 0 if draw_no <= 4 else (1 if draw_no <= 8 else 2)


def evaluate(
    decider: HeuristicDecider,
    table: dict[str, dict[str, float]],
    state: replay.ReplayState,
    draw_no: int,
    seat: int,
) -> dict | None:
    counts = state.seats[seat].hand
    if sum(counts) % 3 != 2:  # 非决策张数，跳过（重建异常或非本人回合）
        return None
    situation = state.situation_for(seat)
    hand = situation.hand
    god_now = hand.god_count
    meld_bucket = min(len(state.seats[seat].melds), 2)
    phase = phase_of(draw_no)

    scores: dict[int, object] = {}
    for tile in range(TILE_KINDS):
        if counts[tile] <= 0:
            continue
        scores[tile] = decider._score_discard(situation, Action(DISCARD, tile=tile))
    if len(scores) < 2:
        return None

    total_a: dict[int, float] = {}
    total_a2: dict[int, float] = {}
    price_b: dict[int, float] = {}
    win_b: dict[int, float] = {}
    # **「候选跨几个桶」是这张表的硬前提**：副露数与阶段对同一副手牌的所有候选都相同，
    # 所以候选之间只有「向听」和「财神数」可能不同。同桶 = 表给同一个价 = 对排序零分辨力。
    # 我们真正要审计的「同向听候选之间的选择」恰好全落在同桶里——这一步就是为了把它量出来。
    buckets: set[tuple[int, int, int, int]] = set()
    for tile, score in scores.items():
        total_a[tile] = score.total
        # 去掉喂牌与财神项：只剩「向听 + 骨架厚度」
        total_a2[tile] = -decider.config.shanten_weight * score.shanten + score.blocks
        god_after = god_now - (1 if tile == GOD else 0)
        key = (min(score.shanten, 4), min(god_after, 2), meld_bucket, phase)
        buckets.add(key)
        quoted = price(table, *key)
        if quoted is None:
            return None  # 有任何候选落进样本不足的格子，就整条作废，避免半张定价
        price_b[tile] = quoted[0]
        win_b[tile] = quoted[1]

    set_a, set_a2 = argmax_set(total_a), argmax_set(total_a2)
    set_b, set_bw = argmax_set(price_b), argmax_set(win_b)
    pick = min(set_a)
    pick2 = min(set_a2)
    return {
        "shanten": scores[pick].shanten,
        "candidates": len(scores),
        "buckets": len(buckets),
        "b_ties": len(set_b),
        "a_top1": bool(set_a & set_b),
        "a_overlap": len(set_a & set_b) / max(len(set_a), len(set_b)),
        "a2_top1": bool(set_a2 & set_b),
        "a2_overlap": len(set_a2 & set_b) / max(len(set_a2), len(set_b)),
        "bw_top1": bool(set_a & set_bw),
        "a_ties": len(set_a),
        # **分位是最耐平局的指标**：A 的选择在表的定价里排在第几（0=最好，0.5=随机）。
        "a_pct": percentile(price_b, pick),
        "a2_pct": percentile(price_b, pick2),
        # A 的选择在表的定价下，比 B 的最优差多少（≤0 表示 A 选的更差）
        "gap": price_b[min(set_b)] - price_b[pick],
        "gap_win": win_b[min(set_bw)] - win_b[pick],
        # A2（去掉喂牌/财神项）同样的两个量：**这是「喂牌项到底伤不伤」在这张表上
        # 唯一可用的测法**——比较两者与表的最优之间的 regret。
        "same_pick": pick == pick2,
        "a2_gap": price_b[min(set_b)] - price_b[pick2],
        "a2_gap_win": win_b[min(set_bw)] - win_b[pick2],
    }


def percentile(prices: dict[int, float], pick: int) -> float:
    """``pick`` 在 ``prices`` 里的分位：严格更好的算 1、与它同价的算 0.5。"""
    mine = prices[pick]
    better = sum(1 for value in prices.values() if value > mine)
    equal = sum(1 for value in prices.values() if value == mine) - 1
    return (better + 0.5 * equal) / max(1, len(prices) - 1)


def summarize(rows: list[dict], label: str) -> None:
    n = len(rows)
    if not n:
        print(f"  {label}: 无样本")
        return
    print(
        f"  {label:12s} n={n:6d}  前1一致 {sum(r['a_top1'] for r in rows) / n:6.1%}"
        f"  A2前1 {sum(r['a2_top1'] for r in rows) / n:6.1%}"
        f"  平均交集 {sum(r['a_overlap'] for r in rows) / n:6.1%}"
        f"  A2交集 {sum(r['a2_overlap'] for r in rows) / n:6.1%}"
        f"  期望分差 {sum(r['gap'] for r in rows) / n:+7.4f}"
        f"  胜率差 {sum(r['gap_win'] for r in rows) / n:+7.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="total 定价 vs 真值表定价 的排序对拍")
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--table", default=str(TABLE))
    parser.add_argument("--limit", type=int, default=0, help="最多处理几个事件流（0=全部）")
    parser.add_argument("--stride", type=int, default=1, help="抽样步长：1=全量，4=每 4 个取 1")
    parser.add_argument("--json", default="", help="把逐位置明细写到这个路径（可选）")
    args = parser.parse_args(argv)

    files = [
        line.strip()
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ][:: max(1, args.stride)]
    if args.limit:
        files = files[: args.limit]
    table = load_table(Path(args.table))
    decider = HeuristicDecider()

    rows: list[dict] = []
    skipped = 0
    for index, (state, draw_no, seat) in enumerate(positions(files, OUR_UID)):
        row = evaluate(decider, table, state, draw_no, seat)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if (index + 1) % 2000 == 0:
            print(f"  ... 已处理 {index + 1} 个出牌点（{len(rows)} 条有效）", file=sys.stderr)

    print(f"事件流 {len(files)} 份；出牌点 {len(rows) + skipped} 个，有效 {len(rows)} 个"
          f"（作废 {skipped}，含格子样本不足）")
    if not rows:
        return 1
    discrim = [r for r in rows if r["buckets"] >= 2]
    flat = [r for r in rows if r["buckets"] == 1]
    n = len(rows)
    print(f"\n候选牌数均值 {sum(r['candidates'] for r in rows) / n:.2f}"
          f"  A 侧并列均 {sum(r['a_ties'] for r in rows) / n:.2f} 张"
          f"  B 侧并列均 {sum(r['b_ties'] for r in rows) / n:.2f} 张")
    print(f"**跨桶位置 {len(discrim)} 个（{len(discrim) / n:.1%}）**；"
          f"**同桶位置 {len(flat)} 个（{len(flat) / n:.1%}）——表在这类位置上给所有候选"
          f"同一个价，对排序零分辨力**。")
    print("  注意：跨桶位置的共识几乎只是「都偏好低向听」，是平凡的。"
          "真正要审计的同向听选择全在同桶里。")

    print("\n== 全部位置（前1一致率被平局抬高，只看分位）==")
    summarize(rows, "整体")
    print(f"  （另：A 的前1 与 B_win（不乘番）一致率 "
          f"{sum(r['bw_top1'] for r in rows) / n:.1%} —— 番项该不该进的裁决）")
    if flat:
        print(f"  同桶子集：A 分位 {sum(r['a_pct'] for r in flat) / len(flat):.1%}"
              f"  A2 分位 {sum(r['a2_pct'] for r in flat) / len(flat):.1%}"
              f"（0%=表的定价顶点，50%=随机）")

    print("\n== 跨桶位置（表确实有分辨力的部分）==")
    summarize(discrim, "跨桶")

    print("\n== 按向听分层（B 的交叉点在 2，所以必须分层看）==")
    by_shanten: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_shanten[min(row["shanten"], 4)].append(row)
    for shanten in sorted(by_shanten):
        summarize(by_shanten[shanten], f"向听 {shanten}")

    if discrim:
        print("\n== 喂牌项到底伤不伤（在这张表上唯一可用的测法：比较 regret）==")
        same = [r for r in discrim if r["same_pick"]]
        diff = [r for r in discrim if not r["same_pick"]]
        print(f"  A 与 A2 选出同一张的比例 {len(same) / len(discrim):.1%}"
              f"；不同的 {len(diff)} 个")
        print(f"  A  的 regret：期望分 {sum(r['gap'] for r in discrim) / len(discrim):+.5f}"
              f"  胜率 {sum(r['gap_win'] for r in discrim) / len(discrim):+.5f}")
        print(f"  A2 的 regret：期望分 {sum(r['a2_gap'] for r in discrim) / len(discrim):+.5f}"
              f"  胜率 {sum(r['a2_gap_win'] for r in discrim) / len(discrim):+.5f}")
        if diff:
            print(f"  只在两者分歧的 {len(diff)} 个位置上："
                  f"A regret {sum(r['gap_win'] for r in diff) / len(diff):+.5f} vs "
                  f"A2 regret {sum(r['a2_gap_win'] for r in diff) / len(diff):+.5f}")
        print("  读法：regret ≈ 0 且 A/A2 无差别 ⇒ 这张表**没有能力**定价喂牌项"
              "（它根本没有喂牌维度），故此处得不出「喂牌项对/错」的结论——"
              "只能得「表的粒度不足以审计这一项」。")

    print("\n== 判据（预登记口径，**已按上面的发现改写一次，写在这里备查**）==")
    print("  原判据（前1一致率 60%/80%）作废：B 侧并列均 "
          f"{sum(r['b_ties'] for r in rows) / n:.2f} 张，交集非空几乎必然，该指标虚高。")
    print("  改用**分位**（0=最好，50%=随机），只在跨桶子集上判：")
    if not discrim:
        print("    跨桶位置为 0 → 表无法给出任何排序信息。")
        return 1
    top1 = sum(r["a_top1"] for r in discrim) / len(discrim)
    a_pct = sum(r["a_pct"] for r in discrim) / len(discrim)
    a2_pct = sum(r["a2_pct"] for r in discrim) / len(discrim)
    print(f"    跨桶子集 n={len(discrim)}：A 分位 {a_pct:.1%}  A2 分位 {a2_pct:.1%}"
          f"  前1一致 {top1:.1%}")
    if a2_pct <= 0.25:
        print("    → 去喂牌/财神后与表的定价基本同向（≤25%）⇒ **组合方向没错**，"
              "缺的是维度；A 与 A2 的差就是喂牌项的影响，而表看不见喂牌、无从裁决它。")
    elif a2_pct > 0.40:
        print("    → A2 分位 > 40% 已接近随机 ⇒ **组合方式本身偏离真值**，"
              "该换统一期望得分。")
    else:
        print("    → 25–40% 灰区，人工看 20 个不一致的例子再定。")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n明细已写 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
