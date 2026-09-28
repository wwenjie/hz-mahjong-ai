"""中段进度诊断：按**出牌序号**分层比较「到听率」（只读事件流）。

**要回答什么**：我们中段落后吗？落后多少？在哪一段开始落后？

**为什么不能按局统计**（我第一版就错在这）：按局算「本局至少一次到听」得到
我们 **69%** vs 对手 **45.5%**——看起来我们领先。但那是**机会不等**造成的假象：
被早胡掉的短局四家都摸得少，而按局的分母把它们一并算上，于是「谁被早胡掉」
变成了主因。**按出牌序号分层才是同机会比较**，结果完全反过来（见下）。

**实测（2026-09-28，120 文件 / 960 局 / 本家 7661 个出牌点、对手 25927 个）**：

| 出牌序号 | 我们 到听率 | 对手 到听率 | 差 |
|---|---|---|---|
| 1–4 | 0.9 → 12.5% | 1.0 → 13.1% | ≈0 |
| 5 | 18.6% | 22.1% | −3.5 |
| 7 | 32.4% | 39.7% | −7.3 |
| 9 | 47.5% | 54.9% | −7.4 |
| 10 | 52.5% | 62.3% | −9.8 |

⇒ **前四张打平、第 5 张起持续落后 3.5–9.8pp**；我们**又慢又窄**（不是「快而窄」）。
这与 agent B 的「向听 1–2 被反超、3–4 更快」的交叉点同源，但这是直接测到听率、不经过向听口径。

**⚠ 必须与结论一起引用的混淆**：出牌序号**不是**完美的机会度量——
**副露多的一方不出牌也在推进**，而对手副露是我们的 1.85 倍（1.09 vs 0.59/局）。
所以「同等序号落后」里有一部分是「他们靠副露多走了一步」。工具因此**按当前副露数分层**，
而分层结果把这条缺口定位得很具体（2026-09-28，12 房）：

| 层 | 序号 5 | 7 | 9 | 10 | 12+ |
|---|---|---|---|---|---|
| **0 副露** | 打平 | 打平 | — | 我们 **+12.0** | **+29.1** |
| **有副露** | −6.4 | −9.4 | **−14.2** | **−16.2** | −11.7 |

⇒ **「我们落后」几乎全部由有副露层贡献**：0 副露时我们反而领先，
一旦有副露就落后 7–16pp。所以缺口不是「副露太少」（`meld-equal` 放宽闸门已被 A/B 一致否掉，
且它不抬听口宽度），而是**我们的副露手推进得更差**。
**这是当前最具体的一条线索**，但尚未归因到具体决策（是「吃碰哪种」还是「吃碰后怎么打」）。

**用法**：

    uv run python tools/analyze_tenpai_timing.py --limit 150
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.sim import replay

SEATS = 4
OUR_UID = "u_a7f7c67bb14a"
US = "我们"
THEM = "对手"
MAX_ORDINAL = 12  # 第 12 及以上合成一格（样本骤减）


def rooms_first(paths: list[str], rooms: int) -> list[str]:
    """**按房抽样**，不按文件抽样。

    手册里写明过的坑：一个房四家固定、每房 10 个文件，按文件取前 N 个会得到
    「少数几个房的全部文件」——我 2026-09-28 又踩了一次：`--limit 40` 与 `--limit 120`
    在同一份排序上分别落到不同房，两跑给出**相反结论**。所以这里改成先按房分组，
    再取前 ``rooms`` 个房的**全部**文件。
    """
    by_room: dict[str, list[str]] = {}
    for path in paths:
        by_room.setdefault(Path(path).parts[-3], []).append(path)
    picked: list[str] = []
    for room in sorted(by_room)[:rooms]:
        picked.extend(sorted(by_room[room]))
    return picked


def scan(payload: dict, tally: dict, counters: collections.Counter) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or OUR_UID not in ids:
        return
    mine = ids.index(OUR_UID)
    for state, events in replay.iter_rounds(payload):
        ordinal = [0] * SEATS
        first_meld: list[int | None] = [None] * SEATS
        counters["局数"] += 1
        for event in events:
            seat = event.get("seat")
            kind = str(event.get("type"))
            if kind in ("chi", "peng", "gang") and isinstance(seat, int) and 0 <= seat < SEATS:
                if first_meld[seat] is None:
                    first_meld[seat] = ordinal[seat]
            elif kind == "tile_discarded":
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                # **序号只数自由决策点**：抓打圈强制的出牌不算一步（与到听率同口径），
                # 否则同一批数据会因为「谁被抓打圈」而落到不同序号格。
                if (
                    isinstance(seat, int)
                    and 0 <= seat < SEATS
                    and tile is not None
                    and state.opened
                    and not (event.get("data") or {}).get("catch_play")
                ):
                    ordinal[seat] += 1
                    counts = list(state.seats[seat].hand)
                    if counts[tile] > 0:
                        counts[tile] -= 1
                    melds = len(state.seats[seat].melds)
                    try:
                        value = shanten_module.shanten_any(counts, melds)
                    except Exception:  # noqa: BLE001 —— 算不出来就计入失败数，不猜
                        counters["向听失败"] += 1
                        value = -1
                    if value >= 0:
                        group = US if seat == mine else THEM
                        # **按当前副露数分层**：出牌序号不是完美的机会度量，副露多的一方
                        # 不出牌也在推进。把「0 副露」与「已有副露」分开后，
                        # 若缺口在 0 副露层里也照样存在，那才不能归因于副露。
                        meld_layer = "0副露" if melds == 0 else "有副露"
                        bucket = min(ordinal[seat], MAX_ORDINAL)
                        cell = tally[(group, meld_layer, bucket)]
                        cell[0] += 1
                        if value == 0:
                            cell[1] += 1
            replay.apply_event(state, event)
        # **首次副露发生在第几手**：层内对比还差一个混淆——同为「1 组副露」，
        # 在第 3 手动和第 9 手动，可利用的巡数完全不同。这条量它。
        for seat, index in enumerate(first_meld):
            if index is not None:
                counters[f"{US if seat == mine else THEM} 首次副露序号"] += index
                counters[f"{US if seat == mine else THEM} 首次副露次数"] += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按出牌序号的到听率（同机会比较）")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument("--rooms", type=int, default=12, help="**按房**抽样的房数（不是文件数）")
    args = parser.parse_args(argv)

    paths = rooms_first(sorted(glob.glob(args.events)), args.rooms)
    if not paths:
        print("没有文件", file=sys.stderr)
        return 1
    tally: dict = collections.defaultdict(lambda: [0, 0])
    counters: collections.Counter = collections.Counter()
    kept = 0
    for index, path in enumerate(paths):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}", file=sys.stderr)
            continue
        before = counters["局数"]
        scan(payload, tally, counters)
        kept += counters["局数"] > before
        if (index + 1) % 50 == 0:
            print(f"  ... {index + 1}/{len(paths)}", file=sys.stderr, flush=True)

    print(f"文件 {len(paths)} 个（含本家 {kept} 个）；局数 {counters['局数']}；"
          f"向听不可算 {counters['向听失败']} 次\n")
    print("== 按出牌序号的到听率（序号相同 = 机会相同）==")
    for layer in ("0副露", "有副露"):
        print(f"\n-- {layer} 层 --")
    print(f"{'序号':>4s} {'我们 n':>8s} {'我们%':>7s} {'对手 n':>8s} {'对手%':>7s} {'差':>7s}")
    for k in range(1, MAX_ORDINAL + 1):
        ours, theirs = tally[(US, layer, k)], tally[(THEM, layer, k)]
        if ours[0] < 30 or theirs[0] < 90:  # 样本不足不报数，避免把噪声当趋势
            continue
        label = f"{k}" if k < MAX_ORDINAL else f"{k}+"
        pa, pb = 100 * ours[1] / ours[0], 100 * theirs[1] / theirs[0]
        print(f"{label:>4s} {ours[0]:>8d} {pa:>7.1f} {theirs[0]:>8d} {pb:>7.1f} {pa - pb:>+7.1f}")

    print("\n== 差距的构成分解（副露「份额效应」vs「层内水平效应」）==")
    print("口径：令 p = **0 副露层占比**，则到听率 = p×r0 + (1−p)×r1。以我们为参照拆「我们 − 对手」：")
    print("  份额效应 = (p_我们 − p_对手) × (r0_我们 − r1_我们)")
    print("  水平效应 = p_对手 × (r0_我们 − r0_对手) + (1−p_对手) × (r1_我们 − r1_对手)")
    print("  交叉项为余数。**份额效应大 ⇒ 是「他们副露更多」的构成问题；水平效应大 ⇒ 是层内打法差异。**")
    print(f"{'序号':>4s} {'我们0副露占比':>13s} {'对手0副露占比':>13s} {'总差':>7s} "
          f"{'份额':>7s} {'水平':>7s} {'交叉':>7s}")
    for k in range(2, MAX_ORDINAL + 1):
        cells = {}
        ok = True
        for who in (US, THEM):
            a, b = tally[(who, "0副露", k)], tally[(who, "有副露", k)]
            if a[0] + b[0] < 120 or a[0] < 25 or b[0] < 25:
                ok = False
                break
            cells[who] = (a[0] / (a[0] + b[0]), a[1] / a[0], b[1] / b[0])
        if not ok:
            continue
        pu, r0u, r1u = cells[US]
        pt, r0t, r1t = cells[THEM]
        total = (pu * r0u + (1 - pu) * r1u) - (pt * r0t + (1 - pt) * r1t)
        share_eff = (pu - pt) * (r0u - r1u)
        rate_eff = pt * (r0u - r0t) + (1 - pt) * (r1u - r1t)
        label = str(k) if k < MAX_ORDINAL else f"{k}+"
        print(f"{label:>4s} {pu:>13.1%} {pt:>13.1%} "
              f"{100 * total:>+7.1f} {100 * share_eff:>+7.1f} {100 * rate_eff:>+7.1f} "
              f"{100 * (total - share_eff - rate_eff):>+7.1f}")
    print("\n读法：前几序号打平、中段开始落后 ⇒ 中段（向听 2~1）是缺口所在；"
          "\n全程落后 ⇒ 起手/早期就有结构问题。")
    print("\n== 首次副露发生在第几手（层内混淆的补量）==")
    for group in (US, THEM):
        n = counters[f"{group} 首次副露次数"]
        if n:
            print(f"  {group}: 均值 **{counters[f'{group} 首次副露序号'] / n:.2f}** 手（n={n}）")
    print("  同为「1 组副露」，早副露可利用的巡数更多 ⇒ 若我们明显更晚，"
          "「有副露层」的水平差就有一部分是**时机**而非打法。")
    print("⚠ 混淆：出牌序号不是完美的机会度量——副露多的一方不出牌也在推进，"
          "而对手副露是我们的 1.85 倍。引用时必须写明这条。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
