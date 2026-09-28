"""听口上限诊断：**听牌时我们选的听口，离「同一手牌能选到的最宽听口」有多远**。

**为什么做这个**：`tools/analyze_god_conversion.py` 把缺口定位到了一个很具体的地方——

| 听牌时财神 | 我们 可见张数 | 对手 可见张数 | 差 |
|---|---|---|---|
| 0 张 | 7.64 | 8.41 | −9% |
| 1 张 | 11.73 | 15.64 | **−25%** |
| ≥2 张 | 23.09 | 30.65 | **−25%** |

即：**看不到财神的听口我们大致打平，一旦手里有财神，我们的听口就窄四分之一。**
财神的全部价值就在于「把听口撑宽」，所以这不是小事。

**本工具不猜原因，只量上限**：对每一个「我们听牌」的出牌点，把**所有**候选打牌都算一遍，
看每一张打完之后的**可见张数**，然后回答三个问题：

1. 我们实际打的那张，平均还剩几张可见？
2. 同一手牌里**最好**的候选能剩几张？
3. regret = 最好 − 实际，占我们实际听口的比例是多少？

**判据（预登记）**：若平均 regret 占我们实际听口的 **< 5%**，说明听口窄不是出牌选择的问题
（而是「走到这个听牌态」的路的问题），改出牌层无用；若 **≥ 15%**，则出牌层有可直接兑现
的空间，做法是「听牌时按可见张数排序」——注意现在 `total` 在同向听里按骨架厚度与喂牌算，
**根本不看听口**，而 `ukeire` 只在 `total` 前 2 名里比，所以最宽的那张很可能压根没进候选面。

口径与边界（不写清楚会被误读）：

- 只统计**打完仍然听牌**（向听 0）的候选——打出后脱离听牌的候选不进比较（那属于
  `_choose_win_or_piao` 的「弃胡求爆头」决策，是另一个问题，不要混进来）。
- 「可见张数」= 4 − 已见（本方暗手 + 四家副露 + 四家弃牌），与
  `analyze_wait_quality.py` **同一口径**，跨工具引用时不会有歧义。
- 抓打圈强制的出牌不是自由决策点，剔除。
- 开销：`winning_draws` 约 0.1 s/次，只在候选确实听牌时才算，故必须限 `--limit`。

用法::

    uv run python tools/analyze_wait_ceiling.py --limit 150
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win as win_module
from majiang.sim import replay

SEATS = 4
DISCARDED = "tile_discarded"
US = "我们"
THEM = "对手"


def live_copies(state: replay.ReplayState, seat: int, waits: tuple[int, ...]) -> int:
    visible = shanten_module.visible_counts(
        list(state.seats[seat].hand),
        [meld.tiles for other in state.seats for meld in other.melds],
        [list(other.discards) for other in state.seats],
    )
    return sum(max(0, tiles.COPIES_PER_KIND - visible[tile]) for tile in waits)


def god_bucket(gods: int) -> str:
    return "0张" if gods == 0 else ("1张" if gods == 1 else "≥2张")


def evaluate(state: replay.ReplayState, seat: int, chosen: int) -> dict | None:
    """对一个出牌点算「实际 vs 上限」。返回 None 表示该点不适用。"""
    hand = list(state.seats[seat].hand)
    meld_count = len(state.seats[seat].melds)
    if sum(hand) % 3 != 2:
        return None
    options: dict[int, int] = {}
    for tile in range(tiles.TILE_KINDS):
        if hand[tile] <= 0:
            continue
        counts = list(hand)
        counts[tile] -= 1
        try:
            value = shanten_module.shanten_any(counts, meld_count)
        except Exception:  # noqa: BLE001 —— 向听算不出来就跳过该候选，不猜
            continue
        if value != 0:
            continue  # 打完不听牌：不属于「听口选择」问题
        waits = win_module.winning_draws(counts, meld_count)
        if not waits:
            continue
        options[tile] = live_copies(state, seat, waits)
    if chosen not in options or len(options) < 2:
        return None
    mine = options[chosen]
    best = max(options.values())
    kinds = sum(1 for value in options.values() if value > mine)
    return {
        "god": god_bucket(hand[tiles.GOD] - (1 if chosen == tiles.GOD else 0)),
        "n_options": len(options),
        "mine": mine,
        "best": best,
        "regret": best - mine,
        "percentile": kinds / (len(options) - 1),
    }


def scan(
    payload: dict, ours: str, rows: list[dict], other_rows: list[dict], counters: Counter
) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or ours not in ids:
        return
    mine = ids.index(ours)
    for state, events in replay.iter_rounds(payload):
        for event in events:
            if event.get("type") == DISCARDED:
                seat = event.get("seat")
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                forced = bool((event.get("data") or {}).get("catch_play"))
                if isinstance(seat, int) and 0 <= seat < SEATS and tile is not None and not forced:
                    if state.opened:
                        row = evaluate(state, seat, tile)
                        if row is not None:
                            if seat == mine:
                                rows.append(row)
                            else:
                                other_rows.append(row)
                            counters["计入的听牌出牌点"] += 1
            replay.apply_event(state, event)


def report(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"  {label}: 无样本")
        return
    n = len(rows)
    mine = sum(r["mine"] for r in rows) / n
    best = sum(r["best"] for r in rows) / n
    regret = sum(r["regret"] for r in rows) / n
    pct = sum(r["percentile"] for r in rows) / n
    print(f"  {label:8s} n={n:6d}  候选均 {sum(r['n_options'] for r in rows) / n:4.2f} 张"
          f"  实际可见 {mine:6.2f}  上限 {best:6.2f}"
          f"  regret {regret:5.2f}（{regret / mine:6.1%}）  分位 {pct:5.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="听口上限诊断")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--events", default="")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args(argv)

    if args.events:
        paths = sorted(glob.glob(args.events))
    else:
        paths = [
            line.strip()
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有文件", file=sys.stderr)
        return 1

    rows: list[dict] = []
    other_rows: list[dict] = []
    counters: Counter = Counter()
    for index, path in enumerate(paths):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        scan(payload, args.ours, rows, other_rows, counters)
        if (index + 1) % 20 == 0:
            print(f"  ... {index + 1}/{len(paths)}（我们 {len(rows)} 点 / "
                  f"对手 {len(other_rows)} 点）", file=sys.stderr, flush=True)

    print(f"文件 {len(paths)} 个；计入的听牌出牌点 {counters['计入的听牌出牌点']} 个\n")
    print("== 听牌时「实际听口 vs 同手牌上限」（可见张数）==")
    report(rows, US)
    report(other_rows, THEM)
    print("\n== 我们按财神数分层 ==")
    by_god: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_god[row["god"]].append(row)
    for gods in ("0张", "1张", "≥2张"):
        report(by_god.get(gods, []), gods)

    print("\n== 判据（预登记）==")
    if not rows:
        return 1
    regret_ratio = sum(r["regret"] for r in rows) / sum(r["mine"] for r in rows)
    print(f"  平均 regret 占实际听口 {regret_ratio:.1%}")
    if regret_ratio < 0.05:
        print("  < 5% → 听口窄**不是出牌选择的问题**，改出牌层无用；"
              "要往「怎么走到这个听牌态」上找（前中期结构）。")
    elif regret_ratio >= 0.15:
        print("  ≥ 15% → 出牌层有**可直接兑现**的空间：听牌时应按可见张数排序。"
              "注意当前 total 在同向听里根本不看听口，而 ukeire 只在 total 前 2 名里比。")
    else:
        print("  5–15% → 灰区，按财神分层看：若「有财神」那几层显著高于 5%，"
              "则只对有财神的听牌点改排序，收益/风险比最好。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
