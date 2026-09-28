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


def live_copies(
    counts: list[int], melds_flat: list[list[int]], discards: list[list[int]], waits
) -> int:
    """听口里还剩几张可摸 = Σ(4 − 已见)。已见 = 本方暗手 + 四家副露 + 四家弃牌。

    ``melds_flat``/``discards`` 必须是**按座位的二维序列**：``visible_counts`` 内部
    会对每个元素再迭代一次，传扁平的一维会在 ``for tile in group`` 上炸
    "int object is not iterable"（同一坑踩过三次，这里写死说明）。
    """
    visible = shanten_module.visible_counts(counts, melds_flat, discards)
    return sum(max(0, tiles.COPIES_PER_KIND - visible[tile]) for tile in waits)


def decider_by_room() -> dict[str, str]:
    """房间 → 当时跑的策略名（来自 `data/auto_sessions/sessions.jsonl`）。

    **为什么必须有这道过滤**：冻结清单 `notes/manifest-20260926.txt` 是 09-26 冻结的，
    而我们在 09-27 把默认档从 v1（`tiebreak=blocks`，**完全不看进张**）切到了
    v2（`exact-ukeire`）。清单里绝大多数文件是 v1 时代的——在整份清单上算出来的
    「我们的听口 regret」是**两个策略的混合**，既不代表 v1 也不代表 v2，
    拿它当「当前档位还有多少可改进空间」会系统性高估。同一条警告适用于任何
    在整份清单上做「我们 vs 对手」对照的分析。
    """
    ledger = Path("data/auto_sessions/sessions.jsonl")
    mapping: dict[str, str] = {}
    if not ledger.exists():
        return mapping
    for line in ledger.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        room = entry.get("room_id")
        decider = entry.get("decider")
        if room and decider:
            mapping[str(room)] = str(decider)
    return mapping


def room_of(path: str) -> str:
    """从 `data/auto_sessions/<room>/events/<file>.json` 取 `<room>`。"""
    return Path(path).parts[-3]


def filter_era(paths: list[str], eras: list[str]) -> list[str]:
    if not eras:
        return paths
    mapping = decider_by_room()
    unknown = 0
    kept: list[str] = []
    for path in paths:
        decider = mapping.get(room_of(path))
        if decider is None:
            unknown += 1
            continue
        if any(decider.startswith(era) for era in eras):
            kept.append(path)
    if unknown:
        print(f"  （{unknown} 个文件的房间不在台账里，已剔除——宁可少算也不混口径）",
              file=sys.stderr)
    return kept


def god_bucket(gods: int) -> str:
    return "0张" if gods == 0 else ("1张" if gods == 1 else "≥2张")


def ceiling(
    counts: list[int],
    meld_count: int,
    melds_flat: list[list[int]],
    discards: list[list[int]],
    chosen: int,
) -> dict | None:
    """核心度量：给定「14 张手牌 + 公开量 + 实际打出的那张」，返回 regret 那组数。

    **只用公开信息与自己的暗手**：``melds_flat``/``discards`` 是四家的副露与弃牌，
    这与真机决策时的信息集一致——本函数既能在历史事件流上跑，也能在自对弈里跑
    （两条路都把同一个 ``(手牌, 副露, 弃牌)`` 喂进来），所以两边**同一把尺子**。
    """
    if sum(counts) % 3 != 2:
        return None
    options: dict[int, int] = {}
    for tile in range(tiles.TILE_KINDS):
        if counts[tile] <= 0:
            continue
        after = list(counts)
        after[tile] -= 1
        try:
            value = shanten_module.shanten_any(after, meld_count)
        except Exception:  # noqa: BLE001 —— 算不出向听的候选跳过，不猜
            continue
        if value != 0:
            continue  # 打完不听牌：不属于「听口选择」问题（弃胡求爆头是另一个决策）
        waits = win_module.winning_draws(after, meld_count)
        if not waits:
            continue
        options[tile] = live_copies(after, melds_flat, discards, waits)
    if chosen not in options or len(options) < 2:
        return None
    mine = options[chosen]
    best = max(options.values())
    better = sum(1 for value in options.values() if value > mine)
    god_after = counts[tiles.GOD] - (1 if chosen == tiles.GOD else 0)
    return {
        "god": god_bucket(god_after),
        "n_options": len(options),
        "mine": mine,
        "best": best,
        "regret": best - mine,
        "percentile": better / (len(options) - 1),
        # **风险度量**：这版改动会不会为了宽听口而把财神打出去？财神是唯一的链货币，
        # 也是「4 白板 ×2」的那一环；若 regret 降了但打财神变多，就是拿一条轴的收益
        # 换另一条轴的损失，必须同时看。
        "chosen_is_god": chosen == tiles.GOD,
    }


def evaluate(state: replay.ReplayState, seat: int, chosen: int) -> dict | None:
    """历史事件流路径：从 ``ReplayState`` 取公开量，喂给 :func:`ceiling`。"""
    hand = list(state.seats[seat].hand)
    return ceiling(
        hand,
        len(state.seats[seat].melds),
        [meld.tiles for other in state.seats for meld in other.melds],
        [list(other.discards) for other in state.seats],
        chosen,
    )


class _CeilingProbe:
    """自对弈路径：包一层 decider，在**我们**的出牌决策上量同一个 regret。

    **为什么必须做成 decider 包装而不是 observer**：包装层拿到的 ``situation`` 就是
    真机决策时的那份信息集，量出来的 regret 与历史事件流那一侧**同口径**；
    若改用 observer 钩子去读 ``RoundState``，就多出了一条「只有自对弈才有」的读法，
    两边结果不可比——而不可比正是过去踩过的那类错配。
    """

    def __init__(self, inner, rows: list[dict], seat: int = 0) -> None:
        self.inner = inner
        self.rows = rows
        self.seat = seat

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "probe")

    def configure(self, *args, **kwargs):  # noqa: ANN002, ANN003 —— 透传，真机路径才用
        return self.inner.configure(*args, **kwargs)

    def observe_state(self, *args, **kwargs):  # noqa: ANN002, ANN003 —— 兼容 observer 协议
        handler = getattr(self.inner, "observe_state", None)
        if handler is not None:
            handler(*args, **kwargs)

    def choose(self, situation, actions, *, budget_ms: int = 0):  # noqa: ANN001, ANN201
        from majiang.rules.action import DISCARD  # noqa: PLC0415

        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        if situation.seat == self.seat and choice is not None and choice.kind == DISCARD:
            row = ceiling(
                list(situation.hand.counts),
                situation.hand.meld_count,
                [list(meld.tiles) for meld in situation.all_melds],
                [list(seat) for seat in situation.discards],
                choice.tile,
            )
            if row is not None:
                self.rows.append(row)
        return choice


def selfplay_rows(
    decider_name: str, field: str, matches: int, rounds: int, seed: int
) -> list[dict]:
    """在自对弈里用同一把尺子量 regret——这是验证「干预真的动过那条轴」的手段。"""
    import random  # noqa: PLC0415

    from majiang.cli import make_decider  # noqa: PLC0415
    from majiang.sim.batch import next_dealer  # noqa: PLC0415
    from majiang.sim.round import run_round  # noqa: PLC0415
    from majiang.strategy.policy import Mode  # noqa: PLC0415

    rows: list[dict] = []
    for index in range(matches):
        # **0 号座的档位在每个座位上都跑一遍**：只用 index%SEATS 当庄会把
        # 庄闲赔付的差异混进 regret 分布里。
        for seat in range(SEATS):
            deciders = [make_decider(field, Mode.QUALIFIER) for _ in range(SEATS)]
            deciders[seat] = _CeilingProbe(make_decider(decider_name, Mode.QUALIFIER), rows, seat)
            rng = random.Random(seed * 100003 + index)
            dealer = (index + seat) % SEATS
            for round_no in range(1, rounds + 1):
                outcome = run_round(deciders, dealer=dealer, round_no=round_no, base_score=1, rng=rng)
                dealer = next_dealer(dealer, outcome)
    return rows


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
          f"  regret {regret:5.2f}（{regret / mine:6.1%}）  分位 {pct:5.1%}"
          f"  打财神 {sum(r['chosen_is_god'] for r in rows) / n:5.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="听口上限诊断")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--events", default="")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument(
        "--era",
        default="",
        help=(
            "只统计这些策略时代的房间（逗号分隔，前缀匹配，例如 `v2` 或 `heuristic,v1`）。"
            "**默认不过滤是危险的**：冻结清单跨 v1/v2 两代策略，混算出来的 regret 是"
            "两个策略的平均，会高估当前档位的可改进空间。"
        ),
    )
    parser.add_argument(
        "--decider",
        default="",
        help=(
            "自对弈模式：只量这个档位的 regret（不再读历史事件流）。**这是验证干预是否"
            "真的动过听口那条轴的唯一可用手段**——历史事件流只能量已部署的档位，"
            "未上线的改动若只用总得分判，就会重复「自对弈有差异、真机没有」的老坑。"
        ),
    )
    parser.add_argument("--field", default="heuristic", help="自对弈模式里另三座坐谁")
    parser.add_argument("--matches", type=int, default=40, help="自对弈模式跑几场")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)

    if args.decider:
        rows = selfplay_rows(args.decider, args.field, args.matches, args.rounds, args.seed)
        print(f"自对弈档位 {args.decider}（另三座 {args.field}、{args.matches} 场 × "
              f"{args.rounds} 局、四座位轮转）；计入听牌出牌点 {len(rows)} 个\n")
        print("== 听牌时「实际听口 vs 同手牌上限」（可见张数）==")
        report(rows, US)
        print("\n== 按财神数分层 ==")
        by_god: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_god[row["god"]].append(row)
        for gods in ("0张", "1张", "≥2张"):
            report(by_god.get(gods, []), gods)
        if rows:
            ratio = sum(r["regret"] for r in rows) / sum(r["mine"] for r in rows)
            print(f"\n  平均 regret 占实际听口 {ratio:.1%}"
                  f"（真机已部署档位的历史实测是 6.5%）")
        return 0

    if args.events:
        paths = sorted(glob.glob(args.events))
    else:
        paths = [
            line.strip()
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    # **先按时代过滤再截断**：冻结清单是按时间顺序排的，先截断会把新策略的文件全砍掉
    # （实测踩过：先 limit 150 再过滤 v2 得到 0 个文件）。
    eras = [item.strip() for item in args.era.split(",") if item.strip()]
    if eras:
        before = len(paths)
        paths = filter_era(paths, eras)
        print(f"时代过滤 {eras}：{before} -> {len(paths)} 个文件", file=sys.stderr)
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
