"""独立复算：`search-v3` / `search-deep-v3` vs `v3`（不 import agent-d 的 nnrl.eval）。

为什么能「从原始对局重实现」
--------------------------
`run_ab.py` 只落**聚合值**（mean/se/t/n），不落逐场明细。但 agent-d 的 `paired_ab`
保证：**每场只由 `(names, index, rounds, base_score, seed)` 完全确定**，
对局种子 = `seed * 100003 + index`，`start_dealer = index % SEATS`。
⇒ 我可以**用自己的实现重跑同一批对局**，与它的聚合读数逐位对照——这正是独立复算。

本脚本只读主仓（`src/majiang/**`），完全**不 import** `nnrl.*`。
臂构造按 agent-d 的 `nnrl/v3arms.py` 配方（等价于主仓 `search` 但内层换 v3）：
  search-v3      = SearchDecider(versions.build("v3", QUALIFIER), SearchConfig(samples=6,  top_k=2))
  search-deep-v3 = SearchDecider(versions.build("v3", QUALIFIER), SearchConfig(samples=16, top_k=3))
  baseline v3    = versions.build("v3", QUALIFIER)

同时验证 agent-d 点名的两点：
  ⓐ `_rank_discards` 对 w/wo `wait_aware_tenpai` 逐点相同（空干预的直接检验）；
  ⓑ `search-deep-v3` 各段与 `search-v3` 同向（此处复算每一段的 5 项指标）。

用法（链退出、负载降下后再跑）：
  nice -n 19 PYTHONPATH=src .venv/bin/python agent/verify/recompute_search_v3.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

REPO = Path("/home/wuwenjie01/majiang_ai")
sys.path.insert(0, str(REPO / "src"))

from majiang.cli import DECIDERS  # noqa: E402
from majiang.sim.batch import run_match  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode  # noqa: E402
from majiang.strategy.search import SearchConfig, SearchDecider  # noqa: E402
from majiang.strategy.versions import build as build_version  # noqa: E402

LAMBDA = 100003
SEATS = 4
LABELS = ("总得分", "名次分", "白板数", "胡次数", "番数总和")


def make(name: str):
    """按名字造决策器（自带，不 import nnrl）。"""
    if name == "search-v3":
        return SearchDecider(build_version("v3", Mode.QUALIFIER), SearchConfig(samples=6, top_k=2))
    if name == "search-deep-v3":
        return SearchDecider(build_version("v3", Mode.QUALIFIER), SearchConfig(samples=16, top_k=3))
    if name == "v3":
        return build_version("v3", Mode.QUALIFIER)
    if name == "v2":
        return build_version("v2", Mode.QUALIFIER)
    if name in DECIDERS:
        from majiang.cli import make_decider

        return make_decider(name, Mode.QUALIFIER)
    raise SystemExit(f"未知臂 {name!r}")


def play(names: list[str], index: int, *, rounds: int, base_score: int, seed: int):
    """复刻 eeval._play：同种子同起庄，返回四座 5 元组。"""
    deciders = [make(n) for n in names]
    result = run_match(
        deciders,
        rounds=rounds,
        base_score=base_score,
        seed=seed * LAMBDA + index,
        labels=names,
        start_dealer=index % SEATS,
    )
    return tuple(
        (s.total_score, s.place_points, s.god_count, s.wins, s.fan_total) for s in result.seats
    )


def paired_ab(treatment: str, baseline: str, *, matches: int, rounds: int, base_score: int, seed: int):
    """复刻 eeval.paired_ab 的配对逻辑（默认场地=baseline）。"""
    field_list = [baseline] * 3

    def names_for(rotation: int, seat_name: str) -> list[str]:
        row = [""] * SEATS
        row[rotation] = seat_name
        for seat, nm in zip([s for s in range(SEATS) if s != rotation], field_list):
            row[seat] = nm
        return row

    base_runs = [play([baseline] * SEATS, i, rounds=rounds, base_score=base_score, seed=seed)
                 for i in range(matches)]
    diffs: dict[str, list[float]] = {lab: [] for lab in LABELS}
    t_score = b_score = 0
    for rotation in range(SEATS):
        names = names_for(rotation, treatment)
        for i in range(matches):
            mine = play(names, i, rounds=rounds, base_score=base_score, seed=seed)[rotation]
            theirs = base_runs[i][rotation]
            for lab, a, b in zip(LABELS, mine, theirs):
                diffs[lab].append(a - b)
            t_score += mine[0]
            b_score += theirs[0]
    return diffs, t_score, b_score


def stat(vals: list[float]):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    return mean, se, (mean / se if se else 0.0), n


def check_rank_discards_identity(samples: int = 4000, want: int = 21) -> None:
    """ⓐ `_rank_discards` 对 w/wo `wait_aware_tenpai` 是否逐点相同（听牌局面）。"""
    from majiang.rules import shanten as shanten_module
    from majiang.rules.action import legal_actions
    from majiang.rules.situation import PHASE_DRAW
    from majiang.sim import round as round_module

    c2 = build_version("v2", Mode.QUALIFIER).config
    c3 = build_version("v3", Mode.QUALIFIER).config
    d2 = SearchDecider(HeuristicDecider(c2), SearchConfig(samples=6, top_k=2))
    d3 = SearchDecider(HeuristicDecider(c3), SearchConfig(samples=6, top_k=2))
    rng = random.Random(31415926)
    found = checked = diff = 0
    while found < want:
        st = round_module.deal(rng, dealer=rng.randrange(4))
        seat = st.turn
        round_module._draw(st, seat)  # noqa: SLF001
        sit = round_module.situation_for(st, seat, PHASE_DRAW)
        hand = sit.hand.counts
        mc = sit.hand.meld_count
        best = 99
        for t in range(34):
            if hand[t] <= 0:
                continue
            after = list(hand)
            after[t] -= 1
            try:
                best = min(best, shanten_module.shanten_any(after, mc))
            except Exception:  # noqa: BLE001
                continue
        if best != 0:
            continue
        found += 1
        actions = legal_actions(sit)
        r2 = [(x.tile, x.total) for x in d2._rank_discards(sit, actions)]
        r3 = [(x.tile, x.total) for x in d3._rank_discards(sit, actions)]
        checked += 1
        if r2 != r3:
            diff += 1
    print(f"ⓐ `_rank_discards` 听牌局面 n={checked}  w/wo wait_aware 有差异: {diff}")


def main() -> int:
    seeds = [20260928, 771014]
    matches, rounds, base_score = 40, 8, 1

    print("=" * 74)
    print("独立复算（不 import nnrl.eval；主仓只读）")
    print("=" * 74)
    check_rank_discards_identity()

    results = {}
    # 读 agent-d 已落盘的聚合值（若有）供对照
    ref_path = Path("/home/wuwenjie01/majiang_rl/records/ab-search-v3-vs-v3.json")
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
    if ref:
        print(f"[对照] 已读取 agent-d 落盘 {ref_path.name}")
    else:
        print("[对照] agent-d 的 ab-search-v3-vs-v3.json 尚未落盘——仅出我方复算值")

    for arm in ("search-v3", "search-deep-v3"):
        for seed in seeds:
            diffs, ts, bs = paired_ab(arm, "v3", matches=matches, rounds=rounds,
                                      base_score=base_score, seed=seed)
            row = {}
            print(f"\n[{arm} seed={seed}] n={matches * SEATS}/指标（treatment − baseline=v3）")
            for lab in LABELS:
                m, se, t, n = stat(diffs[lab])
                row[lab] = {"mean": m, "se": se, "t": t, "n": n}
                refm = ""
                if ref:
                    try:
                        r = ref["arms"][arm][str(seed)][lab]
                        refm = f"   | agent-d: {r['mean']:+.3f} (t{r['t']:+.2f})"
                    except Exception:  # noqa: BLE001
                        refm = ""
                print(f"   {lab:8s} {m:+8.3f} ±{se:5.3f}  t {t:+6.2f}  n={n}{refm}")
            row["_scores"] = {"treatment": ts, "baseline": bs}
            results[f"{arm}@{seed}"] = row

    out = REPO / "agent" / "out" / "recompute-search-v3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n复算结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
