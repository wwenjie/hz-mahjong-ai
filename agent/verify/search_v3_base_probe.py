"""独立探针（两阶段版）：SearchDecider 的「v3 底」是否为空干预？

为什么重写：第一版对 120 个**随机**摸牌局面测出 0 差异，但其中听牌（top_shanten==0）
只有 2 个——而听牌正是 `wait_aware_tenpai` 唯一可能生效的子集。样本薄，不能靠它定论。
本版分两阶段：① 便宜地扫大量发牌、只留下**听牌**局面；② 只在那些局面上跑贵的比较。

静态先验（待实证）
----------------
- v2 vs v3 配置差**只有** `wait_aware_tenpai`。
- 该开关只被 `HeuristicDecider._break_ties_by_ukeire` 读（policy.py:757），
  而它只被 `_choose_discard` 调（policy.py:728）。
- `SearchDecider._rank_discards` → `self.heuristic._score_discard`（search.py:131），
  其配置依赖里**没有** `wait_aware_tenpai`。
- `SearchDecider.choose` 只在 `first` 非出牌（胡/杠）或 budget<200ms 时采用启发式结果；
  主路径一律被搜索自身结果替换。rollout 用 `FastDecider`（与两个底都无关）。
"""

from __future__ import annotations

import random
import sys
import time

sys.path.insert(0, "/home/wuwenjie01/majiang_ai/src")

from majiang.rules import shanten as shanten_module  # noqa: E402
from majiang.rules.action import legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import round as round_module  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode  # noqa: E402
from majiang.strategy.search import SearchConfig, SearchDecider  # noqa: E402
from majiang.strategy.versions import build as build_version  # noqa: E402

BUDGET_MS = 1800


def top_shanten(situation) -> int:
    """该局面的最小向听（打完任一张后的最小值）。"""
    hand = situation.hand.counts
    meld_count = situation.hand.meld_count
    best = 99
    for tile in range(34):
        if hand[tile] <= 0:
            continue
        after = list(hand)
        after[tile] -= 1
        try:
            best = min(best, shanten_module.shanten_any(after, meld_count))
        except Exception:  # noqa: BLE001
            continue
    return best


def collect_tenpai(scan: int, seed: int):
    """阶段①：扫 scan 个发牌，返回听牌局面（带 actions）。"""
    rng = random.Random(seed)
    found = []
    for _ in range(scan):
        st = round_module.deal(rng, dealer=rng.randrange(4))
        seat = st.turn
        round_module._draw(st, seat)  # noqa: SLF001
        sit = round_module.situation_for(st, seat, PHASE_DRAW)
        if top_shanten(sit) != 0:
            continue
        actions = legal_actions(sit)
        if sum(1 for a in actions if a.kind == "discard") < 3:
            continue
        found.append((sit, actions))
    return found


def _act(decider, situation, actions):
    a = decider.choose(situation, actions, budget_ms=BUDGET_MS)
    return None if a is None else (a.kind, a.tile)


def main() -> int:
    import dataclasses

    c2 = build_version("v2", Mode.QUALIFIER).config
    c3 = build_version("v3", Mode.QUALIFIER).config
    diffs = [
        (f.name, getattr(c2, f.name), getattr(c3, f.name))
        for f in dataclasses.fields(c2)
        if getattr(c2, f.name) != getattr(c3, f.name)
    ]
    print(f"[仪表①] v2 vs v3 配置差异 = {diffs}")
    assert diffs == [("wait_aware_tenpai", False, True)], "前提被打破，结论不适用"

    t0 = time.time()
    cases = collect_tenpai(scan=4000, seed=31415926)
    print(f"[阶段①] 扫 4000 副发牌，筛出**听牌**局面 {len(cases)} 个（{time.time()-t0:.0f}s）")
    if len(cases) < 20:
        print("!! 听牌样本仍不足 20，实证腿太薄，不据以下结论")

    cfg = SearchConfig(samples=6, top_k=2)
    h2 = HeuristicDecider(c2)
    h3 = HeuristicDecider(c3)
    d2 = SearchDecider(build_version("v2", Mode.QUALIFIER), cfg)
    d3 = SearchDecider(build_version("v3", Mode.QUALIFIER), cfg)

    n = heur_diff = search_diff = rank_diff = 0
    for sit, actions in cases:
        n += 1
        if _act(h2, sit, actions) != _act(h3, sit, actions):
            heur_diff += 1
        if _act(d2, sit, actions) != _act(d3, sit, actions):
            search_diff += 1
        r2 = [(x.tile, x.total) for x in d2._rank_discards(sit, actions)]
        r3 = [(x.tile, x.total) for x in d3._rank_discards(sit, actions)]
        if r2 != r3:
            rank_diff += 1

    print()
    print(f"听牌局面 n={n}")
    print(f"  启发式 v2≠v3:          {heur_diff}   （>0 ⇒ v3 修复本身有效）")
    print(f"  搜索 v2底≠v3底:        {search_diff}")
    print(f"  _rank_discards v2≠v3:  {rank_diff}")
    print()
    if search_diff == 0 and rank_diff == 0:
        print("⇒ H 成立：**在搜索路径里 v3 底与 v2 底逐位等价（空干预）**。")
        print("   v3 的 wait-aware 修复落在 `_break_ties_by_ukeire`，搜索只走 `_score_discard`，不经过它。")
    else:
        print(f"⇒ H 被证伪：搜索路径里 v3 底确有 {search_diff} 处行为差异。")
    print(f"用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
