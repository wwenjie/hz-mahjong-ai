#!/usr/bin/env python
"""预筛召回率 @ N 探针：`blocks` 代理排序的前 N 名，能否圈住「精确进张真·前 1 / 前 3」。

**背景**（2026-10-01 19:45，A 的派活）：

- 候选面加宽的价值已证（cand5 +0.639 > cand3 +0.451），但真机延迟卡住
  （v5 p99 860ms / 预算 1800ms，每张候选一次精确进张 42–157ms）。
- A 已落地 `ukeire_preselect`：候选面开到 10 张、但只对 `blocks` 代理的前 5 张
  算精确进张（`v5-presel5`）。初版用 `cheap_ukeire` 当代理被成本判据否掉
  （158s > cand10 的 144s），改成零成本的 `blocks`（`_score_discard` 早算好的
  骨架厚度）后 128s ≈ cand5。
- **还没人量过召回率**：`blocks` 排序的前 N 名，有多大把握把「精确进张真·最优」
  圈进去？这正是预筛「几乎无损」还是「砍掉收益」的分水岭。

**本脚本的做法**：

1. 跑 `v5-cand10` 档（精确进张前 10 张、不预筛）自对弈若干种子，在**每个触发
   精确进张的决策点**把「并列候选全集 + 每张的精确进张 copies」完整录下来。
2. 离线对每个决策点：把候选按 `blocks` 降序排（同生产代码的代理键），
   看真·最优（copies argmax）落在 blocks 前 N 名的比例（N=5/6/8），
   以及真·前 3 名**全部**落在 blocks 前 N 名的比例（更关键：预筛砍掉的候选
   不会进最终比较，所以丢的是真·前 3 而不是真·第 1）。
3. 同时按向听层（0=听牌 / 1 / ≥2）拆开报——听牌层走 `_wait_copies`（可见听口），
   与 `ukeire` 的口径不同，召回率可能分层。

**口径**：与生产代码共享同一个 `_break_ties_by_ukeire`（通过 `InstrumentedDecider`
子类化 `HeuristicDecider` 并在每次调用后从 `last_detail` 取结果 + 重放计算
copies）。这样量的就是**线上真实会走到的候选与并列集**，不是重建器。

用法::

    uv run python verify-b/probe_preselect_recall.py --seeds 4 --matches 40 \
        --out verify-b/out/preselect-recall.json

退出码：0 正常；1 样本不足。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, "src")

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win
from majiang.sim.batch import run_match
from majiang.strategy import versions
from majiang.strategy.policy import (
    EXACT_UKEIRE_BUDGET_SEC,
    DiscardScore,
    HeuristicDecider,
    PolicyConfig,
    _goodshape_proxy,
    _wait_copies,
)


@dataclass
class ProbeEvent:
    """一次触发精确进张比较的决策点快照。"""

    shanten: int
    n_tied: int
    # 每张候选：tile, blocks(代理键), copies(精确进张或可见听口)
    rows: list[tuple[int, int, int]] = field(default_factory=list)


class InstrumentedDecider(HeuristicDecider):
    """在 v5-cand10 之上挂探针：把每个精确进张决策点的候选全集 + copies 录下来。"""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(config)
        self.events: list[ProbeEvent] = []

    def _break_ties_by_ukeire(self, situation, scores):  # noqa: ANN001
        top_shanten = scores[0].shanten
        tied = [s for s in scores if s.shanten == top_shanten]
        # 与生产代码同序：ukeire_order == "blocks" 时先按 blocks 重排
        if self.config.ukeire_order == "blocks":
            tied = sorted(tied, key=lambda item: -item.blocks)
        # 生产代码此时会截到前 ukeire_candidates；我们**不截**，对全集算 copies
        exact = self.config.tiebreak in ("exact-ukeire", "tenpai-only")
        if exact and top_shanten <= self.config.ukeire_max_shanten and len(tied) >= 2:
            wait_aware = top_shanten == 0 and self.config.wait_aware_tenpai
            two_ply = exact and top_shanten == 1 and self.config.two_ply_shanten1
            visible = shanten_module.visible_counts(
                situation.hand.counts,
                [meld.tiles for meld in situation.all_melds],
                situation.discards,
            )
            memo: dict = {}
            rows: list[tuple[int, int, int]] = []
            for score in tied:
                counts = list(situation.hand.counts)
                counts[score.tile] -= 1
                if wait_aware:
                    copies = _wait_copies(counts, situation.hand.meld_count, visible, score.tile)
                    if copies is None:
                        continue
                elif two_ply:
                    copies = int(
                        __import__("majiang.strategy.policy", fromlist=["_two_ply_value"])._two_ply_value(
                            counts, situation.hand.meld_count, visible, memo
                        )
                    )
                else:
                    entries = shanten_module.ukeire(
                        counts, situation.hand.meld_count, visible=visible, memo=memo
                    )
                    copies = sum(copy for _, copy in entries)
                rows.append((score.tile, score.blocks, int(copies)))
            if len(rows) >= 2:
                self.events.append(ProbeEvent(shanten=top_shanten, n_tied=len(rows), rows=rows))
        # 然后照常走生产路径（行为不变）
        return super()._break_ties_by_ukeire(situation, scores)


def _recall_at_n(events: list[ProbeEvent], ns: list[int]) -> dict:
    """对每个 N：真·前 1 / 真·前 3 落在 blocks 前 N 名的比例。"""
    out: dict[str, dict] = {}
    for n in ns:
        hit1 = hit3 = total = 0
        by_shanten: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])  # [hit1, hit3, n]
        for ev in events:
            # blocks 降序（同键时 tile 升序，与 sorted 稳定性一致）
            by_blocks = sorted(ev.rows, key=lambda r: (-r[1], r[0]))
            topn_tiles = {r[0] for r in by_blocks[:n]}
            # 真·排序：copies 降序，copies 同则取 tile 小者（与生产 argmax 一致取首个最大）
            by_copies = sorted(ev.rows, key=lambda r: (-r[2], r[0]))
            true_best = by_copies[0][0]
            true_top3 = {r[0] for r in by_copies[:3]}
            hit1 += true_best in topn_tiles
            hit3 += true_top3.issubset(topn_tiles)
            total += 1
            bucket = by_shanten[ev.shanten]
            bucket[0] += true_best in topn_tiles
            bucket[1] += true_top3.issubset(topn_tiles)
            bucket[2] += 1
        out[str(n)] = {
            "recall@1": hit1 / total if total else None,
            "recall@3": hit3 / total if total else None,
            "n_events": total,
            "by_shanten": {
                str(s): {"recall@1": v[0] / v[2], "recall@3": v[1] / v[2], "n": v[2]}
                for s, v in sorted(by_shanten.items())
            },
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--matches", type=int, default=40, help="每种子局数（run_match 的 rounds）")
    parser.add_argument("--base-seed", type=int, default=20261001)
    parser.add_argument("--out", default="verify-b/out/preselect-recall.json")
    args = parser.parse_args()

    # v5-cand10 的 knobs：候选面 10、不预筛——这样我们能录到完整并列集
    knobs = {
        "wait_aware_tenpai": True,
        "shape_value": True,
        "ukeire_order": "blocks",
        "ukeire_max_shanten": 3,
        "ukeire_candidates": 10,
        "ukeire_preselect": 0,
    }
    config = PolicyConfig.for_mode(__import__("majiang.strategy.policy", fromlist=["Mode"]).Mode.QUALIFIER, **knobs)

    probe = InstrumentedDecider(config)
    t0 = time.monotonic()
    for i in range(args.seeds):
        seed = args.base_seed + i
        deciders = [probe, versions.build("v3", config.mode), versions.build("v3", config.mode), versions.build("v3", config.mode)]
        run_match(deciders, seed=seed, rounds=args.matches)
    wall = time.monotonic() - t0

    ns = [5, 6, 8]
    report = _recall_at_n(probe.events, ns)
    result = {
        "config": {"seeds": args.seeds, "matches_per_seed": args.matches, "base_seed": args.base_seed,
                   "knobs": knobs, "wall_sec": round(wall, 1)},
        "n_events": len(probe.events),
        "mean_tied": (sum(e.n_tied for e in probe.events) / len(probe.events)) if probe.events else 0,
        "recall": report,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"events={len(probe.events)}  mean_tied={result['mean_tied']:.1f}  wall={wall:.0f}s")
    for n in ns:
        r = report[str(n)]
        if r["n_events"]:
            print(f"N={n}: recall@1={r['recall@1']:.3f}  recall@3={r['recall@3']:.3f}")
            for s, v in r["by_shanten"].items():
                print(f"    向听{s}: n={v['n']}  @1={v['recall@1']:.3f}  @3={v['recall@3']:.3f}")
    if len(probe.events) < 200:
        print("WARN: 样本不足 200 个决策点，结论不稳", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
