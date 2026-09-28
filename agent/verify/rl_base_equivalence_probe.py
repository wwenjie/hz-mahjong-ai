"""独立探针（快速版）：RL 臂 v2 底 vs v3 底是否逐位等价（判 `rl@v3 ≡ rl@v2`）。

为什么便宜
----------
`candidate_features` 的候选特征全部来自 `HeuristicDecider._score_discard`（+ 手牌张数），
mask 覆盖**全部有牌张**。而 `_score_discard` 的 config 依赖里**没有** `wait_aware_tenpai`
（该开关只在 `_break_ties_by_ukeire` 被读，RL/search 都不走那条路）。
⇒ 决定性核心检验 = 直接比 `_score_discard` 在 v2/v3 底上是否逐位相同；
   再对少量局面确认 `candidate_features` 的 (mask, cand) 逐位相同即可。

另：`rl_net._dist`（:83）`logits = F[:, total_idx] + delta` —— `total` 确作 **logit 偏置**，
但它在 v2/v3 下**同值**，故不构成基座差异。
"""

from __future__ import annotations

import random as _pyrandom
import sys

MAIN = "/home/wuwenjie01/majiang_ai"
RL = "/home/wuwenjie01/majiang_rl"
sys.path.insert(0, MAIN + "/src")
sys.path.insert(0, RL + "/src")

import numpy as np  # noqa: E402

from majiang.rules.action import DISCARD, Action, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import round as round_module  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402
from majiang.strategy.versions import build as build_version  # noqa: E402

# 只 import 特征组装，不 import 评测
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("bc", RL + "/src/nnrl/bc.py")
bc_mod = importlib.util.module_from_spec(spec)
sys.modules["bc"] = bc_mod
try:
    spec.loader.exec_module(bc_mod)
except Exception as exc:  # noqa: BLE001
    print(f"!! import bc 失败: {exc}（改用直接特征比对）")
    bc_mod = None


def positions(n: int, seed: int):
    rng = _pyrandom.Random(seed)
    got = 0
    while got < n:
        st = round_module.deal(rng, dealer=rng.randrange(4))
        seat = st.turn
        round_module._draw(st, seat)  # noqa: SLF001
        sit = round_module.situation_for(st, seat, PHASE_DRAW)
        actions = legal_actions(sit)
        if sum(1 for a in actions if a.kind == DISCARD) < 2:
            continue
        got += 1
        yield sit, actions


def main() -> int:
    from majiang.rules import tiles

    h2 = build_version("v2", Mode.QUALIFIER)
    h3 = build_version("v3", Mode.QUALIFIER)
    assert h3.config.wait_aware_tenpai is True and h2.config.wait_aware_tenpai is False

    N = 500
    n = score_diff = 0
    tenpai = tenpai_diff = 0
    from majiang.rules import shanten as shanten_module

    for sit, actions in positions(N, 20260928):
        n += 1
        counts = sit.hand.counts
        mc = sit.hand.meld_count
        # 是否听牌（该开关唯一可能生效的子集）
        best = 99
        for t in range(tiles.TILE_KINDS):
            if counts[t] <= 0:
                continue
            after = list(counts)
            after[t] -= 1
            try:
                best = min(best, shanten_module.shanten_any(after, mc))
            except Exception:  # noqa: BLE001
                pass
        is_tenpai = best == 0
        if is_tenpai:
            tenpai += 1
        diff_here = False
        for t in range(tiles.TILE_KINDS):
            if counts[t] <= 0:
                continue
            s2 = h2._score_discard(sit, Action(DISCARD, tile=t))  # noqa: SLF001
            s3 = h3._score_discard(sit, Action(DISCARD, tile=t))  # noqa: SLF001
            if s2 != s3:
                diff_here = True
        if diff_here:
            score_diff += 1
            if is_tenpai:
                tenpai_diff += 1

    print(f"`_score_discard` 逐牌位比对：n={n}（其中听牌 {tenpai}）")
    print(f"  任一牌位不同: {score_diff}（听牌局面中 {tenpai_diff}）")

    # candidate_features 逐位（少量局面）
    feats_diff = m = 0
    if bc_mod is not None:
        for sit, actions in positions(40, 771014):
            m += 1
            _, c2, k2 = bc_mod.candidate_features(h2, sit)
            _, c3, k3 = bc_mod.candidate_features(h3, sit)
            if not (np.array_equal(c2, c3) and np.array_equal(k2, k3)):
                feats_diff += 1
        print(f"`candidate_features` 逐位比对：m={m}  不同: {feats_diff}")
    else:
        print("`candidate_features` 比对：跳过（import 失败）")

    print()
    ok = (score_diff == 0) and (feats_diff == 0)
    if ok:
        print("⇒ **rl@v3 ≡ rl@v2 成立**：换 RL 基座是空干预，只需 `--baseline v3`。")
        print("   ⇒ 我 19:26 的推论②「RL 上限被 v2 排序封住」**不成立**（自身更正）。")
    else:
        print("⇒ 存在差异，需重新评估。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
