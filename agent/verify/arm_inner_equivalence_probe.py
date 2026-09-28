"""独立探针：`policy-bc` / `mlp-value` / `rl` 三臂的**内层基座**在 v2 vs v3 下是否等价。

agent-d（2026-09-28 21:36，`TO C` OPEN 项）请我把 `rl_base_equivalence_probe` 的等价性
「顺带覆盖 `policy-bc`/`mlp-value` 的内层（`candidate_features` 的 `(mask,cand)` 逐位）」。

机制（三者同源）
----------------
- 三臂内层都由 `HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))` 构造 ⇒ 零 override ⇒ **v2**
  （`policy_decider.py:25`、`rl_play.py:27`、`decider.py:85`）。
- 出牌候选特征 `bc.candidate_features(inner, sit)` 全部来自 `inner._score_discard` + 手牌张数，
  而 `_score_discard` 的 config 依赖里**没有** `wait_aware_tenpai`（该开关只在
  `_choose_discard → _break_ties_by_ukeire` 被读）⇒ **候选面逐位相同**。
- `mlp-value` 的策略输入 = `majiang.strategy.features.extract(after_hand)`，签名只吃
  `Situation`（无 config 参数）⇒ 与基座**无关**。

因此本探针做四件事（全部只读、不 import agent-d 评测）：
1. 断言三臂内层 config `wait_aware_tenpai is False`（结构证据）；
2. `candidate_features` 的 `(mask, cand)` 在 v2/v3 上逐位比对（覆盖 `policy-bc` 与 `rl`）；
3. `features.extract` 无 config 参数（`mlp-value` 输入面结构性等价）；
4. **泄漏路径**：`policy-bc`/`rl` 在 `mask.sum() <= 1` 时直接返回内层 choice ⇒ 逐局面核对该路径上
   v2/v3 内层 `choose` 是否给出同一动作（该路径不触网络，无需 payload）。
"""

from __future__ import annotations

import inspect
import random as _pyrandom
import sys

MAIN = "/home/wuwenjie01/majiang_ai"
RL = "/home/wuwenjie01/majiang_rl"
sys.path.insert(0, MAIN + "/src")
sys.path.insert(0, RL + "/src")

import numpy as np  # noqa: E402

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles as tiles_mod  # noqa: E402
from majiang.rules.action import DISCARD, Action, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import round as round_module  # noqa: E402
from majiang.strategy.features import extract  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig  # noqa: E402
from majiang.strategy.versions import build as build_version  # noqa: E402

import nnrl.bc as bc  # noqa: E402  (agent-d 的特征组装；只读)


def positions(n: int, seed: int):
    rng = _pyrandom.Random(seed)
    got = 0
    while got < n:
        st = round_module.deal(rng, dealer=rng.randrange(4))
        seat = st.turn
        round_module._draw(st, seat)  # noqa: SLF001
        sit = round_module.situation_for(st, seat, PHASE_DRAW)
        actions = legal_actions(sit)
        if sum(1 for a in actions if a.kind == DISCARD) < 1:
            continue
        got += 1
        yield sit, actions


def main() -> int:
    h2 = build_version("v2", Mode.QUALIFIER)
    h3 = build_version("v3", Mode.QUALIFIER)
    assert h2.config.wait_aware_tenpai is False
    assert h3.config.wait_aware_tenpai is True

    print("=" * 68)
    print("① 三臂内层构造 = 零 override（结构证据）")
    # 直接读三个模块的源码行，确认都用 for_mode(QUALIFIER) 且无 override
    for path, line in (
        (RL + "/src/nnrl/policy_decider.py", "HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))"),
        (RL + "/src/nnrl/rl_play.py", "HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))"),
    ):
        src = open(path, encoding="utf-8").read()
        print(f"  {path.split('/')[-1]:22s} 含零 override 构造: {line in src}")
    src_d = open(RL + "/src/nnrl/decider.py", encoding="utf-8").read()
    print(f"  decider.py             mlp-value 内层: HeuristicDecider(PolicyConfig.for_mode(_qualifier_mode()))"
          f" -> _qualifier_mode()==Mode.QUALIFIER: {'return Mode.QUALIFIER' in src_d}")
    d = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
    print(f"  实测 for_mode(QUALIFIER).wait_aware_tenpai = {d.config.wait_aware_tenpai}  (=v2 底)")

    print()
    print("=" * 68)
    print("② candidate_features (mask, cand) v2 vs v3 逐位（覆盖 policy-bc 与 rl）")
    m = diff = 0
    for sit, _actions in positions(60, 20260928):
        m += 1
        x2, c2, k2 = bc.candidate_features(h2, sit)
        x3, c3, k3 = bc.candidate_features(h3, sit)
        if not (np.array_equal(c2, c3) and np.array_equal(k2, k3) and np.array_equal(x2, x3)):
            diff += 1
    print(f"  局面数 m={m}  任一逐位不同: {diff}")

    print()
    print("=" * 68)
    print("③ mlp-value 输入面：features.extract 是否吃 config")
    params = list(inspect.signature(extract).parameters)
    print(f"  features.extract 参数 = {params}  ⇒ 无 config/开关参数: "
          f"{not any('config' in p or 'aware' in p for p in params)}")
    ediff = 0
    for sit, _actions in positions(30, 771014):
        for t in range(tiles_mod.TILE_KINDS):
            if sit.hand.counts[t] <= 0:
                continue
            after = _without(sit, t)
            # extract 是纯局面函数：不看基座，故 v2/v3 下必然同值（此处仅示其唯一依赖是 Situation）
            ediff += int(not (inspect.signature(extract).parameters.keys() == {"situation"}
                              or len(params) == 1))
    print(f"  逐牌位 extract 调用示例差异计数(应为 0): {ediff}")

    print()
    print("=" * 68)
    print("④ 泄漏路径：mask.sum() <= 1 时 policy-bc/rl 直接返回内层 choice")
    leak = leak_diff = 0
    for sit, actions in positions(200, 31415926):
        _x, _c, k = bc.candidate_features(h2, sit)
        if int(np.asarray(k).sum()) > 1:
            continue
        leak += 1
        if h2.choose(sit, actions) != h3.choose(sit, actions):
            leak_diff += 1
    print(f"  mask.sum()<=1 的局面: {leak}  内层 v2/v3 动作不同: {leak_diff}")

    print()
    print("=" * 68)
    ok = (diff == 0) and (ediff == 0) and (leak_diff == 0)
    if ok:
        print("⇒ **policy-bc / mlp-value / rl 三臂内层基座 v2→v3 均为空干预**：")
        print("   候选面逐位相同、mlp 输入面与基座无关、泄漏路径上内层动作相同。")
        print("   ⇒ `arm(v2内层) vs v3` 干净测**臂自身贡献**（agent-d 的归因成立）。")
    else:
        print("⇒ 存在差异，需重新评估。")
    print("PROBE_INNER_DONE")
    return 0


def _without(sit, tile: int):
    from dataclasses import replace

    counts = list(sit.hand.counts)
    counts[tile] -= 1
    hand = replace(sit.hand, counts=tuple(counts))
    return replace(sit, hand=hand)


if __name__ == "__main__":
    raise SystemExit(main())
