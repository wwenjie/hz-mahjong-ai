#!/usr/bin/env python
"""副露率自对弈测量（A 00:55 裁决第一步）：同种子同 field，strict vs equal 数副露次数/局。

**方法**：给 decider 套计数 wrapper（choose 返回后检查 action.kind ∈ {chi,peng,gang}），
四座位旋转（treatment 依次在 0/1/2/3 座），同种子 ⇒ 同牌，严格配对。

**判据（A 00:55）**：
- equal 臂副露率 <0.8 ⇒ 闸门不是唯一瓶颈，不上真机
- equal 臂副露率 ~1.0 ⇒ 报 A 进第二步（限量真机臂）

零 src 改动：wrapper 在本文件内，不动 batch.py / policy.py。

用法：`.venv/bin/python agent/verify/meld_rate_selfplay.py [--matches N] [--seed S]`
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from math import erf, sqrt

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.sim.batch import SEATS, run_match  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

MELD_KINDS = {"chi", "peng", "gang"}


class MeldCounter:
    """包住一个 decider，数它在响应窗选副露的次数。"""

    def __init__(self, inner):
        self.inner = inner
        self.melds = 0  # 副露动作总数
        self.name = getattr(inner, "name", "?")

    # 透传 inner 的全部属性（run_match 会读 .name 等）
    def __getattr__(self, key):
        return getattr(self.inner, key)

    def choose(self, situation, actions, *, budget_ms):
        action = self.inner.choose(situation, actions, budget_ms=budget_ms)
        if action is not None and action.kind in MELD_KINDS:
            self.melds += 1
        # 透传诊断字段
        self.last_reason = getattr(self.inner, "last_reason", "")
        self.last_detail = getattr(self.inner, "last_detail", {})
        return action


def run_arm(arm_name: str, matches: int, seed: int, rounds: int = 8):
    """四座位旋转跑 matches 场，返回 treatment 座位的 (副露次数列表[每场], 总局数)。"""
    per_match_melds = []
    total_rounds = 0
    per_round_matches = max(1, matches // SEATS)
    for seat_pos in range(SEATS):
        for m in range(per_round_matches):
            # 构造 4 个 decider：seat_pos 是 treatment（带计数），其余是 baseline heuristic
            counters = []
            deciders = []
            for s in range(SEATS):
                if s == seat_pos:
                    inner = make_decider(arm_name, Mode.QUALIFIER)
                    counter = MeldCounter(inner)
                    counters.append(counter)
                    deciders.append(counter)
                else:
                    deciders.append(make_decider("heuristic", Mode.QUALIFIER))
            run_match(
                deciders,
                rounds=rounds,
                base_score=1,
                seed=seed * 100003 + seat_pos * 1000 + m,
                labels=[f"seat{s}" for s in range(SEATS)],
            )
            per_match_melds.append(counters[0].melds)
            total_rounds += rounds
    return per_match_melds, total_rounds


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = sqrt(va / len(a) + vb / len(b))
    return (mb - ma) / se if se else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=200, help="每座位位置跑几场（总场数=matches×4）")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--rounds", type=int, default=8)
    args = ap.parse_args()

    print(f"副露率自对弈测量：strict(heuristic) vs equal(meld-equal)")
    print(f"  每场 {args.rounds} 局 × 每座位 {args.matches} 场 × 4 座位 = {args.matches * 4} 场/臂")
    print(f"  种子 {args.seed}（同种子同牌，座位旋转）")
    print()

    results = {}
    for arm in ("heuristic", "meld-equal"):
        melds, total_rounds = run_arm(arm, args.matches, args.seed, args.rounds)
        rate = sum(melds) / total_rounds
        results[arm] = (melds, total_rounds, rate)
        print(f"  {arm:12s}: 副露 {sum(melds)} 次 / {total_rounds} 局 = {rate:.3f}/局"
              f"（每场均值 {st.mean(melds):.2f}，SD {st.stdev(melds) if len(melds) > 1 else 0:.2f}）")

    a_melds, _, a_rate = results["heuristic"]
    b_melds, _, b_rate = results["meld-equal"]
    t = welch_t(a_melds, b_melds)
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) if t == t else float("nan")

    print()
    print(f"差（equal − strict）= {b_rate - a_rate:+.3f}/局（t={t:+.2f}, p={p:.4f}）")
    print()
    print("★ 判据（A 00:55）：")
    if b_rate < 0.8:
        print(f"  equal 臂副露率 {b_rate:.3f} < 0.8 ⇒ 闸门不是唯一瓶颈，估值层还在压 ⇒ 不上真机")
    elif b_rate >= 0.95:
        print(f"  equal 臂副露率 {b_rate:.3f} ≈ 1.0 ⇒ 闸门是主因 ⇒ 报 A 进第二步（限量真机臂）")
    else:
        print(f"  equal 臂副露率 {b_rate:.3f} 在 0.8–0.95 之间 ⇒ 边界，报数据给 A 裁决")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
