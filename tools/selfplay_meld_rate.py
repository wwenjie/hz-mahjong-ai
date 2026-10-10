"""自对弈场地/档位的**副露率**（每局每座），用于挑「像真机对手那样副露」的场地臂。

**为什么需要**：`tools/meld_rate_census.py` 量到真机同批局里我方 0.623/局、三家各 1.113/局
⇒ `ab_test` 的默认场地（三座 baseline）副露率只有真机场地的 ~56%。要检验「放宽吃碰闸门」
就必须先把场地换成**副露率可比**的臂，否则测的是「对着不副露的对手放宽」——那不是真机问题。

实现：包一层决策器统计 `choose` 返回的 CHI/PENG/GANG 次数（副露动作只在响应窗口出现）。
相比读 `SeatStats`，这不需要引擎暴露中间状态，且与决策层口径一致。

用法::
    .venv/bin/python tools/selfplay_meld_rate.py --arms v7,v7m-keepchi,meld-equal --matches 4
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules.action import CHI, GANG, PENG  # noqa: E402
from majiang.sim.batch import run_match  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

MELD_KINDS = (CHI, PENG, GANG)


class CountingDecider:
    """透明包一层：只统计副露动作，不改决策。"""

    def __init__(self, inner):
        self.inner = inner
        self.melds = 0
        self.decisions = 0

    def choose(self, situation, actions, *, budget_ms: int = 0):
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        self.decisions += 1
        if choice is not None and choice.kind in MELD_KINDS:
            self.melds += 1
        return choice

    def __getattr__(self, name):  # 其余属性转发（决策器可能被其它组件内省）
        return getattr(self.inner, name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="自对弈副露率（按档位）")
    ap.add_argument("--arms", default="v7,v7m-keepchi,meld-equal")
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args(argv)

    print("座位全是同一档（自对弈），每局每座副露：")
    for arm in [name.strip() for name in args.arms.split(",") if name.strip()]:
        total_melds = 0
        total_rounds = 0
        for index in range(args.matches):
            wrapped = [CountingDecider(make_decider(arm, Mode.QUALIFIER)) for _ in range(4)]
            run_match(
                wrapped,
                rounds=args.rounds,
                base_score=1,
                seed=args.seed * 100003 + index,
                labels=[arm] * 4,
            )
            total_melds += sum(decider.melds for decider in wrapped)
            total_rounds += args.rounds * 4
        rate = total_melds / max(1, total_rounds)
        print(f"  {arm:<20} 副露 {total_melds:>6} / {total_rounds} 座·局 "
              f"⇒ {rate:.3f}/局/座")
    print("  （真机参照：三家各 1.113/局，我方 0.623/局）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
