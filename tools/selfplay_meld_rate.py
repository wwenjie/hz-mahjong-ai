"""自对弈副露率（EQUAL 臂第一步：量「闸门放开」到底有没有动副露）。

**为什么需要它**：``analyze_meld_gate.py`` 只能告诉我们**真机**上被闸门拒了多少窗口
（93%），但拒得多不等于放开后就会副露——放开后还要过「吃了之后向听/听口是否可接受」
这一层。本工具在**自对弈**里直接量：把某个决策器放满四座（镜像对局），
逐局数每座实际持有几组副露、胡率多少。

用法::

    uv run python tools/selfplay_meld_rate.py --deciders v6,meld-equal,meld-equal-early --matches 60

口径：**持有**（局末 ``len(melds)``，补杠升级不重复计），与 ``tools/meld_census.py`` 一致；
分母是「座位·局」（matches × rounds × 4）。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from majiang.cli import DECIDERS, make_decider  # noqa: E402
from majiang.sim.round import SEATS, run_round  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402


def build(name: str, mode: Mode):
    """策略名或版本号（`v3`…`v6`）→ 决策器工厂。"""
    if name in DECIDERS:
        return lambda: make_decider(name, mode)
    if versions.is_version(name):
        return lambda: versions.build(name, mode)
    raise SystemExit(f"未知策略 {name!r}（见 majiang.cli.DECIDERS 与 strategy/versions.py）")


def run_one(name: str, matches: int, rounds: int, seed: int, mode: Mode) -> dict:
    factory = build(name, mode)
    rng = random.Random(seed)
    melds = wins = flows = 0
    for index in range(matches):
        dealer = index % SEATS
        seat_deciders = [factory() for _ in range(SEATS)]
        for _ in range(rounds):
            seen: dict = {}

            def observer(state, seen=seen):  # RoundState 原地修改，持引用即可
                seen["state"] = state

            result = run_round(
                seat_deciders,
                dealer=dealer,
                round_no=1,
                base_score=1,
                rng=rng,
                observer=observer,
            )
            state = seen.get("state")
            if state is not None:
                melds += sum(len(seat_state.melds) for seat_state in state.seats)
            if result.is_flow:
                flows += 1
            else:
                wins += 1
            if not (result.is_flow or result.winner == dealer):
                dealer = (dealer + 1) % SEATS
    per = matches * rounds * SEATS
    return {
        "name": name,
        "matches": matches,
        "rounds": matches * rounds,
        "meld_per_seat_round": melds / per if per else 0.0,
        "meld_per_round": melds / (matches * rounds) if matches * rounds else 0.0,
        "win_rate": wins / (matches * rounds) if matches * rounds else 0.0,
        "flow_rate": flows / (matches * rounds) if matches * rounds else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="自对弈副露率（镜像四座）")
    ap.add_argument("--deciders", required=True, help="逗号分隔的策略名")
    ap.add_argument("--matches", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--mode", default="qualifier", choices=[m.value for m in Mode])
    args = ap.parse_args(argv)
    mode = Mode(args.mode)
    print(f"{'策略':>18} {'副露/座·局':>10} {'副露/局':>8} {'胡率':>7} {'流局率':>7}  （{args.matches}场×{args.rounds}局）")
    for name in [n.strip() for n in args.deciders.split(",") if n.strip()]:
        row = run_one(name, args.matches, args.rounds, args.seed, mode)
        print(
            f"{row['name']:>18} {row['meld_per_seat_round']:>10.3f} {row['meld_per_round']:>8.3f} "
            f"{row['win_rate']:>6.1%} {row['flow_rate']:>6.1%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
