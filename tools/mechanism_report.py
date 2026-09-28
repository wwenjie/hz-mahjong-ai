"""机制报告：自对弈下**分档测量「爆头」这条轴**，不只看总得分。

**为什么需要它**：agent B 的分解显示，我们与对手的均番差（1.269 vs 1.316）几乎全部来自
**爆头占胡比 14.9% vs 22.7%**——这是目前唯一被量化出来的结构性缺口。而总得分是重尾分布，
自对弈噪声底只够做**方向**判断；爆头率是**比率型机制指标**，功效高得多，
能在同样的算力下先回答「这个干预到底动没动那条轴」。

**口径**（必须写清楚，否则一定被误读）：

- 爆头判定用 ``RoundResult.detail`` 里是否含「爆头」标签（与 ``rules/fan.py`` 同源），
  不用自己重算番型。
- ``胜时财神数`` 取 ``RoundResult.god_counts[winner]``——**是局末的四家持牌财神数**，
  不是「胡牌时的持牌数」。两者对赢家相同（赢家不会再动牌），对别人不同。
- **四座位旋转**：每个档位依次坐 0/1/2/3 座，另三座固定为 ``--field``。
  否则庄家的 ×8 赔付与座次时序会把差异混进来。
- 与 ``tools/ab_test.py`` 的区别：那个做**逐场配对差分**给总得分；这个做**比率聚合**
  给机制。两者互补，不要互相替代。

用法::

    uv run python tools/mechanism_report.py --arms heuristic,preserve-god,natural
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from majiang.cli import DECIDERS, make_decider  # noqa: E402
from majiang.sim.batch import SEATS, next_dealer  # noqa: E402
from majiang.sim.round import run_round  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

TALLY = {"heuristic": Mode.QUALIFIER, "final": Mode.FINAL, "qualifier": Mode.QUALIFIER}
ALIASES = {"baseline": "first-legal"}
BAOTOU = "爆头"


def build(name: str):
    if name in TALLY:
        return make_decider("heuristic", TALLY[name])
    resolved = ALIASES.get(name, name)
    if resolved not in DECIDERS:
        raise SystemExit(f"未知策略 {name!r}")
    return make_decider(resolved, Mode.QUALIFIER)


@dataclass
class Arm:
    """一个档位在一个座位上的累计量。"""

    rounds: int = 0
    flows: int = 0
    wins: int = 0
    baotou: int = 0
    god_at_win: int = 0
    fan_total: int = 0

    def merge(self, other: Arm) -> None:
        self.rounds += other.rounds
        self.flows += other.flows
        self.wins += other.wins
        self.baotou += other.baotou
        self.god_at_win += other.god_at_win
        self.fan_total += other.fan_total

    @property
    def win_rate(self) -> float:
        return self.wins / self.rounds if self.rounds else 0.0

    @property
    def baotou_of_wins(self) -> float:
        return self.baotou / self.wins if self.wins else 0.0

    @property
    def mean_fan(self) -> float:
        return self.fan_total / self.wins if self.wins else 0.0

    @property
    def god_at_win_mean(self) -> float:
        return self.god_at_win / self.wins if self.wins else 0.0


def measure(
    arm_name: str,
    field: str,
    *,
    seat: int,
    matches: int,
    rounds: int,
    seed: int,
    base_score: int,
) -> Arm:
    """让 ``arm_name`` 坐 ``seat`` 座跑 ``matches`` 场，另三座为 ``field``。"""
    total = Arm()
    labels = [field] * SEATS
    labels[seat] = arm_name
    for index in range(matches):
        deciders = [build(name) for name in labels]
        # 与 ``sim.batch.run_match`` 的种子链一致，保证同 index 的场次拿到同一副牌
        rng = random.Random(seed * 100003 + index)
        dealer = index % SEATS
        for round_no in range(1, rounds + 1):
            outcome = run_round(
                deciders, dealer=dealer, round_no=round_no, base_score=base_score, rng=rng
            )
            total.rounds += 1
            if outcome.is_flow:
                total.flows += 1
            else:
                # 机制量只统计本座位，且分母用**全部局数**（含流局）——
                # 用「非流局数」做分母会把流局率的变化混进胡率里。
                if outcome.winner == seat:
                    total.wins += 1
                    total.fan_total += outcome.fan
                    total.god_at_win += outcome.god_counts[seat]
                    if BAOTOU in outcome.detail:
                        total.baotou += 1
            dealer = next_dealer(dealer, outcome)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="自对弈机制报告（爆头轴）")
    parser.add_argument("--arms", default="heuristic,preserve-god,natural")
    parser.add_argument("--field", default="heuristic", help="另三座坐谁")
    parser.add_argument("--matches", type=int, default=12, help="每个座位跑几场")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--base-score", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    table: dict[str, Arm] = {}
    started = time.perf_counter()
    for arm_name in arms:
        merged = Arm()
        for seat in range(SEATS):
            part = measure(
                arm_name, args.field, seat=seat, matches=args.matches,
                rounds=args.rounds, seed=args.seed, base_score=args.base_score,
            )
            merged.merge(part)
            print(
                f"  {arm_name:14s} 座{seat}  局{part.rounds:5d}  胡{part.wins:4d}"
                f"  爆头{part.baotou:3d}  爆头/胡 {part.baotou_of_wins:6.1%}"
                f"  均番 {part.mean_fan:5.2f}",
                flush=True,
            )
        table[arm_name] = merged

    elapsed = time.perf_counter() - started
    print(f"\n=== 汇总（每档 {SEATS} 旋转 × {args.matches} 场 × {args.rounds} 局，"
          f"另三座={args.field}，用时 {elapsed:.0f}s）===")
    print(f"{'档位':14s} {'局数':>6s} {'流局':>6s} {'胡率':>7s} {'爆头/胡':>8s}"
          f" {'爆头/局':>8s} {'均番':>6s} {'胜时财神':>8s}")
    for arm_name in arms:
        arm = table[arm_name]
        print(
            f"{arm_name:14s} {arm.rounds:6d} {arm.flows / arm.rounds:6.1%}"
            f" {arm.win_rate:7.2%} {arm.baotou_of_wins:8.1%}"
            f" {arm.baotou / arm.rounds:8.1%} {arm.mean_fan:6.2f}"
            f" {arm.god_at_win_mean:8.2f}"
        )
    print("\n读法：均番差要落在「爆头/胡」这一列上才算打中 B 定位的那条轴；"
          "若只有「胜时财神」动而「爆头/胡」不动，说明财神只是留着、没变成爆头。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
