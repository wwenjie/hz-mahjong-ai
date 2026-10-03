"""批量自对弈与指标统计（tasks.md 6.3）。

指标口径与平台一致：

- **总得分**：本阶段每局局分累计（流局 0 分）
- **名次分**：每场按本场总得分排 1–4 位给 ``+3 / +1 / −1 / −3``，同分共享并列区间平均
  （如 1、2 名并列各 +2），不进总得分
- **白板获取数**：配牌到手 + 过程中摸到（含杠上摸）
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from majiang.rules import table as table_rules

from .round import SEATS, RoundResult, run_round

PLACE_POINTS = (3, 1, -1, -3)


@dataclass
class SeatStats:
    seat: int
    label: str = ""
    total_score: int = 0
    place_points: int = 0
    # **平台口径**的名次分：按**整场**（默认 8 局）累计总得分给四家排 +3/+1/−1/−3。
    #
    # 与 `place_points` 的区别是致命的：`place_points` 是**每局**名次之和（`run_match` 里逐局累加），
    # 而平台的战绩是**每场按累计分排名**（`client/collector` 的 `our_rank` 就是按每场累计分算的）。
    # 2026-10-03 发现这个口径错位时，正好解释了一个反直觉读数：`piao13` 让**每局**名次分 +0.057(t4.99)
    # 同时让**累计**总得分 −0.886(t−6.89)——单局名次与累计分本来就不单调。
    # 新增字段**不动** `place_points`：后者是全部历史对照的基准，改了会让历史实验不可比。
    game_place_points: int = 0
    god_count: int = 0
    rounds: int = 0
    wins: int = 0
    fan_total: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.rounds if self.rounds else 0.0

    @property
    def average_score(self) -> float:
        return self.total_score / self.rounds if self.rounds else 0.0

    @property
    def average_fan(self) -> float:
        return self.fan_total / self.wins if self.wins else 0.0


@dataclass
class BatchResult:
    seats: list[SeatStats]
    rounds: int = 0
    flows: int = 0
    details: Counter = field(default_factory=Counter)
    matches: int = 0

    @property
    def flow_rate(self) -> float:
        return self.flows / self.rounds if self.rounds else 0.0

    def row(self) -> str:
        return " | ".join(
            f"{stat.label or stat.seat}: 分={stat.total_score} 名次分={stat.place_points} "
            f"白板={stat.god_count} 胡率={stat.win_rate:.0%} 均番={stat.average_fan:.1f}"
            for stat in self.seats
        )


def place_points_of(scores: Sequence[int]) -> tuple[int, ...]:
    """按本场总得分给出名次分；同分共享并列区间的平均。

    并列区间必然连续，而 ``PLACE_POINTS`` 的三项连续和都恰好整除，因此结果恒为整数
    （如 1、2 名并列各 +2，2、3、4 名并列各 −1）。
    """
    order = sorted(range(len(scores)), key=lambda seat: scores[seat], reverse=True)
    points = [0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        shared = sum(PLACE_POINTS[index : end + 1]) / (end - index + 1)
        for position in range(index, end + 1):
            points[order[position]] = int(shared)
        index = end + 1
    return tuple(points)


def next_dealer(dealer: int, result: RoundResult) -> int:
    """流局或庄家自摸连庄，其余轮庄（与规则引擎同一约定）。"""
    return table_rules.next_dealer_seat(
        dealer, exhausted=result.is_flow, dealer_won=result.winner == dealer
    )


def run_match(
    deciders: Sequence[object],
    *,
    rounds: int = 8,
    base_score: int = 1,
    seed: int = 0,
    labels: Sequence[str] = (),
    start_dealer: int = 0,
) -> BatchResult:
    """跑一场（默认 8 局），产出一场的统计与名次分。"""
    rng = random.Random(seed)
    stats = [
        SeatStats(seat=seat, label=labels[seat] if seat < len(labels) else str(seat))
        for seat in range(SEATS)
    ]
    result = BatchResult(seats=stats, matches=1)
    dealer = start_dealer
    # 局况接线（2026-09-30）：逐局把「开局时的累计比分」注入 RoundState，让决策器
    # 在自对弈里也能看到 standings（此前只有真机路径有，自对弈恒空 ⇒ 局况臂测成空干预）。
    # ``cumulative[seat]`` 是本局**开局前**的累计净分；流局四家 delta 全 0，自然不改变累计。
    cumulative = [0] * SEATS
    for index in range(rounds):
        outcome = run_round(
            deciders,
            dealer=dealer,
            round_no=index + 1,
            base_score=base_score,
            rng=rng,
            prior_scores=tuple(cumulative),
            rounds_total=rounds,
        )
        result.rounds += 1
        result.flows += outcome.is_flow
        if not outcome.is_flow:
            result.details["+".join(outcome.detail) or "?"] += 1
        for seat in range(SEATS):
            stats[seat].total_score += outcome.scores[seat]
            cumulative[seat] += outcome.scores[seat]
            stats[seat].god_count += outcome.god_counts[seat]
            stats[seat].rounds += 1
            if not outcome.is_flow and outcome.winner == seat:
                stats[seat].wins += 1
                stats[seat].fan_total += outcome.fan
        for seat, points in enumerate(place_points_of(outcome.scores)):
            stats[seat].place_points += points
        dealer = next_dealer(dealer, outcome)
    # 每场一次：按**本场累计总得分**排名（平台口径，见 `SeatStats.game_place_points` 的说明）。
    for seat, points in enumerate(
        place_points_of([stat.total_score for stat in stats])
    ):
        stats[seat].game_place_points += points
    return result


def run_batch(
    factories: Sequence[Callable[[], object]],
    *,
    matches: int = 20,
    rounds: int = 8,
    base_score: int = 1,
    seed: int = 0,
    labels: Sequence[str] = (),
    dealer_rotation: bool = True,
) -> BatchResult:
    """跑多场并汇总。

    ``dealer_rotation`` 为真时每场轮换初始庄家，避免庄闲座次偏差污染对比。
    """
    stats = [
        SeatStats(seat=seat, label=labels[seat] if seat < len(labels) else str(seat))
        for seat in range(SEATS)
    ]
    total = BatchResult(seats=stats, matches=matches)
    for index in range(matches):
        deciders = [factory() for factory in factories]
        result = run_match(
            deciders,
            rounds=rounds,
            base_score=base_score,
            seed=seed * 100003 + index,
            labels=labels,
            start_dealer=index % SEATS if dealer_rotation else 0,
        )
        total.rounds += result.rounds
        total.flows += result.flows
        total.details.update(result.details)
        for seat in range(SEATS):
            stats[seat].total_score += result.seats[seat].total_score
            stats[seat].place_points += result.seats[seat].place_points
            stats[seat].game_place_points += result.seats[seat].game_place_points
            stats[seat].god_count += result.seats[seat].god_count
            stats[seat].rounds += result.seats[seat].rounds
            stats[seat].wins += result.seats[seat].wins
            stats[seat].fan_total += result.seats[seat].fan_total
    return total


def summary_lines(result: BatchResult) -> list[str]:
    lines = [
        f"场数 {result.matches}  局数 {result.rounds}  流局率 {result.flow_rate:.1%}",
        f"番型分布 top: {dict(result.details.most_common(6))}",
    ]
    for stat in result.seats:
        lines.append(
            f"  座位{stat.seat} {stat.label:>12s} 总得分 {stat.total_score:>8d}  "
            f"名次分 {stat.place_points:>5d}  白板 {stat.god_count:>5d}  "
            f"胡率 {stat.win_rate:>5.1%}  均分 {stat.average_score:>7.1f}  均番 {stat.average_fan:>4.1f}"
        )
    return lines


__all__ = [
    "BatchResult",
    "PLACE_POINTS",
    "SeatStats",
    "next_dealer",
    "place_points_of",
    "run_batch",
    "run_match",
    "summary_lines",
]
