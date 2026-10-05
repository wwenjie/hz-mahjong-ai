"""自对弈副露率（EQUAL 臂第一步：量「闸门放开」到底有没有动副露）。

**为什么需要它**：``analyze_meld_gate.py`` 只能告诉我们**真机**上被闸门拒了多少窗口
（93%），但拒得多不等于放开后就会副露——放开后还要过「吃了之后向听/听口是否可接受」
这一层。本工具在**自对弈**里直接量：把某个决策器放满四座（镜像对局），
逐局数每座实际持有几组副露、胡率多少。

用法::

    uv run python tools/selfplay_meld_rate.py --deciders v6,meld-equal,meld-equal-early --matches 60

口径：**持有**（局末 ``len(melds)``，补杠升级不重复计），与 ``tools/meld_census.py`` 一致；
分母是「座位·局」（matches × rounds × 4）。

**列名口径提醒**：本变体是**只自摸**且有财神，自对弈实测流局率 **0.3%~1.9%**
（四座合计胡率接近 100%，所以「任一家胡率」≈ 1 − 流局率，信息量为零）⇒
这里只报 **非流局率**，不报「胡率」以免被误读成单座胜率。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from majiang.cli import DECIDERS, make_decider  # noqa: E402
from majiang.rules import shanten, tiles, win  # noqa: E402
from majiang.rules.action import DISCARD  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim.round import SEATS, run_round  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode  # noqa: E402


def build(name: str, mode: Mode):
    """策略名或版本号（`v3`…`v6`）→ 决策器工厂。"""
    if name in DECIDERS:
        return lambda: make_decider(name, mode)
    if versions.is_version(name):
        return lambda: versions.build(name, mode)
    raise SystemExit(f"未知策略 {name!r}（见 majiang.cli.DECIDERS 与 strategy/versions.py）")


class WaitWidthProbe:
    """给任意决策器**外包一层**，统计「持财神听口种数」（G1 机制门①的量）。

    **为什么由本工具承担**：B' 的财神专项读的是**真机**（bot 6.08 vs 我们 4.53、+34%），
    但 G1 要跑的是**自对弈 A/B**——机制门①（`godwait/v5` 的持财神听口种数比 ≥1.25）**必须能在自对弈里量**，
    否则就会变成「只看胜负、不看机制」，那正是 `v6-equal` 那次的教训（副露 +48% 却输）。

    **口径**：只在「**出牌后**手牌含财神 **且** 已听牌」的决策点记录 `len(winning_draws)`，
    与 S2/财神专项的「听口种数」同义；未听牌或手里无财神不计入（否则把「到听前」混进来）。
    """

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.god_tenpai_widths: list[int] = []
        self.turns = 0                 # 该座做过决策的次数（只看摸牌相 ⇒ 近似巡目）
        self.tenpai_turn: int | None = None
        self.tenpai_with_god = False

    def choose(self, situation, actions, *, budget_ms: int = 1800):
        if situation.phase == PHASE_DRAW:
            self.turns += 1
            if self.tenpai_turn is None:
                hand_counts = situation.hand.counts
                if shanten.shanten_any(hand_counts, situation.hand.meld_count) == 0:
                    # **到听速度（财神桶）**：这是门①的分母——「有财神桶的到听速度」。
                    # 按该座**到听那一刻手里是否持财神**分桶（不是按全局财神数），
                    # 因为要检验的正是「持财神时我们更晚听」这条（C 19:32 的分歧对拍结论）。
                    self.tenpai_turn = self.turns
                    self.tenpai_with_god = hand_counts[tiles.GOD] >= 1
        decision = self.inner.choose(situation, actions, budget_ms=budget_ms)  # type: ignore[attr-defined]
        try:
            if decision is not None and decision.kind == DISCARD and decision.tile is not None:
                counts = list(situation.hand.counts)
                if counts[decision.tile] > 0:
                    counts[decision.tile] -= 1
                    if counts[tiles.GOD] >= 1:
                        if shanten.shanten_any(counts, situation.hand.meld_count) == 0:
                            self.god_tenpai_widths.append(
                                len(win.winning_draws(counts, situation.hand.meld_count))
                            )
        except Exception:  # noqa: BLE001 —— 探针绝不能因为计数失败影响牌局
            pass
        return decision


def run_one(name: str, matches: int, rounds: int, seed: int, mode: Mode) -> dict:
    factory = build(name, mode)
    rng = random.Random(seed)
    melds = wins = flows = 0
    pinghu = baotou = 0
    fan_total = 0
    widths: list[int] = []
    tenpai_god: list[int] = []
    tenpai_nogod: list[int] = []
    for index in range(matches):
        dealer = index % SEATS
        seat_deciders = [WaitWidthProbe(factory()) for _ in range(SEATS)]
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
                fan_total += result.fan
                # **番型分布**（门④）：S2（`agent/out/s2-fan-gap.txt`）说我们的缺口 87% 是「率差」，
                # 具体是**平胡率**（bot 0.284/局 vs 我们 0.202）与**爆头率**（0.088 vs 0.029，只有 1/3）。
                # 这两项必须在**同一批自对弈**里量，否则「v7m 拉开副露后平胡率动没动」无从判读。
                for label in result.detail:
                    if "平胡" in label:
                        pinghu += 1
                    if "爆头" in label:
                        baotou += 1
            if not (result.is_flow or result.winner == dealer):
                dealer = (dealer + 1) % SEATS
        for probe in seat_deciders:
            widths.extend(probe.god_tenpai_widths)
            if probe.tenpai_turn is not None:
                if probe.tenpai_with_god:
                    tenpai_god.append(probe.tenpai_turn)
                else:
                    tenpai_nogod.append(probe.tenpai_turn)
    games = matches * rounds
    per = games * SEATS
    return {
        "name": name,
        "matches": matches,
        "rounds": games,
        "meld_per_seat_round": melds / per if per else 0.0,
        "meld_per_round": melds / games if games else 0.0,
        "non_flow_rate": wins / games if games else 0.0,
        "flow_rate": flows / games if games else 0.0,
        # 与 S2 同口径：单座视角「每局次数」= 总次数 / (局数 × 4)
        "pinghu_per_seat_round": pinghu / per if per else 0.0,
        "baotou_per_seat_round": baotou / per if per else 0.0,
        "average_fan": fan_total / wins if wins else 0.0,
        "god_tenpai_width": sum(widths) / len(widths) if widths else 0.0,
        "god_tenpai_points": len(widths),
        "tenpai_turn_god": sum(tenpai_god) / len(tenpai_god) if tenpai_god else 0.0,
        "tenpai_turn_nogod": sum(tenpai_nogod) / len(tenpai_nogod) if tenpai_nogod else 0.0,
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
    print(
        f"{'策略':>18} {'副露/座·局':>10} {'平胡/座·局':>10} {'爆头/座·局':>10} "
        f"{'均番':>6} {'持财神听口种数':>13} {'到听巡_有财':>10} {'到听巡_无财':>10} {'点数':>7}  （{args.matches}场×{args.rounds}局，镜像四座）"
    )
    for name in [n.strip() for n in args.deciders.split(",") if n.strip()]:
        row = run_one(name, args.matches, args.rounds, args.seed, mode)
        print(
            f"{row['name']:>18} {row['meld_per_seat_round']:>10.3f} {row['pinghu_per_seat_round']:>10.3f} "
            f"{row['baotou_per_seat_round']:>10.3f} {row['average_fan']:>6.3f} "
            f"{row['god_tenpai_width']:>13.3f} {row['tenpai_turn_god']:>9.2f} "
            f"{row['tenpai_turn_nogod']:>9.2f} {row['god_tenpai_points']:>7d}",
            flush=True,
        )
    print(
        "  （参考：头部 bot 真机 副露 1.251 / 平胡 0.284 / 爆头 0.088 / 均番 1.422 / 持财神听口种数 6.08；"
        "我们 v5 真机 0.623 / 0.202 / 0.029 / 1.242 / 4.53。自对弈绝对值系统性偏高，看**臂间差**与**倍数**。）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
