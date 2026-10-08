#!/usr/bin/env python
"""「≤8巡到听率」与「每场名次分」相关性验证（A 2026-10-06 18:31②.1 [待C 零成本优先]）。

**目的**：A 18:31② 诊断链闭合为「早中段全线小劣势」（≤8 巡到听率差 4~10pp），但该领域手调已穷尽。
A 改推：把「累计到听率」做成候选主指标——它在自对弈里比「每场名次分」早得多、噪声小得多。
**判据（照抄 A）**：若相关显著 ⇒ 用它做快速迭代主指标（墙钟 4h → 几分钟）；若不相关 ⇒ 记进口径。

**设计**：
- 自对弈引擎（sim.batch.run_round + observer 钩子），我方启发式占 seat0，其余 3 座 baseline；
- 每场 8 局，逐巡记录我方（seat0）是否 ≤8 巡到听 ⇒ 每场到听率（0~1）；
- 每场名次分 = place_points（run_batch 的 SeatStats.game_place_points 口径）；
- 多个 arm（heuristic / final / qualifier 等）× 多个 seed，配对算 Pearson + Spearman 相关。

**产物**：`agent/out/tenpai-rate-correlation.txt` + 逐场明细 jsonl。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.sim import round as round_mod  # noqa: E402
from majiang.sim.batch import next_dealer, place_points_of  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

EARLY_CAP = 8  # ≤8 巡到听
SEATS = 4


class TenpaiObserver:
    """observer 钩子：记录我方（my_seat）首次到听巡目。

    注意（19:01 实测）：`play_round` 的 `notify()` 在 `state.turn = seat_to_move` **之前**调用，
    observer 里 `state.turn` 是上一个行动的座位，不能用来判断「谁刚摸牌」。
    改用「我方手牌数增加」检测我方摸牌事件（摸牌 n+1 / 打出 n−1 / 副露 n−2 or −3）。
    """

    def __init__(self, my_seat: int = 0):
        self.my_seat = my_seat
        self.first_tenpai_turn: int | None = None
        self.draws = 0
        self._last_n = 0

    def reset(self):
        self.first_tenpai_turn = None
        self.draws = 0
        self._last_n = 0

    def __call__(self, state):
        ss = state.seats[self.my_seat]
        n = sum(ss.hand)
        if n > self._last_n:  # 我方刚摸牌（手牌数增加）
            self.draws += 1
            if self.first_tenpai_turn is None:
                try:
                    sh = shanten_mod.shanten_any(list(ss.hand), len(ss.melds))
                except Exception:  # noqa: BLE001
                    self._last_n = n
                    return
                if sh == 0:
                    self.first_tenpai_turn = self.draws
        self._last_n = n


def run_one_match(mix: list[str], seed: int, rounds: int = 8):
    """跑一场（4 座各用 mix 里的策略），返回 (seat0 每场名次分, seat0 ≤8巡到听率)。

    mix 4 个策略互打（不再 1v3 baseline），让座位间有真实强度差 ⇒ 名次分有方差。
    """
    deciders = []
    for name in mix:
        mode = {"heuristic": Mode.QUALIFIER, "final": Mode.FINAL, "qualifier": Mode.QUALIFIER}.get(name)
        try:
            deciders.append(make_decider("heuristic", mode) if mode is not None else make_decider(name, Mode.QUALIFIER))
        except Exception:  # noqa: BLE001
            return None
    import random
    rng = random.Random(seed)
    dealer = 0
    total_scores = [0] * SEATS
    early_tenpai_rounds = 0
    obs = TenpaiObserver(0)
    for rno in range(1, rounds + 1):
        obs.reset()
        outcome = round_mod.run_round(
            deciders, dealer=dealer, round_no=rno, base_score=1,
            rng=rng, observer=obs,
        )
        for s in range(SEATS):
            total_scores[s] += outcome.scores[s]
        dealer = next_dealer(dealer, outcome)
        if obs.first_tenpai_turn is not None and obs.first_tenpai_turn <= EARLY_CAP:
            early_tenpai_rounds += 1
    pts = place_points_of(total_scores)
    my_place = pts[0]
    tenpai_rate = early_tenpai_rounds / rounds
    return my_place, tenpai_rate


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def main() -> int:
    # 我方变体互打：seat0=被测 arm，其余 3 座=对照 arm。多组配对。
    # 默认：v5-piao05 / v5-piao12 / v5-piao13 / heuristic 四个互打（在跑 A/B 的变体）。
    mixes_raw = sys.argv[1] if len(sys.argv) > 1 else "v5-piao05,v5-piao12,v5-piao13,heuristic"
    matches = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    mix = [s.strip() for s in mixes_raw.split(",")]
    if len(mix) != 4:
        raise SystemExit("mix 需恰好 4 个策略名")
    print(f"mix={mix}  {matches} 场 × 8 局", flush=True)

    rows = []  # (seat0_arm, seed, place_points, tenpai_rate)
    for i in range(matches):
        seed = 20261006 + i
        r = run_one_match(mix, seed)
        if r is None:
            print(f"  seed={seed} 构造失败，跳过", flush=True)
            continue
        rows.append((mix[0], seed, r[0], r[1]))
    print(f"  完成 {len(rows)} 场", flush=True)

    # 落明细
    det = ROOT / "agent/out/tenpai-rate-correlation.jsonl"
    with det.open("w", encoding="utf-8") as fh:
        for arm, seed, pp, tr in rows:
            fh.write(json.dumps({"arm": arm, "seed": seed, "place_points": pp, "tenpai_rate_le8": tr}) + "\n")

    xs = [r[3] for r in rows]  # tenpai_rate
    ys = [r[2] for r in rows]  # place_points
    p = pearson(xs, ys)
    s = spearman(xs, ys)

    lines = []
    lines.append(f"「≤8巡到听率」与「每场名次分」相关性（mix={mix}，共 {len(rows)} 场，seat0={mix[0]}）")
    lines.append("")
    lines.append(f"Pearson  r = {p:+.3f}" if p is not None else "Pearson 无法算（方差为0或样本<3）")
    lines.append(f"Spearman ρ = {s:+.3f}" if s is not None else "Spearman 无法算")
    lines.append("")
    lines.append("判据（A 18:31②.1）：|r| 显著（如 |r|≥0.3 且方向一致）⇒ 可用作主指标；≈0 ⇒ 脱钩记进口径")
    lines.append("")
    lines.append(f"seat0={mix[0]}：n={len(rows)}  ≤8巡到听率 mean={statistics.mean(xs):.1%}（{min(xs):.0%}~{max(xs):.0%}）  名次分 mean={statistics.mean(ys):+.2f}（{min(ys)}~{max(ys)}）")
    text = "\n".join(lines)
    print(text)
    (ROOT / "agent/out/tenpai-rate-correlation.txt").write_text(text, encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
