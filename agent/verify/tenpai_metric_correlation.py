#!/usr/bin/env python
"""「累计到听率能否当主指标」验证（A 2026-10-06 18:55 口径，零 A/B 成本）。

**口径（照抄 A 18:55 ②）**：
- 自变量：对**每个臂**跑自对弈镜像（该臂占 4 座），量 **≤8 巡累计到听率**（分母=座位·局）；
- 因变量：该臂**已有的 `每场名次分`**（`notes/experiments.json` done 结果，同臂多种子取均值）；
- 样本：带 `每场名次分` 的臂（A 数了 9 个）⇒ n≈9 点；Pearson/Spearman + 散点；
- **预登记判据**：|r|≥0.6 且 p<0.05 ⇒ 可用主指标；|r|<0.4 ⇒ 脱钩；中间带 ⇒ 换窗宽加样本；
- **混淆控制（A ④）**：臂非随机样本，须报「按 mtime/field 分层后的相关」。

**镜像场口径**：4 座同臂（self-play mirror），≤8 巡到听率 = 每局四座位 ≤8 巡到听的占比均值
（分母=座位·局=4×8×场数）。observer 钩子记每座位首次到听巡目（用「手牌数增加」检测摸牌）。

产物：`agent/out/tenpai-rate-metric-correlation.txt` + `.jsonl`。
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
from majiang.sim.batch import next_dealer  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

EARLY_CAP = 8
SEATS = 4
MATCHES_PER_ARM = 30  # 每臂镜像场数（A 估 ~3min/臂）


class MirrorTenpaiObserver:
    """镜像场 observer：记录**每个座位**的首次到听巡目（手牌数增加=该座刚摸牌）。"""

    def __init__(self):
        self.first_tenpai: dict[int, int | None] = {s: None for s in range(SEATS)}
        self.draws: dict[int, int] = {s: 0 for s in range(SEATS)}
        self._last_n: dict[int, int] = {s: 0 for s in range(SEATS)}

    def reset(self):
        self.first_tenpai = {s: None for s in range(SEATS)}
        self.draws = {s: 0 for s in range(SEATS)}
        self._last_n = {s: 0 for s in range(SEATS)}

    def __call__(self, state):
        for seat in range(SEATS):
            ss = state.seats[seat]
            n = sum(ss.hand)
            if n > self._last_n[seat]:  # 该座刚摸牌
                self.draws[seat] += 1
                if self.first_tenpai[seat] is None:
                    try:
                        sh = shanten_mod.shanten_any(list(ss.hand), len(ss.melds))
                    except Exception:  # noqa: BLE001
                        self._last_n[seat] = n
                        continue
                    if sh == 0:
                        self.first_tenpai[seat] = self.draws[seat]
            self._last_n[seat] = n


def arm_tenpai_rate(arm: str, matches: int, rounds: int = 8) -> float | None:
    """该臂镜像场（4 座同臂）的 ≤8 巡累计到听率（分母=座位·局）。"""
    mode = {"heuristic": Mode.QUALIFIER, "final": Mode.FINAL, "qualifier": Mode.QUALIFIER}.get(arm)
    try:
        deciders = [make_decider("heuristic", mode) if mode is not None else make_decider(arm, Mode.QUALIFIER)
                    for _ in range(SEATS)]
    except Exception as e:  # noqa: BLE001
        print(f"    {arm} 构造失败: {e}", flush=True)
        return None
    obs = MirrorTenpaiObserver()
    hit = 0
    total = 0
    import random
    for m in range(matches):
        rng = random.Random(20261006 + m)
        dealer = 0
        for rno in range(1, rounds + 1):
            obs.reset()
            outcome = round_mod.run_round(
                deciders, dealer=dealer, round_no=rno, base_score=1,
                rng=rng, observer=obs,
            )
            dealer = next_dealer(dealer, outcome)
            for seat in range(SEATS):
                total += 1
                ft = obs.first_tenpai[seat]
                if ft is not None and ft <= EARLY_CAP:
                    hit += 1
    return hit / total if total else None


def existing_place_points() -> dict[str, float]:
    """从 experiments.json 取各臂已有每场名次分（同臂多种子取均值）。"""
    d = json.loads((ROOT / "notes/experiments.json").read_text(encoding="utf-8"))
    jobs = d if isinstance(d, list) else d.get("jobs", [])
    arms: dict[str, list[float]] = {}
    for j in jobs:
        if j.get("status") != "done":
            continue
        gp = (j.get("result", {}).get("metrics", {}).get("每场名次分") or {}).get("mean")
        if gp is None:
            continue
        arms.setdefault(j["treatment"], []).append(gp)
    return {t: sum(v) / len(v) for t, v in arms.items()}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None, None
    r = cov / (sx * sy)
    # t 检验 p 值（双尾，df=n-2）
    import math
    t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
    # 近似 p（t 分布，用正态近似在大样本下可；n=9 用 t 临界值粗判）
    # df=7: t=2.365 → p=0.05; t=3.499 → p=0.01
    df = n - 2
    crit = {0.05: 2.365, 0.01: 3.499} if df == 7 else {0.05: 1.96, 0.01: 2.576}
    p_sig = "p<0.01" if abs(t) > crit[0.01] else ("p<0.05" if abs(t) > crit[0.05] else f"p>0.05 (t={t:.2f})")
    return r, p_sig


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
        return None, None
    return pearson(ranks(xs), ranks(ys))


def main() -> int:
    arms = sorted(existing_place_points().keys())
    print(f"带每场名次分的臂 = {len(arms)}: {arms}", flush=True)
    place = existing_place_points()

    rates: dict[str, float] = {}
    progress = ROOT / "agent/out/tenpai-metric-progress.jsonl"
    # 断点续跑：已完成的臂直接读进度文件
    done_arms: dict[str, float] = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done_arms[rec["arm"]] = rec["tenpai_rate_le8"]
            except Exception:  # noqa: BLE001
                pass
    for arm in arms:
        if arm in done_arms:
            rates[arm] = done_arms[arm]
            print(f"  {arm:20s} ≤8巡到听率={rates[arm]:.1%}（进度文件）", flush=True)
            continue
        r = arm_tenpai_rate(arm, MATCHES_PER_ARM)
        if r is not None:
            rates[arm] = r
            with progress.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"arm": arm, "tenpai_rate_le8": r}) + "\n")
        print(f"  {arm:20s} ≤8巡到听率={f'{r:.1%}' if r is not None else 'FAIL'}  名次分={place[arm]:+.3f}", flush=True)

    pairs = [(rates[a], place[a]) for a in arms if a in rates]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    r_p, sig_p = pearson(xs, ys)
    r_s, sig_s = spearman(xs, ys)

    # 落明细
    det = ROOT / "agent/out/tenpai-rate-metric-correlation.jsonl"
    with det.open("w", encoding="utf-8") as fh:
        for a in arms:
            if a in rates:
                fh.write(json.dumps({"arm": a, "tenpai_rate_le8": rates[a], "place_points": place[a]}) + "\n")

    lines = []
    lines.append(f"「≤8巡累计到听率」×「每场名次分」臂间相关（{len(pairs)} 臂，镜像场各 {MATCHES_PER_ARM} 场 × 8 局）")
    lines.append("")
    lines.append("臂 | ≤8巡到听率 | 每场名次分")
    lines.append("---|-----------|----------")
    for a in arms:
        if a in rates:
            lines.append(f"{a:>18} | {rates[a]:>9.1%} | {place[a]:+.3f}")
    lines.append("")
    lines.append(f"Pearson  r = {r_p:+.3f}（{sig_p}）" if r_p is not None else "Pearson 无法算")
    lines.append(f"Spearman ρ = {r_s:+.3f}（{sig_s}）" if r_s is not None else "Spearman 无法算")
    lines.append("")
    lines.append("预登记判据（A 18:55③）：|r|≥0.6 且 p<0.05 ⇒ 可用主指标；|r|<0.4 ⇒ 脱钩；0.4≤|r|<0.6 ⇒ 换窗宽加样本")
    lines.append("混淆控制（A ④）：臂非随机样本——判读时须注意；若分层后不稳定 ⇒ 判不可用")
    text = "\n".join(lines)
    print(text)
    (ROOT / "agent/out/tenpai-rate-metric-correlation.txt").write_text(text, encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
