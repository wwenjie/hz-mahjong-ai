"""独立复算：`search-v3` / `search-deep-v3` vs `v3`（不 import agent-d 的 nnrl.eval）。

为什么能「从原始对局重实现」
--------------------------
`run_ab.py` 只落**聚合值**（mean/se/t/n），不落逐场明细。但 agent-d 的 `paired_ab`
保证：**每场只由 `(names, index, rounds, base_score, seed)` 完全确定**，
对局种子 = `seed * 100003 + index`，`start_dealer = index % SEATS`。
⇒ 我用自己的实现重跑同一批对局，与它的聚合读数对照——这就是独立复算。

本脚本只读主仓（`src/majiang/**`），完全**不 import** `nnrl.*`。
臂构造按 agent-d 的 `nnrl/v3arms.py` 配方（等价于主仓 `search` 但内层换 v3）。

用法::

    # 仪表自检（小样本）
    nice -n 10 .venv/bin/python agent/verify/recompute_search_v3.py \
        --arms search-v3 --seeds 20260928 --matches 2 --workers 4

    # 正式复算（等链与 RL 对拍跑完、核空闲时）
    nice -n 10 .venv/bin/python agent/verify/recompute_search_v3.py \
        --arms search-v3,search-deep-v3 --seeds 20260928,771014 --matches 40 --workers 12
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

REPO = Path("/home/wuwenjie01/majiang_ai")
RL_ROOT = Path("/home/wuwenjie01/majiang_rl")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(RL_ROOT / "src"))

from majiang.cli import DECIDERS  # noqa: E402
from majiang.sim.batch import run_match  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig  # noqa: E402
from majiang.strategy.search import SearchConfig, SearchDecider  # noqa: E402
from majiang.strategy.versions import build as build_version  # noqa: E402

# agent-d 本线三臂的纯 Python 产物（只读；**不 import nnrl.eval**——测量侧保持独立）
_ARM_PAYLOADS = {
    "rl": "runs/rl-s20260928/params.json",
    "mlp-value": "runs/mlp-value-s20260928/model.json",
    "policy-bc": "runs/policy-bc-v2/model.json",
}


def _load_arm_payload(name: str) -> dict:
    return json.loads((RL_ROOT / _ARM_PAYLOADS[name]).read_text(encoding="utf-8"))

LAMBDA = 100003
SEATS = 4
LABELS = ("总得分", "名次分", "白板数", "胡次数", "番数总和")


def make(name: str):
    """按名字造决策器（自带，不 import nnrl）。"""
    if name == "search-v3":
        return SearchDecider(build_version("v3", Mode.QUALIFIER), SearchConfig(samples=6, top_k=2))
    if name == "search-deep-v3":
        return SearchDecider(build_version("v3", Mode.QUALIFIER), SearchConfig(samples=16, top_k=3))
    if name in ("v1", "v2", "v3"):
        return build_version(name, Mode.QUALIFIER)
    if name in _ARM_PAYLOADS:
        # 构造**臂本体**（不是评测器）：rl / mlp-value / policy-bc
        if name == "rl":
            from nnrl.rl_play import RLPolicy

            p = _load_arm_payload("rl")
            return RLPolicy(p.get("params", p), sample=False, seed=0, label="rl")
        if name == "mlp-value":
            from majiang.strategy.features import extract
            from nnrl.decider import MLPValueDecider

            h = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
            return MLPValueDecider(_load_arm_payload("mlp-value"), h,
                                   feature_extract=extract, label="mlp-value")
        from nnrl.policy_decider import PolicyNetDecider

        return PolicyNetDecider(_load_arm_payload("policy-bc"), label="policy-bc")
    if name in DECIDERS:
        from majiang.cli import make_decider

        return make_decider(name, Mode.QUALIFIER)
    raise SystemExit(f"未知臂 {name!r}")


def _play_task(task: tuple) -> tuple:
    """worker 入口（模块级，fork 传参）：跑一场，返回四座 5 元组。"""
    names, index, rounds, base_score, seed = task
    deciders = [make(n) for n in names]
    result = run_match(
        deciders,
        rounds=rounds,
        base_score=base_score,
        seed=seed * LAMBDA + index,
        labels=list(names),
        start_dealer=index % SEATS,
    )
    return tuple(
        (s.total_score, s.place_points, s.god_count, s.wins, s.fan_total) for s in result.seats
    )


def _names_for(rotation: int, seat_name: str, baseline: str) -> list[str]:
    row = [""] * SEATS
    row[rotation] = seat_name
    for seat, nm in zip([s for s in range(SEATS) if s != rotation], [baseline] * 3):
        row[seat] = nm
    return row


def paired_ab_par(pool, treatment: str, baseline: str, *, matches: int, rounds: int,
                  base_score: int, seed: int):
    """复刻 nnrl.eval.paired_ab 的配对逻辑（默认场地=baseline），多进程执行。"""
    base_tasks = [(tuple([baseline] * SEATS), i, rounds, base_score, seed) for i in range(matches)]
    base_res = pool.map(_play_task, base_tasks, chunksize=1)

    treat_tasks, keys = [], []
    for rotation in range(SEATS):
        names = tuple(_names_for(rotation, treatment, baseline))
        for i in range(matches):
            treat_tasks.append((names, i, rounds, base_score, seed))
            keys.append((rotation, i))
    treat_res = pool.map(_play_task, treat_tasks, chunksize=1)

    diffs: dict[str, list[float]] = {lab: [] for lab in LABELS}
    t_score = b_score = 0
    for (rotation, i), mine in zip(keys, treat_res):
        theirs = base_res[i][rotation]
        m = mine[rotation]
        for lab, a, b in zip(LABELS, m, theirs):
            diffs[lab].append(a - b)
        t_score += m[0]
        b_score += theirs[0]
    return diffs, t_score, b_score


def stat(vals: list[float]):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    return mean, se, (mean / se if se else 0.0), n


def check_rank_discards_identity(want: int = 21) -> None:
    """ⓐ `_rank_discards` 对 w/wo `wait_aware_tenpai` 是否逐点相同（听牌局面）。"""
    from majiang.rules import shanten as shanten_module
    from majiang.rules.action import legal_actions
    from majiang.rules.situation import PHASE_DRAW
    from majiang.sim import round as round_module

    c2 = build_version("v2", Mode.QUALIFIER).config
    c3 = build_version("v3", Mode.QUALIFIER).config
    d2 = SearchDecider(HeuristicDecider(c2), SearchConfig(samples=6, top_k=2))
    d3 = SearchDecider(HeuristicDecider(c3), SearchConfig(samples=6, top_k=2))
    rng = random.Random(31415926)
    found = diff = 0
    while found < want:
        st = round_module.deal(rng, dealer=rng.randrange(4))
        seat = st.turn
        round_module._draw(st, seat)  # noqa: SLF001
        sit = round_module.situation_for(st, seat, PHASE_DRAW)
        hand, mc = sit.hand.counts, sit.hand.meld_count
        best = 99
        for t in range(34):
            if hand[t] <= 0:
                continue
            after = list(hand)
            after[t] -= 1
            try:
                best = min(best, shanten_module.shanten_any(after, mc))
            except Exception:  # noqa: BLE001
                continue
        if best != 0:
            continue
        found += 1
        actions = legal_actions(sit)
        if [(x.tile, x.total) for x in d2._rank_discards(sit, actions)] != \
           [(x.tile, x.total) for x in d3._rank_discards(sit, actions)]:
            diff += 1
    print(f"ⓐ `_rank_discards` 听牌局面 n={found}  w/wo wait_aware 有差异: {diff}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="search-v3,search-deep-v3")
    ap.add_argument("--seeds", default="20260928,771014")
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0, help="0=核数-4")
    ap.add_argument("--out", default=str(REPO / "agent" / "out" / "recompute-search-v3.json"))
    ap.add_argument("--chunks", default=str(REPO / "agent" / "out" / "recompute-chunks"),
                    help="分块目录（幂等续跑）")
    ap.add_argument("--ref", default="/home/wuwenjie01/majiang_rl/records/ab-search-v3-vs-v3.json",
                    help="agent-d 的参照产物（只读，用于并排显示）")
    ap.add_argument("--identity", action="store_true", help="先做 ⓐ 同一性检验")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    workers = args.workers or max(1, (mp.cpu_count() or 4) - 4)

    print("=" * 74)
    print(f"独立复算（不 import nnrl.eval；主仓只读）arms={arms} seeds={seeds} "
          f"matches={args.matches} workers={workers}")
    print("=" * 74)

    ref_path = Path(args.ref)
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else None

    chunks = Path(args.chunks)
    chunks.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    t0 = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        if args.identity:
            check_rank_discards_identity()
        for arm in arms:
            for seed in seeds:
                key = f"{arm}@{seed}"
                chunk = chunks / f"{key}.json"
                # **幂等续跑**：已完成的块直接读回，不重算（抗进程回收）
                if chunk.exists():
                    try:
                        results[key] = json.loads(chunk.read_text())
                        print(f"\n[{key}] 已存在，跳过（幂等）")
                        continue
                    except Exception:  # noqa: BLE001
                        pass
                diffs, ts, bs = paired_ab_par(pool, arm, "v3", matches=args.matches,
                                              rounds=args.rounds, base_score=1, seed=seed)
                row = {}
                print(f"\n[{key}] n={args.matches * SEATS}（treatment − baseline=v3）")
                for lab in LABELS:
                    m, se, t, n = stat(diffs[lab])
                    row[lab] = {"mean": m, "se": se, "t": t, "n": n}
                    refm = ""
                    if ref:
                        try:
                            r = ref["arms"][arm][str(seed)][lab]
                            refm = f"   | agent-d: {r['mean']:+.3f} (t{r['t']:+.2f})"
                        except Exception:  # noqa: BLE001
                            refm = ""
                    print(f"   {lab:8s} {m:+8.3f} ±{se:5.3f}  t {t:+6.2f}  n={n}{refm}")
                row["_scores"] = {"treatment": ts, "baseline": bs}
                # 原子落块
                tmp = chunk.with_name(chunk.name + ".tmp")
                tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(chunk)
                results[key] = row

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"\n复算结果写入 {out}（块目录 {chunks}）  用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
