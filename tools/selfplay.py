"""批量自对弈评测（tasks.md 6.3）。

用法::

    # 我方占 1 座、基线占 3 座，跑 50 场 × 8 局
    uv run python tools/selfplay.py --mix heuristic,baseline,baseline,baseline --matches 50

    # 不同参数对比：决赛模式 vs 晋级轮模式
    uv run python tools/selfplay.py --mix final,qualifier,qualifier,qualifier --matches 50

策略名见 ``majiang.cli.DECIDERS``；``heuristic`` 支持 ``--mode`` 前缀写法
（``final`` / ``qualifier`` 直接作为策略名使用）。
"""

from __future__ import annotations

import argparse
import sys
import time

from majiang.cli import DECIDERS, make_decider
from majiang.sim.batch import run_batch, summary_lines
from majiang.strategy.policy import Mode

TALLY = {"heuristic": Mode.QUALIFIER, "final": Mode.FINAL, "qualifier": Mode.QUALIFIER}
ALIASES = {"baseline": "first-legal"}
AVAILABLE = sorted(set(DECIDERS) | set(TALLY) | set(ALIASES))


def build(name: str):
    if name in TALLY:
        return make_decider("heuristic", TALLY[name])
    resolved = ALIASES.get(name, name)
    if resolved in DECIDERS:
        return make_decider(resolved, Mode.QUALIFIER)
    raise SystemExit(f"未知策略 {name!r}，可选: {AVAILABLE}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量自对弈评测")
    parser.add_argument(
        "--mix",
        default="heuristic,baseline,baseline,baseline",
        help="四个座位依次使用的策略，逗号分隔",
    )
    parser.add_argument("--matches", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=8, help="每场局数")
    parser.add_argument("--base-score", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.mix.split(",")]
    if len(names) != 4:
        raise SystemExit(f"--mix 需要恰好四个策略名，实际 {len(names)} 个")
    unknown = [name for name in names if name not in AVAILABLE]
    if unknown:
        raise SystemExit(f"存在未知策略 {unknown}，可选: {AVAILABLE}")

    factories = [lambda name=name: build(name) for name in names]
    print(f"座次 {names}  场数 {args.matches}  每场 {args.rounds} 局  底分 {args.base_score}")
    started = time.perf_counter()
    result = run_batch(
        factories,
        matches=args.matches,
        rounds=args.rounds,
        base_score=args.base_score,
        seed=args.seed,
        labels=names,
    )
    elapsed = time.perf_counter() - started
    for line in summary_lines(result):
        print(line)
    print(f"用时 {elapsed:.1f}s（{result.rounds} 局，约 {elapsed / max(1, result.rounds) * 1000:.0f} ms/局）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
