"""Game-level parity: fast (.so) vs pure Python shanten, same seed.

Run twice: once with .so present (fast), once with .so moved away (pure).
Outputs JSON line to stdout for comparison.
"""
import json
import sys

sys.path.insert(0, "src")

from majiang.strategy import policy  # noqa: F401  (triggers monkey-patch if .so present)
from majiang.rules import shanten as sm
from majiang.sim.batch import run_match
from majiang.strategy.versions import build
from majiang.strategy.policy import Mode


def main():
    impl = sm.shanten.__module__  # 'majiang.rules.shanten_fast' or 'majiang.rules.shanten'
    deciders = [build("v5", Mode.QUALIFIER) for _ in range(4)]
    result = run_match(deciders, seed=20261003, rounds=4)
    summary = {
        "impl": impl,
        "total_scores": [s.total_score for s in result.seats],
        "wins": [s.wins for s in result.seats],
        "place_points": [s.place_points for s in result.seats],
        "god_count": [s.god_count for s in result.seats],
        "rounds": result.rounds,
        "flows": result.flows,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
