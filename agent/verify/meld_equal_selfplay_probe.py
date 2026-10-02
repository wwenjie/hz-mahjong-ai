"""A 2026-10-03 01:40 裁决第一步：自对弈量 meld-equal 副露率（零成本，不上平台）。

口径：同 field、同种子、四座同一档位（对手窗基线等价于四座同档），
只把 meld_tolerance 从 strict 换成 equal，数我方副露次数/局。
判据：副露率仍 <0.8 ⇒ 闸门不是唯一瓶颈，不上真机；升到 ~1.0 ⇒ 进第二步。

实现：不包装 decider（multiprocessing pickle 会崩），改为 monkey-patch
`majiang.sim.round` 里的 apply_peng/apply_chi/apply_gang/apply_minggang 计数。

用法：`.venv/bin/python agent/verify/meld_equal_selfplay_probe.py [--matches 30] [--rounds 8] [--seed 20260923]`
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.sim import round as round_mod  # noqa: E402
from majiang.sim.round import SEATS, run_round  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

# ---- monkey-patch 计数 -------------------------------------------------

_meld_counts = [0] * SEATS

_orig_peng = round_mod.apply_peng
_orig_chi = round_mod.apply_chi
_orig_gang = round_mod.apply_gang
_orig_minggang = round_mod.apply_minggang


def _counting_peng(state, seat, discarder, tile):
    _meld_counts[seat] += 1
    return _orig_peng(state, seat, discarder, tile)


def _counting_chi(state, seat, discarder, tile, action):
    _meld_counts[seat] += 1
    return _orig_chi(state, seat, discarder, tile, action)


def _counting_gang(state, seat, action):
    _meld_counts[seat] += 1
    return _orig_gang(state, seat, action)


def _counting_minggang(state, seat, discarder, tile):
    _meld_counts[seat] += 1
    return _orig_minggang(state, seat, discarder, tile)


def _install():
    round_mod.apply_peng = _counting_peng
    round_mod.apply_chi = _counting_chi
    round_mod.apply_gang = _counting_gang
    round_mod.apply_minggang = _counting_minggang


def _reset():
    for i in range(SEATS):
        _meld_counts[i] = 0


# ---- 主流程 ------------------------------------------------------------


def run(decider_name: str, matches: int, rounds: int, base_score: int, seed: int) -> dict:
    rng = random.Random(seed)
    total_rounds = 0
    total_melds = 0
    for _ in range(matches):
        deciders = [make_decider(decider_name, Mode.QUALIFIER) for _ in range(SEATS)]
        dealer = 0
        for r in range(rounds):
            _reset()
            outcome = run_round(
                deciders,
                dealer=dealer,
                round_no=r + 1,
                base_score=base_score,
                rng=rng,
            )
            total_rounds += 1
            total_melds += sum(_meld_counts)
            if outcome.is_flow:
                pass  # 流局连庄
            else:
                dealer = (dealer + 1) % SEATS if outcome.winner != dealer else dealer
    return {
        "decider": decider_name,
        "rounds_total": total_rounds,
        "total_melds": total_melds,
        "melds_per_round_per_seat": total_melds / SEATS / total_rounds if total_rounds else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="meld-equal vs strict 自对弈副露率（monkey-patch 计数）")
    ap.add_argument("--matches", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--base-score", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--arms", default="heuristic,meld-equal",
                    help="逗号分隔的档位名（默认 heuristic=strict 基线 + meld-equal）")
    args = ap.parse_args(argv)

    _install()

    print(f"同 field 同种子自对弈：matches={args.matches} rounds={args.rounds} seed={args.seed}")
    print(f"{'档位':>16} {'总局数':>7} {'总副露':>7} {'副露/局/座':>11}")
    for name in args.arms.split(","):
        name = name.strip()
        r = run(name, args.matches, args.rounds, args.base_score, args.seed)
        print(f"{name:>16} {r['rounds_total']:>7} {r['total_melds']:>7} "
              f"{r['melds_per_round_per_seat']:>11.3f}")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
