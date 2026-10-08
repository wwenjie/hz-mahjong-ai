"""Parity test: Cython shanten_fast vs pure-Python shanten.

Same hands through both implementations, outputs must be identical.
"""
import random
import sys

sys.path.insert(0, "src")

from majiang.rules import shanten as py_shanten
from majiang.rules import shanten_fast as cy_shanten
from majiang.rules.tiles import TILE_KINDS, GOD, COPIES_PER_KIND


def random_hand(rng: random.Random, n: int, max_god: int = 3) -> list[int]:
    counts = [0] * TILE_KINDS
    placed = 0
    while placed < n:
        t = rng.randrange(TILE_KINDS)
        if t == GOD and counts[GOD] >= max_god:
            continue
        if counts[t] >= COPIES_PER_KIND:
            continue
        counts[t] += 1
        placed += 1
    return counts


def main(n_cases: int = 300, seed: int = 20261005) -> int:
    rng = random.Random(seed)
    failures = 0
    checked = 0
    for i in range(n_cases):
        meld = rng.choice([0, 0, 0, 1, 2])
        hand_size = 13 - 3 * meld
        counts = random_hand(rng, hand_size)
        memo_py, memo_cy = {}, {}
        try:
            s_py = py_shanten.shanten(counts, meld, memo=memo_py)
        except Exception as e:
            print(f"[{i}] PY shanten raised: {e}")
            failures += 1
            continue
        try:
            s_cy = cy_shanten.shanten(counts, meld, memo=memo_cy)
        except Exception as e:
            print(f"[{i}] CY shanten raised: {e}")
            failures += 1
            continue
        checked += 1
        if s_py != s_cy:
            print(f"[{i}] shanten MISMATCH meld={meld} py={s_py} cy={s_cy} counts={counts}")
            failures += 1
            continue
        # best_shanten on hand+1 tile
        if hand_size + 1 <= 14:
            t = rng.randrange(TILE_KINDS - 1)  # avoid god overflow
            if counts[t] < COPIES_PER_KIND:
                counts2 = list(counts)
                counts2[t] += 1
                b_py = py_shanten.best_shanten(counts2, meld)
                b_cy = cy_shanten.best_shanten(counts2, meld)
                checked += 1
                if b_py != b_cy:
                    print(f"[{i}] best_shanten MISMATCH py={b_py} cy={b_cy}")
                    failures += 1
                    continue
        # ukeire (only for 0-meld, keep runtime sane)
        if meld == 0 and s_py > 0:
            u_py = py_shanten.ukeire(counts, meld)
            u_cy = cy_shanten.ukeire(counts, meld)
            checked += 1
            if u_py != u_cy:
                print(f"[{i}] ukeire MISMATCH py={u_py} cy={u_cy}")
                failures += 1
    print(f"\nchecked={checked} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
