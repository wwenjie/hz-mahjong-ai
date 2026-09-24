"""胡牌判定与分支番型。

成立形态（平台 §1.2）：

- 普通胡：4 组面子 + 1 对将；面子含顺子、刻子、杠，副露每组折算一个面子。
- 七对子：7 个对子，禁止任何吃碰杠副露。

财神（白板）是百搭，可替代任意普通牌完成面子或对子，本身不作为自然牌参与组合。

判定算法是「枚举将牌 + 对剩余牌做面子分解」的完整回溯：先定下将牌（两张真牌 /
一张真牌加一张财神 / 两张财神），再从最小的真牌出发枚举该牌所属面子的所有形态
（刻子用 k 张真牌、顺子以三个候选起点枚举真牌子集）。当真牌张数少于待组面子数时，
额外保留一条「整组用财神凑」的分支——这是必需的，因为此时总有面子分不到真牌。

正确性由 `tests/test_win_reference.py` 的差分测试保证：那里用一套结构完全独立的
暴力实现（把财神逐个指派成具体牌种，再走不含百搭的教科书式递归）逐例对拍。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import tiles
from .tiles import GOD, TILE_KINDS

BRANCH_FAN: dict[str, int] = {
    "平胡": 1,
    "七对": 2,
    "豪华七对×1": 4,
    "豪华七对×2": 8,
    "豪华七对×3": 16,
}
SEVEN_PAIRS = 7
SEVEN_PAIRS_MAX_QUADS = 3
_SEVEN_PAIRS_NAMES: tuple[str, ...] = ("七对", "豪华七对×1", "豪华七对×2", "豪华七对×3")


@dataclass(frozen=True, slots=True)
class Branch:
    fan: int
    name: str
    seven_pairs: bool = False


def expected_drawn_size(meld_count: int) -> int:
    return 14 - tiles.MELD_SLOTS * meld_count


def expected_waiting_size(meld_count: int) -> int:
    return tiles.HAND_SIZE - tiles.MELD_SLOTS * meld_count


def _first_nonzero(counts: Sequence[int]) -> int | None:
    for tile, amount in enumerate(counts):
        if amount:
            return tile
    return None


def _search_sets(
    counts: list[int],
    real_left: int,
    wildcards: int,
    sets_left: int,
    memo: dict[tuple[bytes, int, int], bool],
) -> bool:
    if sets_left == 0:
        return real_left == 0 and wildcards == 0
    if real_left + wildcards != tiles.SET_LENGTH * sets_left:
        return False
    key = (bytes(counts), wildcards, sets_left)
    cached = memo.get(key)
    if cached is not None:
        return cached
    result = _try_set_branches(counts, real_left, wildcards, sets_left, memo)
    memo[key] = result
    return result


def _try_set_branches(
    counts: list[int],
    real_left: int,
    wildcards: int,
    sets_left: int,
    memo: dict[tuple[bytes, int, int], bool],
) -> bool:
    first = _first_nonzero(counts)
    if first is None:
        return wildcards == tiles.SET_LENGTH * sets_left

    for used_real in range(min(tiles.SET_LENGTH, counts[first]), 0, -1):
        need_wild = tiles.SET_LENGTH - used_real
        if need_wild > wildcards:
            continue
        counts[first] -= used_real
        solved = _search_sets(counts, real_left - used_real, wildcards - need_wild, sets_left - 1, memo)
        counts[first] += used_real
        if solved:
            return True

    for start in (first - 2, first - 1, first):
        if not tiles.run_is_valid(start) or not start <= first <= start + tiles.RUN_LENGTH - 1:
            continue
        for mask in range(1, 1 << tiles.RUN_LENGTH):
            if not (mask >> (first - start)) & 1:
                continue
            used_real = 0
            feasible = True
            for offset in range(tiles.RUN_LENGTH):
                if mask >> offset & 1:
                    if counts[start + offset] <= 0:
                        feasible = False
                        break
                    used_real += 1
            need_wild = tiles.RUN_LENGTH - used_real
            if not feasible or need_wild > wildcards:
                continue
            for offset in range(tiles.RUN_LENGTH):
                if mask >> offset & 1:
                    counts[start + offset] -= 1
            solved = _search_sets(counts, real_left - used_real, wildcards - need_wild, sets_left - 1, memo)
            for offset in range(tiles.RUN_LENGTH):
                if mask >> offset & 1:
                    counts[start + offset] += 1
            if solved:
                return True

    if wildcards >= tiles.SET_LENGTH and real_left < sets_left:
        if _search_sets(counts, real_left, wildcards - tiles.SET_LENGTH, sets_left - 1, memo):
            return True

    return False


def _can_form_regular(
    counts: list[int],
    real_left: int,
    wildcards: int,
    sets_needed: int,
    memo: dict[tuple[bytes, int, int], bool],
) -> bool:
    if wildcards >= 2 and _search_sets(counts, real_left, wildcards - 2, sets_needed, memo):
        return True
    for tile in range(TILE_KINDS):
        if tile == GOD or counts[tile] == 0:
            continue
        if counts[tile] >= 2:
            counts[tile] -= 2
            solved = _search_sets(counts, real_left - 2, wildcards, sets_needed, memo)
            counts[tile] += 2
            if solved:
                return True
        if wildcards >= 1:
            counts[tile] -= 1
            solved = _search_sets(counts, real_left - 1, wildcards - 1, sets_needed, memo)
            counts[tile] += 1
            if solved:
                return True
    return False


def _winning_shape_with_memo(
    counts: Sequence[int],
    meld_count: int,
    memo: dict[tuple[bytes, int, int], bool],
) -> bool:
    if tiles.total_tiles(counts) != expected_drawn_size(meld_count):
        return False
    if meld_count == 0 and seven_pairs_quads(counts) is not None:
        return True
    work = list(counts)
    wildcards = work[GOD]
    work[GOD] = 0
    real_left = sum(work)
    return _can_form_regular(work, real_left, wildcards, tiles.SETS_PER_HAND - meld_count, memo)


def is_winning_shape(counts: Sequence[int], meld_count: int = 0) -> bool:
    """给定已摸牌在内的暗手牌，判断是否构成胡牌形态（普通胡或七对子）。"""
    if not 0 <= meld_count <= tiles.MAX_MELDS:
        return False
    return _winning_shape_with_memo(counts, meld_count, {})


def seven_pairs_quads(counts: Sequence[int]) -> int | None:
    """七对成立时返回豪华组数（0–3），否则返回 None。"""
    if tiles.total_tiles(counts) != 14 or counts[GOD] > tiles.COPIES_PER_KIND:
        return None
    work = list(counts)
    wildcards = work[GOD]
    work[GOD] = 0

    pairs = sum(amount // 2 for amount in work)
    singles = sum(amount % 2 for amount in work)
    natural_pairs = sum(1 for amount in work if amount == 2)
    quads = sum(1 for amount in work if amount == tiles.COPIES_PER_KIND)

    completed = min(singles, wildcards)
    pairs += completed
    singles -= completed
    wildcards -= completed
    pairs += wildcards // 2
    leftover = singles + wildcards % 2

    if pairs != SEVEN_PAIRS or leftover != 0:
        return None

    if (
        counts[GOD] == tiles.COPIES_PER_KIND
        and singles == 0
        and natural_pairs == (14 - tiles.COPIES_PER_KIND) // 2
        and all(amount in (0, 2) for amount in work)
    ):
        quads = max(quads, 1)
    return min(quads, SEVEN_PAIRS_MAX_QUADS)


def best_branch(counts: Sequence[int], meld_count: int = 0) -> Branch | None:
    """返回番值最高的成立形态，不成立返回 None。"""
    if not is_winning_shape(counts, meld_count):
        return None
    best = Branch(fan=BRANCH_FAN["平胡"], name="平胡")
    if meld_count == 0:
        quads = seven_pairs_quads(counts)
        if quads is not None:
            name = _SEVEN_PAIRS_NAMES[quads]
            if BRANCH_FAN[name] > best.fan:
                best = Branch(fan=BRANCH_FAN[name], name=name, seven_pairs=True)
    return best


def winning_draws(counts: Sequence[int], meld_count: int = 0) -> tuple[int, ...]:
    """听牌张：摸到哪些牌种可以完成胡牌。已耗尽的牌种不计入。"""
    expected = expected_waiting_size(meld_count)
    if tiles.total_tiles(counts) != expected:
        raise ValueError(f"暗手牌张数应为 {expected}，实际 {tiles.total_tiles(counts)}")
    work = list(counts)
    memo: dict[tuple[bytes, int, int], bool] = {}
    found: list[int] = []
    for tile in range(TILE_KINDS):
        if work[tile] >= tiles.COPIES_PER_KIND:
            continue
        work[tile] += 1
        if _winning_shape_with_memo(work, meld_count, memo):
            found.append(tile)
        work[tile] -= 1
    return tuple(found)


def drawable_kind_count(counts: Sequence[int]) -> int:
    """还有可能摸到的牌种数量。"""
    return sum(1 for amount in counts if amount < tiles.COPIES_PER_KIND)


def is_baotou(counts: Sequence[int], meld_count: int = 0) -> bool:
    """爆头：听任意，即任意一张还能摸到的牌都能完成胡牌。"""
    expected = expected_waiting_size(meld_count)
    if tiles.total_tiles(counts) != expected:
        return False
    work = list(counts)
    memo: dict[tuple[bytes, int, int], bool] = {}
    drawable = 0
    for tile in range(TILE_KINDS):
        if work[tile] >= tiles.COPIES_PER_KIND:
            continue
        drawable += 1
        work[tile] += 1
        solved = _winning_shape_with_memo(work, meld_count, memo)
        work[tile] -= 1
        if not solved:
            return False
    return drawable > 0
