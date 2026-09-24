"""向听数与进张。

向听数 = 还需要更换多少张牌才能进入听牌态；听牌态记 0。

一般形用标准公式 ``8 - 2×面子数 - 搭子数 - 将``：面子与搭子共享 4 个槽位，将另占
一个槽位。财神按百搭参与，因此面子/搭子的枚举与胡牌判定同源（最小的真牌必须被
本层消耗，顺子枚举三个候选起点）。

七对形用 ``6 - 对数``（七对不允许任何副露），财神可补对，四张同牌算两对。

``shanten`` 只接受未摸牌张数的暗手牌（``13 - 3×副露数``）。0 向听与
:func:`majiang.rules.win.winning严格等价，这一点由测试逐例保证。
"""

from __future__ import annotations

from collections.abc import Sequence

from . import tiles
from .tiles import GOD, TILE_KINDS

SHANTEN_MAX = 8
SEVEN_PAIRS_MAX_PAIRS = 6


class ShantenError(ValueError):
    """入参非法。"""


def _first_nonzero(counts: Sequence[int]) -> int | None:
    for tile, amount in enumerate(counts):
        if amount:
            return tile
    return None


def _upper_bound(tiles_available: int, slots: int) -> int:
    """在牌数与槽位限制下，``2×面子 + 搭子`` 的上界。"""
    best = 0
    for sets in range(slots + 1):
        if 3 * sets > tiles_available:
            break
        partials = min(slots - sets, (tiles_available - 3 * sets) // 2)
        best = max(best, 2 * sets + partials)
    return best


def _blocks_value(
    counts: list[int],
    wildcards: int,
    slots: int,
    memo: dict[tuple[bytes, int, int], int],
) -> int:
    """在至多 ``slots`` 个槽位内最大化 ``2×面子数 + 搭子数``。"""
    if slots <= 0:
        return 0
    available = sum(counts) + wildcards
    if available < 2:
        return 0
    key = (bytes(counts), wildcards, slots)
    cached = memo.get(key)
    if cached is not None:
        return cached
    bound = _upper_bound(available, slots)
    best = 0
    first = _first_nonzero(counts)
    if first is None:
        best = bound
        memo[key] = best
        return best

    for used_real in range(min(tiles.SET_LENGTH, counts[first]), 0, -1):
        need_wild = tiles.SET_LENGTH - used_real
        if need_wild > wildcards:
            continue
        counts[first] -= used_real
        candidate = 2 + _blocks_value(counts, wildcards - need_wild, slots - 1, memo)
        counts[first] += used_real
        best = max(best, candidate)
        if best >= bound:
            memo[key] = best
            return best

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
            candidate = 2 + _blocks_value(counts, wildcards - need_wild, slots - 1, memo)
            for offset in range(tiles.RUN_LENGTH):
                if mask >> offset & 1:
                    counts[start + offset] += 1
            best = max(best, candidate)
            if best >= bound:
                memo[key] = best
                return best

    if counts[first] >= 2:
        counts[first] -= 2
        best = max(best, 1 + _blocks_value(counts, wildcards, slots - 1, memo))
        counts[first] += 2

    if wildcards >= 1:
        counts[first] -= 1
        best = max(best, 1 + _blocks_value(counts, wildcards - 1, slots - 1, memo))
        counts[first] += 1

    for offset in (1, 2):
        partner = first + offset
        if not tiles.is_number(first) or partner >= TILE_KINDS:
            continue
        if tiles.rank(partner) - tiles.rank(first) != offset:
            continue
        if counts[partner] < 1:
            continue
        counts[first] -= 1
        counts[partner] -= 1
        best = max(best, 1 + _blocks_value(counts, wildcards, slots - 1, memo))
        counts[first] += 1
        counts[partner] += 1

    if wildcards >= 2:
        best = max(best, 1 + _blocks_value(counts, wildcards - 2, slots - 1, memo))

    counts[first] -= 1
    best = max(best, _blocks_value(counts, wildcards, slots, memo))
    counts[first] += 1

    memo[key] = best
    return best


def _seven_pairs_shanten(counts: Sequence[int]) -> int:
    real = list(counts)
    wildcards = real[GOD]
    real[GOD] = 0
    pairs = sum(amount // 2 for amount in real)
    singles = sum(amount % 2 for amount in real)
    completed = min(singles, wildcards)
    pairs += completed
    wildcards -= completed
    pairs += wildcards // 2
    return SEVEN_PAIRS_MAX_PAIRS - min(pairs, SEVEN_PAIRS_MAX_PAIRS)


Memo = dict[tuple[bytes, int, int], int]


def seven_pairs_shanten(counts: Sequence[int]) -> int:
    """七对形的向听数（不取与一般形的最小值）。"""
    return _seven_pairs_shanten(counts)


def shanten(
    counts: Sequence[int],
    meld_count: int = 0,
    *,
    memo: Memo | None = None,
) -> int:
    """未摸牌状态下的最小向听数。

    ``memo`` 可跨多次调用共享：键由（真牌计数、财神数、剩余槽位）构成，与具体是哪一手
    无关，因此比较多个出牌候选时共用一个记忆表能显著省时。
    """
    if not 0 <= meld_count <= tiles.MAX_MELDS:
        raise ShantenError(f"副露数越界: {meld_count}")
    expected = tiles.HAND_SIZE - tiles.MELD_SLOTS * meld_count
    if tiles.total_tiles(counts) != expected:
        raise ShantenError(f"暗手牌张数应为 {expected}，实际 {tiles.total_tiles(counts)}")

    work = list(counts)
    wildcards = work[GOD]
    work[GOD] = 0
    slots = tiles.SETS_PER_HAND - meld_count
    table: Memo = {} if memo is None else memo
    best = SHANTEN_MAX - 2 * meld_count - _blocks_value(work, wildcards, slots, table)

    for tile, amount in enumerate(work):
        if amount >= 2:
            work[tile] -= 2
            best = min(
                best,
                SHANTEN_MAX - 2 * meld_count - 1 - _blocks_value(work, wildcards, slots, table),
            )
            work[tile] += 2
        if amount >= 1 and wildcards >= 1:
            work[tile] -= 1
            best = min(
                best,
                SHANTEN_MAX
                - 2 * meld_count
                - 1
                - _blocks_value(work, wildcards - 1, slots, table),
            )
            work[tile] += 1
    if wildcards >= 2:
        best = min(
            best,
            SHANTEN_MAX - 2 * meld_count - 1 - _blocks_value(work, wildcards - 2, slots, table),
        )

    if meld_count == 0:
        best = min(best, _seven_pairs_shanten(counts))
    return max(0, best)


def shanten_any(
    counts: Sequence[int],
    meld_count: int = 0,
    *,
    memo: Memo | None = None,
) -> int:
    """当前张数下的最小向听。

    未摸牌张数（``13 - 3×副露``）直接算；摸牌后的多一张状态取「打出任意一张后的最小」。
    决策层拿到的往往是摸牌后的手牌，直接用 :func:`shanten` 会因张数不符而抛错——这正是
    之前把一手已听牌的牌算成 8 向听的成因。
    """
    target = tiles.HAND_SIZE - tiles.MELD_SLOTS * meld_count
    total = tiles.total_tiles(counts)
    if total == target:
        return shanten(counts, meld_count, memo=memo)
    if total == target + 1:
        return best_shanten(counts, meld_count, memo=memo)
    raise ShantenError(f"手牌张数 {total} 与 {meld_count} 组副露不符（应为 {target} 或 {target + 1}）")


def natural_shanten(counts: Sequence[int], meld_count: int = 0) -> int:
    """「财神当将、只需做成 4 组**自然**面子」时的向听数。

    爆头（听任意）要求 4 组自然面子 + 1 张闲余财神。把财神当百搭用掉就得不到这个形态，
    所以这条路线必须按**纯真牌**衡量进度，并把将牌视为已由财神兜底（公式里 h 恒为 1）。
    """
    real = list(counts)
    wildcards = real[GOD]
    real[GOD] = 0
    sets, partials, _pair = quick_blocks(real)
    value = 2 * sets + partials
    return max(0, SHANTEN_MAX - 2 * meld_count - value - (1 if wildcards else 0))


def best_shanten(
    counts: Sequence[int],
    meld_count: int = 0,
    *,
    memo: Memo | None = None,
) -> int:
    """摸牌后打出任意一张可达到的最小向听数。``counts`` 为摸牌后的张数。"""
    work = list(counts)
    table: Memo = {} if memo is None else memo
    best = SHANTEN_MAX
    for tile in range(tiles.TILE_KINDS):
        if work[tile] == 0:
            continue
        work[tile] -= 1
        try:
            best = min(best, shanten(work, meld_count, memo=table))
        finally:
            work[tile] += 1
    return best


def ukeire(
    counts: Sequence[int],
    meld_count: int = 0,
    *,
    visible: Sequence[int] | None = None,
    memo: Memo | None = None,
) -> tuple[tuple[int, int], ...]:
    """能降低**最小向听数**的进张，返回 ``(牌种, 剩余可见张数)``。

    ``counts`` 是打出一张之后的暗手牌（未摸牌张数）。``visible`` 为各牌种已可见张数
    （本手暗牌 + 四家副露 + 弃牌），缺省只用本手牌计数。

    注意开销：内部对每个可摸牌种各做一次 ``best_shanten``，实测单次调用约 0.1–0.2 秒
    量级，**不适合放在决策热路径**，主要用于离线分析与评测。
    """
    work = list(counts)
    table: Memo = {} if memo is None else memo
    current = shanten(work, meld_count, memo=table)
    if current == 0:
        return ()
    baseline = list(visible) if visible is not None else list(work)
    found: list[tuple[int, int]] = []
    for tile in range(tiles.TILE_KINDS):
        remaining = tiles.COPIES_PER_KIND - baseline[tile]
        if remaining <= 0:
            continue
        work[tile] += 1
        improved = best_shanten(work, meld_count, memo=table) < current
        work[tile] -= 1
        if improved:
            found.append((tile, remaining))
    return tuple(found)


def quick_blocks(counts: Sequence[int]) -> tuple[int, int, int]:
    """按花色贪心分解手牌骨架，返回 ``(面子数, 搭子数, 含对子)``。

    这是**近似**值，不保证最优，但代价是微秒级。用途是在决策热路径上做同向听候选项
    之间的次排序——精确的 ``best_shanten`` 实测约 4.9 ms、``ukeire`` 约 0.1–0.2 s，
    放不进每个候选的评估里。

    结果按向听的块位约束**裁剪**：面子 + 搭子至多 4 组。不裁剪的话，手里的搭子超过
    4 个时 `2×面子 + 搭子` 仍会继续变大，于是「随便加一张牌都算进步」，用作进张估计
    会严重失真（实测报出 61–80 张进张，而精确值只有 8–25）。
    """
    sets = partials = pairs = 0
    real = list(counts)
    real[GOD] = 0
    for start, size in ((0, 9), (9, 9), (18, 9), (27, 7)):
        group = real[start : start + size]
        if start == 27:
            for amount in group:
                sets += amount // tiles.SET_LENGTH
                remainder = amount % tiles.SET_LENGTH
                pairs += remainder // 2
                partials += remainder // 2
            continue
        for index in range(size):
            while group[index] >= tiles.SET_LENGTH:
                group[index] -= tiles.SET_LENGTH
                sets += 1
        for index in range(size - tiles.RUN_LENGTH + 1):
            while group[index] and group[index + 1] and group[index + 2]:
                group[index] -= 1
                group[index + 1] -= 1
                group[index + 2] -= 1
                sets += 1
        for index in range(size):
            if group[index] >= 2:
                group[index] -= 2
                partials += 1
                pairs += 1
        for index in range(size - 1):
            if group[index] and group[index + 1]:
                group[index] -= 1
                group[index + 1] -= 1
                partials += 1
        for index in range(size - 2):
            if group[index] and group[index + 2]:
                group[index] -= 1
                group[index + 2] -= 1
                partials += 1
    partials += counts[GOD]
    slots = tiles.SETS_PER_HAND
    sets = min(sets, slots)
    partials = min(partials, slots - sets)
    return sets, partials, min(1, pairs)


def quick_shanten(counts: Sequence[int], meld_count: int = 0) -> int:
    """骨架版的向听近似（微秒级），用于同向听候选项之间的次排序与廉价进张估计。"""
    sets, partials, pair = quick_blocks(counts)
    total_sets = sets + meld_count
    value = 2 * sets + partials
    return max(0, SHANTEN_MAX - 2 * meld_count - value - pair)


def visible_counts(
    counts: Sequence[int],
    melds: Sequence[Sequence[int]] = (),
    discards: Sequence[Sequence[int]] = (),
) -> list[int]:
    """各牌种的已可见张数：本手暗牌 + 各家副露 + 各家弃牌。"""
    seen = list(counts)
    for group in melds:
        for tile in group:
            seen[tile] += 1
    for group in discards:
        for tile in group:
            seen[tile] += 1
    return seen
