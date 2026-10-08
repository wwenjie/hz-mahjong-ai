# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""majiang.rules.shanten 的 Cython 加速移植。

只移植热路径：_blocks_value 递归核心 + tiles 内联助手 + shanten/best_shanten/ukeire。
语义必须与纯 Python 版 `majiang/rules/shanten.py` 逐例一致（parity 测试保证）。

牌索引约定（与 tiles.py 相同）：0-8 万、9-17 筒、18-26 条、27-33 字牌，GOD=33。
"""

from libc.stdint cimport uint8_t, int32_t

# ---- 常量（与 tiles.py 对齐）----
DEF TILE_KINDS = 34
DEF NUMBER_KINDS = 27
DEF HONOR_START = 27
DEF GOD = 33
DEF NUMBER_SUIT_SIZE = 9
DEF RUN_LENGTH = 3
DEF SET_LENGTH = 3
DEF HAND_SIZE = 13
DEF MELD_SLOTS = 3
DEF SETS_PER_HAND = 4
DEF MAX_MELDS = 4
DEF SHANTEN_MAX = 8
DEF SEVEN_PAIRS_MAX_PAIRS = 6
DEF COPIES_PER_KIND = 4


cdef inline bint _is_number(int tile) noexcept:
    return 0 <= tile < NUMBER_KINDS


cdef inline int _rank(int tile) noexcept:
    """数牌返回 1-9，字牌返回 1-7。"""
    if _is_number(tile):
        return tile % NUMBER_SUIT_SIZE + 1
    return tile - HONOR_START + 1


cdef inline bint _run_is_valid(int start) noexcept:
    return _is_number(start) and _rank(start) + RUN_LENGTH - 1 <= NUMBER_SUIT_SIZE


cdef inline int _first_nonzero(const uint8_t[:] counts) noexcept:
    cdef int tile
    for tile in range(TILE_KINDS):
        if counts[tile]:
            return tile
    return -1


cdef inline int _total(const uint8_t[:] counts) noexcept:
    cdef int s = 0, i
    for i in range(TILE_KINDS):
        s += counts[i]
    return s


cdef int _upper_bound(int tiles_available, int slots) noexcept:
    """在牌数与槽位限制下，2×面子 + 搭子 的上界。"""
    cdef int best = 0, sets, partials
    for sets in range(slots + 1):
        if 3 * sets > tiles_available:
            break
        partials = min(slots - sets, (tiles_available - 3 * sets) // 2)
        if 2 * sets + partials > best:
            best = 2 * sets + partials
    return best


cdef inline tuple _memo_key(const uint8_t[:] counts, int wildcards, int slots):
    """counts 打包成 bytes（34 字节），与 Python 版 bytes(list(counts)) 语义一致。"""
    cdef bytes b = bytes(bytearray([counts[i] for i in range(TILE_KINDS)]))
    return (b, wildcards, slots)


cdef int _blocks_value_rec(
    uint8_t[:] counts,
    int wildcards,
    int slots,
    dict memo,
) noexcept:
    """在至多 slots 个槽位内最大化 2×面子数 + 搭子数。与 Python 版逐例等价。"""
    cdef int available, bound, best, first, candidate
    cdef int used_real, need_wild, start, mask, offset, partner
    cdef bint feasible
    cdef tuple key
    cdef object cached

    if slots <= 0:
        return 0
    available = _total(counts) + wildcards
    if available < 2:
        return 0
    key = _memo_key(counts, wildcards, slots)
    cached = memo.get(key)
    if cached is not None:
        return <int>cached
    bound = _upper_bound(available, slots)
    best = 0
    first = _first_nonzero(counts)
    if first < 0:
        memo[key] = bound
        return bound

    # --- 刻子分支：first 处拿 1..min(3, counts[first]) 张真牌 + 财神补满 ---
    cdef int max_real = counts[first]
    if max_real > SET_LENGTH:
        max_real = SET_LENGTH
    used_real = max_real
    while used_real >= 1:
        need_wild = SET_LENGTH - used_real
        if need_wild <= wildcards:
            counts[first] -= used_real
            candidate = 2 + _blocks_value_rec(counts, wildcards - need_wild, slots - 1, memo)
            counts[first] += used_real
            if candidate > best:
                best = candidate
            if best >= bound:
                memo[key] = best
                return best
        used_real -= 1

    # --- 顺子分支：枚举包含 first 的三个候选起点 × 非空 mask ---
    for start in (first - 2, first - 1, first):
        if not _run_is_valid(start):
            continue
        if not (start <= first <= start + RUN_LENGTH - 1):
            continue
        mask = 1
        while mask < (1 << RUN_LENGTH):
            if (mask >> (first - start)) & 1:
                used_real = 0
                feasible = True
                for offset in range(RUN_LENGTH):
                    if (mask >> offset) & 1:
                        if counts[start + offset] <= 0:
                            feasible = False
                            break
                        used_real += 1
                need_wild = RUN_LENGTH - used_real
                if feasible and need_wild <= wildcards:
                    for offset in range(RUN_LENGTH):
                        if (mask >> offset) & 1:
                            counts[start + offset] -= 1
                    candidate = 2 + _blocks_value_rec(counts, wildcards - need_wild, slots - 1, memo)
                    for offset in range(RUN_LENGTH):
                        if (mask >> offset) & 1:
                            counts[start + offset] += 1
                    if candidate > best:
                        best = candidate
                    if best >= bound:
                        memo[key] = best
                        return best
            mask += 1

    # --- 对子分支（真对）---
    if counts[first] >= 2:
        counts[first] -= 2
        candidate = 1 + _blocks_value_rec(counts, wildcards, slots - 1, memo)
        counts[first] += 2
        if candidate > best:
            best = candidate

    # --- 单真牌 + 1 财神成对 ---
    if wildcards >= 1:
        counts[first] -= 1
        candidate = 1 + _blocks_value_rec(counts, wildcards - 1, slots - 1, memo)
        counts[first] += 1
        if candidate > best:
            best = candidate

    # --- 两面/坎张搭（真+真）---
    for offset in (1, 2):
        partner = first + offset
        if not _is_number(first) or partner >= TILE_KINDS:
            continue
        if _rank(partner) - _rank(first) != offset:
            continue
        if counts[partner] < 1:
            continue
        counts[first] -= 1
        counts[partner] -= 1
        candidate = 1 + _blocks_value_rec(counts, wildcards, slots - 1, memo)
        counts[first] += 1
        counts[partner] += 1
        if candidate > best:
            best = candidate

    # --- 双财神成搭 ---
    if wildcards >= 2:
        candidate = 1 + _blocks_value_rec(counts, wildcards - 2, slots - 1, memo)
        if candidate > best:
            best = candidate

    # --- 丢弃 first（不使用这张牌）---
    counts[first] -= 1
    candidate = _blocks_value_rec(counts, wildcards, slots, memo)
    counts[first] += 1
    if candidate > best:
        best = candidate

    memo[key] = best
    return best


cdef int _seven_pairs_shanten_arr(const uint8_t[:] counts) noexcept:
    cdef int i, pairs = 0, singles = 0, completed
    cdef int wildcards = counts[GOD]
    for i in range(TILE_KINDS):
        if i == GOD:
            continue
        pairs += counts[i] // 2
        singles += counts[i] % 2
    completed = singles if singles < wildcards else wildcards
    pairs += completed
    wildcards -= completed
    pairs += wildcards // 2
    if pairs > SEVEN_PAIRS_MAX_PAIRS:
        pairs = SEVEN_PAIRS_MAX_PAIRS
    return SEVEN_PAIRS_MAX_PAIRS - pairs


class ShantenError(ValueError):
    """入参非法（与 Python 版同名异常）。"""


def shanten(counts, meld_count: int = 0, memo=None) -> int:
    """未摸牌状态下的最小向听数（与 Python 版 shanten 逐例等价）。"""
    cdef int i, total = 0, v
    cdef uint8_t work[TILE_KINDS]
    n = len(counts)
    if n != TILE_KINDS:
        raise ShantenError(f"counts 长度应为 {TILE_KINDS}，实际 {n}")
    for i in range(TILE_KINDS):
        v = counts[i]
        if v < 0:
            raise ShantenError(f"计数为负: tile={i} amount={v}")
        work[i] = v
        total += v
    if not 0 <= meld_count <= MAX_MELDS:
        raise ShantenError(f"副露数越界: {meld_count}")
    expected = HAND_SIZE - MELD_SLOTS * meld_count
    if total != expected:
        raise ShantenError(f"暗手牌张数应为 {expected}，实际 {total}")

    cdef int wildcards = work[GOD]
    work[GOD] = 0
    cdef int slots = SETS_PER_HAND - meld_count
    cdef dict table = {} if memo is None else memo
    cdef int best = SHANTEN_MAX - 2 * meld_count - _blocks_value_rec(work, wildcards, slots, table)

    # 将牌分支：真对
    cdef int bv
    cdef int candidate
    cdef int sp
    cdef uint8_t orig[TILE_KINDS]
    for i in range(TILE_KINDS):
        if work[i] >= 2:
            work[i] -= 2
            bv = _blocks_value_rec(work, wildcards, slots, table)
            work[i] += 2
            candidate = SHANTEN_MAX - 2 * meld_count - 1 - bv
            if candidate < best:
                best = candidate
    # 将牌分支：单真牌 + 财神
    if wildcards >= 1:
        for i in range(TILE_KINDS):
            if work[i] >= 1:
                work[i] -= 1
                bv = _blocks_value_rec(work, wildcards - 1, slots, table)
                work[i] += 1
                candidate = SHANTEN_MAX - 2 * meld_count - 1 - bv
                if candidate < best:
                    best = candidate
    # 将牌分支：双财神
    if wildcards >= 2:
        bv = _blocks_value_rec(work, wildcards - 2, slots, table)
        candidate = SHANTEN_MAX - 2 * meld_count - 1 - bv
        if candidate < best:
            best = candidate

    if meld_count == 0:
        # 七对用原始 counts（含财神）
        for i in range(TILE_KINDS):
            orig[i] = counts[i]
        sp = _seven_pairs_shanten_arr(orig)
        if sp < best:
            best = sp
    return best if best > 0 else 0


def best_shanten(counts, meld_count: int = 0, memo=None) -> int:
    """摸牌后打出任意一张可达到的最小向听数。"""
    cdef int i
    cdef uint8_t work[TILE_KINDS]
    cdef list wlist = [int(counts[i]) for i in range(TILE_KINDS)]
    cdef dict table = {} if memo is None else memo
    cdef int best = SHANTEN_MAX
    cdef int s
    for tile in range(TILE_KINDS):
        if wlist[tile] == 0:
            continue
        wlist[tile] -= 1
        s = shanten(wlist, meld_count, memo=table)
        wlist[tile] += 1
        if s < best:
            best = s
    return best


def shanten_any(counts, meld_count: int = 0, memo=None) -> int:
    """当前张数下的最小向听（未摸牌直接算；摸牌后取打任意一张的最小）。"""
    target = HAND_SIZE - MELD_SLOTS * meld_count
    total = sum(int(counts[i]) for i in range(TILE_KINDS))
    if total == target:
        return shanten(counts, meld_count, memo=memo)
    if total == target + 1:
        return best_shanten(counts, meld_count, memo=memo)
    raise ShantenError(f"手牌张数 {total} 与 {meld_count} 组副露不符（应为 {target} 或 {target + 1}）")


def seven_pairs_shanten(counts) -> int:
    cdef uint8_t arr[TILE_KINDS]
    cdef int i
    for i in range(TILE_KINDS):
        arr[i] = counts[i]
    return _seven_pairs_shanten_arr(arr)


def ukeire(counts, meld_count: int = 0, visible=None, memo=None):
    """能降低最小向听数的进张，返回 ((牌种, 剩余可见张数), ...)。与 Python 版等价。"""
    cdef int tile, remaining
    cdef list wlist = [int(counts[i]) for i in range(TILE_KINDS)]
    cdef dict table = {} if memo is None else memo
    cdef int current = shanten(wlist, meld_count, memo=table)
    if current == 0:
        return ()
    cdef list baseline = [int(visible[i]) for i in range(TILE_KINDS)] if visible is not None else list(wlist)
    cdef list found = []
    for tile in range(TILE_KINDS):
        remaining = COPIES_PER_KIND - baseline[tile]
        if remaining <= 0:
            continue
        wlist[tile] += 1
        improved = best_shanten(wlist, meld_count, memo=table) < current
        wlist[tile] -= 1
        if improved:
            found.append((tile, remaining))
    return tuple(found)
