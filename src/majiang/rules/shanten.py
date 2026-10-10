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

import os
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


# 超额对子奖励的系数（字牌对子不能被吃、是最好的碰材，故高于数牌）：见 shape_value 尾部注释。
PAIR_BONUS_HONOR = 0.5
PAIR_BONUS_SUITED = 0.3
# 奖励**总量上限**（乘以 keep_extra_pairs）：多个多余对子累加后仍须 < 1.0，
# 否则会越过 `len(taken)` 的「块数差 1」并覆盖排序语义——这是 shape_value 的核心不变量。
PAIR_BONUS_CAP = 0.9


def shape_value(
    counts: Sequence[int],
    meld_count: int = 0,
    *,
    edge_partial_weight: float | None = None,
    keep_extra_pairs: float = 0.0,
    grade: str = "mean",
) -> float:
    """骨架的**加权形质值**（微秒级），用于替代 ``2×面子 + 搭子`` 的次排序。

    **为什么需要它**：``quick_blocks`` 把「两面 / 对子 / 坎张」都记作 1 个搭子，
    再 ``partials = min(partials, 4 - sets)`` —— 而手牌几乎总有 ≥4 个块，
    于是这个裁剪**恒饱和**，同向听候选之间的取值高度集中。实测（3493 个真机决策点）：

    =========  =========  ==================  ==========
    副露数      决策点数   **全并列占比**      不同值数
    =========  =========  ==================  ==========
    0            3493      **77.2%**            1.23
    1            1255      **89.2%**            1.11
    2             217      **93.5%**            1.07
    =========  =========  ==================  ==========

    ⇒ 次排序一旦并列，``_choose_discard`` 就只剩「喂牌」一项在起作用，
    **中段等于完全没有形质概念**。这也解释了为什么 `ukeire-wide / ukeire-hand /
    ukeire-early` 全族测平：候选池是按这个退化键排序的，加宽池子等于随机采样。

    **设计要点（刻意做成窄改动）**：返回值 = ``2×面子 + 块数 + 形质修正``，其中
    形质修正的幅度 **< 1**（``mean_w − 1`` 的绝对值不超过 0.4）。
    这保证它**只能打破并列、永远不会覆盖「块数差 1」**——排序语义不变，
    只是把原本并列的候选分开。搭子按「能等到几张」分级：两面 1.2（8 张）、
    对子 1.0（2 张成刻且可当雀头）、坎张/边张 0.7（4 张）、财神 1.4（百搭）。

    **顺带修一处副露口径错**：需要几个块是 ``SETS_PER_HAND - meld_count``
    （含雀头），而 ``quick_blocks`` 恒按 4 裁。``quick_shanten`` 用 ``−2×meld_count``
    补偿了向听，但 ``2×面子 + 搭子`` 这个**次排序项没有补偿** ⇒ 副露越多，
    「多留一个用不上的搭子」越被加分。
    **超额对子奖励**（``keep_extra_pairs``，默认 0.0 ⇒ 逐位等于旧行为）：上面那个
    ``weights[:need]`` 裁剪把**排在刀口外的块**当 0，于是「多出来的对子」在估值里
    完全消失——拆掉它不掉分。实测（1200 副「唯一对子=东东」手牌）唯一对子不会
    被拆，但当手里有 ≥2 个对子时，多余的对子被当 0，会与孤立字牌在平局规则里被
    重新排序 ⇒ 偶尔拆掉一个对子（含字牌对子，而它不能被吃、是最好的碰材）。本参数
    给被裁掉的**对子**一个折扣正值（字牌 0.5、数牌 0.3，乘以 ``keep_extra_pairs``），
    使「多留一个对子」优于「留一张孤张」，且幅度 < 1 ⇒ 仍只打破并列、不覆盖块数差。
    """
    real = list(counts)
    real[GOD] = 0
    sets = 0
    # 每个块 = (搭子权重, 该块是「对子」时的**超额对子奖励**)。
    # 奖励在裁剪之后只对**被裁掉的**对子累加（见函数尾），保证 `keep_extra_pairs=0` 时逐位等于旧行为。
    pair_bonus_honor = PAIR_BONUS_HONOR * keep_extra_pairs
    pair_bonus_suited = PAIR_BONUS_SUITED * keep_extra_pairs
    blocks: list[tuple[float, float]] = []
    for start, size in ((0, 9), (9, 9), (18, 9), (27, 7)):
        group = real[start : start + size]
        if start == 27:  # 字牌不能成顺
            for amount in group:
                sets += amount // tiles.SET_LENGTH
                if amount % tiles.SET_LENGTH >= 2:
                    blocks.append((1.0, pair_bonus_honor))
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
        # 两面优先于对子：它等到 8 张，对子只等到 2 张（另可当雀头，故不低太多）
        for index in range(size - 1):
            while group[index] and group[index + 1]:
                group[index] -= 1
                group[index + 1] -= 1
                # **边张搭（12 / 89）不是两面**：它只等到 1 张牌种（4 张），与坎张同量级，
                # 而真两面（如 45）等到 2 个牌种（8 张）。2026-10-02 实测：
                # `8w9w` 与 `4w5w` 在本函数里**同值**（都算 1.2），进张却是 4 vs 8 ⇒ 边张搭高估一倍。
                # 这是 agent-c 22:20 的机制信号（「排序键对边张/边搭取舍疑似反了」）量化后的形态：
                # 不是反了，是**没有区分**。`edge_partial_weight=None` 保持旧行为；
                # 给 0.7 就与坎张同权，给 0.8 则介于坎张与真两面之间。
                is_edge = index == 0 or index == size - 2  # 组内 rank 1-2 或 8-9
                if edge_partial_weight is not None and is_edge:
                    blocks.append((edge_partial_weight, 0.0))
                else:
                    blocks.append((1.2, 0.0))
        for index in range(size):
            while group[index] >= 2:
                group[index] -= 2
                blocks.append((1.0, pair_bonus_suited))
        for index in range(size - 2):
            while group[index] and group[index + 2]:
                group[index] -= 1
                group[index + 2] -= 1
                blocks.append((0.7, 0.0))
    if counts[GOD]:
        blocks.append((1.4, 0.0))
    # 只需要 `SETS_PER_HAND - meld_count` 个块，取**最好的**那几个（按副露数裁剪）。
    # 排序键**只用搭子权重**（元组第二项是对子奖励，不参与排序），与旧 `weights.sort(reverse=True)`
    # 逐位等价（同为稳定排序、同插入序）。
    need = max(0, tiles.SETS_PER_HAND - meld_count - sets)
    blocks.sort(key=lambda item: item[0], reverse=True)
    taken = blocks[:need]
    # **超额对子奖励**：被裁掉的对子不再计 0，而是累加一个折扣值（`keep_extra_pairs`）。
    # 累加后**封顶**在 `PAIR_BONUS_CAP`（< 1）⇒ 只打破并列、不覆盖块数差（见 Docstring）。
    # 非对子块的奖励为 0，故不影响它们。
    if keep_extra_pairs:
        extra_pair_bonus = min(
            PAIR_BONUS_CAP * keep_extra_pairs,
            sum(bonus for _, bonus in blocks[need:]),
        )
    else:
        extra_pair_bonus = 0.0
    if not taken:
        return 2.0 * sets + extra_pair_bonus
    # **形质修正的两种算法**（`grade`）：
    # - `"mean"`（默认，逐位等于旧行为）：`0.9 × (均重 − 1)`。
    # - `"lex"`（实验档，2026-10-10 A）：`0.9 × Σ_i (w_i − 1) × 0.5^i`（`taken` 已按权重降序）。
    #
    # **为什么要 `lex`**：均值会**抹掉多重集的形状信息，甚至给出反向排序**。反例（同一 `need`）：
    # `(1.2, 1.2, 0.7)` 与 `(1.2, 1.0, 1.0)` —— 前者有**两个两面**、后者只有一个，
    # 但**均值**前者 1.033 ⇒ 修正 **+0.03**、后者 1.067 ⇒ **+0.06** ⇒ **均值偏好后者（错）**；
    # `lex` 给出 0.2025 vs 0.180 ⇒ 偏好前者（对）。这不是标定问题，是**聚合方式**问题。
    # **幅度仍在设计上界内**：`Σ 0.5^i ≤ 1 − 2^{−need} < 1`、`|w−1| ≤ 0.4`
    # ⇒ `|0.9 × Σ| < 0.36`，与旧式同级 ⇒ 仍然**只打破并列、不覆盖「块数差 1」**。
    if grade == "lex":
        correction = 0.9 * sum(
            (weight - 1.0) * (0.5 ** index) for index, (weight, _) in enumerate(taken)
        )
    else:
        correction = 0.9 * (sum(weight for weight, _ in taken) / len(taken) - 1.0)
    return 2.0 * sets + len(taken) + correction + extra_pair_bonus


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


def shape_mix(counts: Sequence[int], meld_count: int = 0) -> tuple[float, int, int]:
    """骨架里三类搭子的**数量**：``(两面 对子数, 愚形数)``。

    **为什么需要它**：`shape_value` 把「两面 1.2 / 对子 1.0 / 坎张 0.7」加权成一个标量，
    于是「2 両面 + 2 対子」（RiichiBook ch3 §3.4 的 **perfect 1-away**，双面听牌率 100%）
    与「1 両面 + 3 対子」可能得到相近的加权值，但前者的好型率高得多。
    本函数把**结构**暴露出来，供上层做精确的 perfect n-away 判定。

    **口径与近似**（刻意写清，避免被当精确好型率引用）：
    这是**结构性代理**，不是「枚举所有进张后统计落到两面听的加权占比」那个精确好型率——
    后者要对每个进张各算一次听口类型，实测成本会压到 v4 已有的 p99 851ms/1800ms 之上。
    分解顺序与 `shape_value` 同源（先刻子后顺子、两面优先于对子），所以两者可比。
    """
    real = list(counts)
    real[GOD] = 0
    runs = pairs = kanchan = 0
    for start, size in ((0, 9), (9, 9), (18, 9), (27, 7)):
        group = real[start : start + size]
        if start == 27:  # 字牌不能成顺
            for amount in group:
                pairs += (amount % tiles.SET_LENGTH) // 2
            continue
        for index in range(size):
            while group[index] >= tiles.SET_LENGTH:
                group[index] -= tiles.SET_LENGTH
        for index in range(size - tiles.RUN_LENGTH + 1):
            while group[index] and group[index + 1] and group[index + 2]:
                group[index] -= 1
                group[index + 1] -= 1
                group[index + 2] -= 1
        for index in range(size - 1):
            while group[index] and group[index + 1]:
                group[index] -= 1
                group[index + 1] -= 1
                runs += 1
        for index in range(size):
            while group[index] >= 2:
                group[index] -= 2
                pairs += 1
        for index in range(size - 2):
            while group[index] and group[index + 2]:
                group[index] -= 1
                group[index + 2] -= 1
                kanchan += 1
    need = max(0, tiles.SETS_PER_HAND - meld_count - 0)
    return float(runs), int(pairs), int(kanchan) if need else int(kanchan)


# ---------------------------------------------------------------------------
# **Cython 快核（可选，默认关）**——2026-10-08 23:50 A 接。
#
# 背景：`shanten_fast.pyx`（agent-b 2026-10-05 写并编译）与纯 Python 实现同 API，
# 本机已验证 **逐位一致**（`test_shanten_parity.py` 777/777；随机 300 手牌 300/300），
# 且 `shanten` 快 **3.6×**（0.850 → 0.234 ms/次）。
#
# **为什么默认关**：这条路径会替换**冠军档**的向听计算，属「会移动出牌」的改动。
# 纪律要求：不可默认生效、不可静默替换（`parity` 只覆盖 777 例，不是穷尽）。
# 开启方式：环境变量 `MAJIANG_SHANTEN_FAST=1`（用于离线扫描/对拍，把 57ms/决策 打下来）。
# 若要在冠军档默认启用，须先过：① `test_shanten_parity.py`；② 决策级逐位不变
# （改前/改后在真机决策点上 0 分歧，见 `/tmp/dump_v5.py`、`/tmp/dump_resp.py`）；③ 全量测试。
#
# 注意：`.pyx` 版本**不做入参校验**（纯 Python 版会对张数不符抛 `ShantenError`）。
# 决策热路径的入参本就合法，但离线脚本若靠异常做分支，开快核后会**静默走另一支**。
_FAST_ENABLED = os.environ.get("MAJIANG_SHANTEN_FAST", "0") not in ("0", "", "False", "false")
if _FAST_ENABLED:  # pragma: no cover - 环境相关
    try:
        from . import shanten_fast as _fast  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 —— 未编译/ABI 不符都退回纯 Python
        _fast = None  # type: ignore[assignment]
    if _fast is not None:
        # 直接**重绑模块级名字**：`shanten_any` / `best_shanten` / `ukeire` 内部引用的是
        # 全局名，重绑后它们的调用点自动走快核，无需改函数体。
        shanten = _fast.shanten  # type: ignore[assignment]
        best_shanten = _fast.best_shanten  # type: ignore[assignment]
        seven_pairs_shanten = _fast.seven_pairs_shanten  # type: ignore[assignment]
        ukeire = _fast.ukeire  # type: ignore[assignment]
