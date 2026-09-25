"""检验廉价进张估计（`cheap_ukeire`）是否可用于决策（tasks.md 5.4）。

**动机**：5.4 的附注记载「同向听改按进张排序在 480 局 A/B 中 22.1% vs 25.2%，无增益」，
据此把 ``tiebreak="ukeire"`` 关掉了。但那个档位用的是 ``cheap_ukeire``——**以
``quick_shanten``（贪心近似）代替精确向听**。若这个近似在排序上不可靠，那么当时测的
就是噪声，「进张排序无效」这个结论也就不成立，方向会被错误地关掉。

本工具直接在**真实的 1 向听手牌**上对比两个估计：

- ``cheap_ukeire``（1.1–1.6 ms）
- 精确 ``shanten.ukeire``（42–157 ms）

输出三件事：
1. 两者选出的「最佳打牌」是否一致
2. 不一致时，廉价选法的**精确进张张数**差多少（regret）
3. 若按精确进张排序，平均进张张数相对当前的提升

用法::

    uv run python tools/analyze_ukeire_fidelity.py --hands 300
"""

from __future__ import annotations

import argparse
import random
import sys
import statistics as stats
from collections import Counter

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.strategy.policy import cheap_ukeire


def random_hand(rng: random.Random, gods: int) -> list[int]:
    """随机发一副 **14 张**的暗手（含指定数量的财神）。

    必须是 14 张：正常回合是「摸完 14 张、打出一张回 13 张」，而 ``ukeire`` 要的正是
    打完之后的 13 张。若从 13 张里再扣一张得到 12 张，``ukeire`` 会返回空表——
    第一版就是这么写的，算出「平均少进张 −30 张」这种荒谬值才发现。
    """
    pool = [tile for tile in range(tiles.TILE_KINDS) if tile != tiles.GOD for _ in range(4)]
    rng.shuffle(pool)
    counts = [0] * tiles.TILE_KINDS
    for tile in pool[: 14 - gods]:
        counts[tile] += 1
    counts[tiles.GOD] = gods
    return counts


def near_tenpai_hand(rng: random.Random, gods: int) -> list[int]:
    """构造一副**几乎听牌**的手牌：拼 4 组面子 + 1 对将（严格 14 张），再把若干张换成财神。

    纯随机 14 张里「打完恰好 1 向听」的位置极少（实测 100 副只有 10 副），样本撑不起
    结论。构造法把命中率提上来，而且这些正是「进张排序决定胜负」的位置。

    每一步都检查余量（同种牌不超过 4 张），因此**总数恒为 14**——不做事后裁剪，
    裁剪会把张数改坏（第一版裁到 11 张，``shanten_any`` 直接抛错）。
    """
    counts = [0] * tiles.TILE_KINDS

    def place(tile: int, amount: int) -> bool:
        if tile == tiles.GOD or counts[tile] + amount > 4:
            return False
        counts[tile] += amount
        return True

    def place_all(tiles_needed: tuple[int, ...]) -> bool:
        """**先整体校验再落子**。用 ``all(place(...))`` 会因短路只落一半：
        一次失败的三连尝试留下 2 张残牌，另一个面子再落 3 张，总数就超过 14 张
        （实测报出 15 张）。"""
        if any(tile == tiles.GOD for tile in tiles_needed):
            return False
        need: dict[int, int] = {}
        for tile in tiles_needed:
            need[tile] = need.get(tile, 0) + 1
        if any(counts[tile] + amount > 4 for tile, amount in need.items()):
            return False
        for tile, amount in need.items():
            counts[tile] += amount
        return True

    for _ in range(3):
        for _attempt in range(50):
            if rng.random() < 0.5:
                start = rng.randrange(0, 27)
                if start % 9 > 6:
                    continue
                if place_all((start, start + 1, start + 2)):
                    break
            else:
                tile = rng.randrange(0, tiles.TILE_KINDS)
                if place_all((tile, tile, tile)):
                    break
    # 将牌
    for _attempt in range(50):
        tile = rng.randrange(0, tiles.TILE_KINDS)
        if place_all((tile, tile)):
            break
    # 一个搭子（两面）＋ 一张孤张 = 14 张。
    # **必须是 1 向听而不是听牌**：听牌时任何合法出牌都保留同一个听口，精确进张完全相同，
    # 测不出排序差异（第一版构造出 4 组面子 + 将，落到听牌上，量纲参考全是 0）。
    for _attempt in range(50):
        start = rng.randrange(0, 27)
        if start % 9 > 6:
            continue
        if place_all((start, start + 1)):
            break
    for _attempt in range(50):
        tile = rng.randrange(0, tiles.TILE_KINDS)
        if place_all((tile,)):
            break

    for _ in range(gods):
        for tile in range(tiles.TILE_KINDS):
            if tile != tiles.GOD and counts[tile] > 0:
                counts[tile] -= 1
                counts[tiles.GOD] += 1
                break
    return counts


def exact_ukeire(counts: list[int], visible: list[int] | None) -> tuple[int, int]:
    """精确进张。**不吞异常**——第一版用裸 except 返回 (0,0)，把「12 张牌非法」
    伪装成「精确进张为 0」，于是结论完全反过来。"""
    entries = shanten_module.ukeire(counts, 0, visible=visible)
    return len(entries), sum(copies for _, copies in entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检验廉价进张估计的排序可靠性")
    parser.add_argument("--hands", type=int, default=200, help="样本手牌数（每副约 0.5 秒）")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    agree = disagree = 0
    regrets: list[int] = []
    gains: list[int] = []
    skipped = Counter()
    positions = 0

    while positions < args.hands:
        counts = near_tenpai_hand(rng, gods=rng.choice([0, 0, 1, 1, 2]))
        candidates = [tile for tile in range(tiles.TILE_KINDS) if counts[tile] > 0 and tile != tiles.GOD]
        cheap_rank: list[tuple[int, int, int]] = []  # (cheap_copies, exact_copies, tile)
        for tile in candidates:
            after = list(counts)
            after[tile] -= 1
            # 只保留「打完恰好 0 或 1 向听」的位置。构造法（4 组面子 + 将）出来的
            # 手牌，扣一张正好是**听牌**（0 向听）——即「这一打决定听什么」的位置，
            # 正是「到听持平但胡得少」时最该检查的地方。
            state = shanten_module.shanten_any(after, 0)
            if state not in (0, 1):
                continue
            copies_c = cheap_ukeire(after)[1]
            copies_e = exact_ukeire(after, None)[1]
            cheap_rank.append((copies_c, copies_e, tile))
        if len(cheap_rank) < 2:
            skipped["候选不足"] += 1
            positions += 1
            continue
        positions += 1

        # **两个估计的数值量纲不同，不能互相减**（第一版就是这么写的，算出「少进张 −14 张」）。
        # 只能各自取 argmax，再在**精确口径**下比较两者的进张。
        by_cheap = max(cheap_rank, key=lambda item: (item[0], -item[2]))
        by_exact = max(cheap_rank, key=lambda item: (item[1], -item[2]))
        best_exact = max(item[1] for item in cheap_rank)
        worst_exact = min(item[1] for item in cheap_rank)

        if by_cheap[2] == by_exact[2]:
            agree += 1
        else:
            disagree += 1
            regrets.append(best_exact - by_cheap[1])
        gains.append(best_exact - worst_exact)

    total = agree + disagree
    print(f"1 向听样本 {total} 副（跳过 {dict(skipped)}）")
    if not total:
        print("  没有样本，检查位置筛选条件")
        return 1
    print(f"  廉价与精确选出同一张打牌: {agree}/{total} = {agree / total:.1%}")
    print(f"  不一致时平均少进张 {stats.mean(regrets):.2f} 张"
          f"（中位 {stats.median(regrets):.0f}，最大 {max(regrets)}）")
    print(f"  量纲参考：同一位置「最佳 vs 最差候选」的进张差 平均 {stats.mean(gains):.2f} 张")
    print()
    if disagree / total > 0.15 and stats.mean(regrets or [0]) >= 1:
        print("结论：**廉价估计不足以承载「按进张排序」这个决策**——"
              "5.4 附注的负结果很可能是噪声，值得用精确进张重测。")
    else:
        print("结论：廉价估计与精确一致到足以承载该决策，5.4 附注的负结果应当采信。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
