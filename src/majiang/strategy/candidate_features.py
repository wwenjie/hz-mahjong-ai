"""Per-候选向听变化特征（Mortal keep/next_shanten_discards 通道的 GBDT 标量化原型）。

**状态：原型，未接入** ``features.py``。刻意做成独立模块、不动 ``FEATURE_NAMES``：
现有价值模型按 29 维训练，改特征数即 train-serve skew；是否接入由 A 评审后决定。

**来源**：Mortal ``libriichi/src/state/obs_repr.rs`` 把「每张候选弃牌打了之后向听怎么变」
编成显式通道（keep_shanten_discards / next_shanten_discards /
unconditional_tenpai）。我们的特征表只有手牌整体向听，缺这一层——而 C 的证据链
（C16←C27←C28）把与强 bot 的差距定位在**同向听层内的进张/留牌决策**，正是这类
特征的管辖面。

**成本约束**：决策热路径 p99 预算 1800ms（v5 实测 860ms），所以这里一律用
``quick_shanten``（微秒级骨架版）而**不**用精确的 ``shanten``/``best_shanten``。
口径是近似（骨架块分解，不枚举个位数进张），与 ``quick_shanten`` 在 v4 次排序里
的既有用法同级——够分辨「打了掉不掉向听」，不够算精确进张数（那是 ukeire 的活，
单样本 0.1–0.2s，进不了特征）。
"""

from __future__ import annotations

from collections.abc import Sequence

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles

FEATURE_NAMES: tuple[str, ...] = (
    "cand_keep_shanten",   # 打后不升向听的候选数（灵活性：越多越不伤结构）
    "cand_lower_shanten",  # 打后能降低向听的候选数（攻击性：多出现在摸后多一张的牌型）
    "cand_raise_shanten",  # 打后向听变差的候选数（陷阱牌：打了就退）
)
FEATURE_COUNT = len(FEATURE_NAMES)


def extract(counts: Sequence[int], meld_count: int = 0) -> list[float]:
    """对摸牌后（多一张）或正常手牌，统计每个可打候选对向听的影响。

    基线 = 当前 ``quick_shanten``；逐候选打出后重算 ``quick_shanten`` 并分桶。
    张数不符（非 target / target+1）时按 ``shanten_any`` 同款语义处理：
    多一张状态下的「当前向听」取各候选打后向听的最小值。
    """
    work = list(counts)
    target = tiles.HAND_SIZE - tiles.MELD_SLOTS * meld_count
    total = tiles.total_tiles(counts)
    if total not in (target, target + 1):
        raise shanten_module.ShantenError(
            f"手牌张数 {total} 与 {meld_count} 组副露不符（应为 {target} 或 {target + 1}）"
        )

    per_discard: list[int] = []
    for tile in range(tiles.TILE_KINDS):
        if work[tile] == 0:
            continue
        work[tile] -= 1
        try:
            per_discard.append(shanten_module.quick_shanten(work, meld_count))
        finally:
            work[tile] += 1

    baseline = min(per_discard) if total == target + 1 else (
        shanten_module.quick_shanten(work, meld_count)
    )
    keep = sum(1 for s in per_discard if s == baseline)
    lower = sum(1 for s in per_discard if s < baseline)
    raise_ = sum(1 for s in per_discard if s > baseline)
    return [float(keep), float(lower), float(raise_)]


__all__ = ["FEATURE_COUNT", "FEATURE_NAMES", "extract"]
