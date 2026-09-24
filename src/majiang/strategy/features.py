"""价值模型的特征提取（tasks.md 5.15 的价值函数路线）。

**数据生成与线上推理共用本模块**——这是刻意的：训练与推理若各写一套特征，就会产生
train-serve skew，模型在离线指标上好看、上线却失效。

特征只用**微秒级**的量（几百万个决策点的数据生成必须负担得起），且全部来自
**公开信息 + 本人手牌**（合规红线：不得使用对手暗牌）。

目标变量是**本局本人最终得分**（局分，零和），由模拟器给出——真实、精确、无需标注。
"""

from __future__ import annotations

from collections.abc import Sequence

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules.situation import Situation
from majiang.rules.tiles import GOD

SEATS = 4
HAND_KINDS = tiles.TILE_KINDS

FEATURE_NAMES: tuple[str, ...] = (
    "quick_shanten",
    "blocks_sets",
    "blocks_partials",
    "blocks_pair",
    "block_value",
    "gods_in_hand",
    "pair_kinds",
    "single_kinds",
    "seven_pairs_shanten",
    "tiles_in_hand",
    "meld_count",
    "chi_count",
    "gang_count",
    "wall_remaining",
    "draws_left",
    "progress",
    "is_dealer",
    "round_no",
    "catch_play",
    "i_am_restricted",
    "melds_0",
    "melds_1",
    "melds_2",
    "melds_3",
    "discards_0",
    "discards_1",
    "discards_2",
    "discards_3",
    "gods_seen",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def _meld_count(situation: Situation, seat: int) -> int:
    return len(situation.melds_for(seat))


def _discard_count(situation: Situation, seat: int) -> int:
    return len(situation.discards[seat]) if seat < len(situation.discards) else 0


def gods_seen(situation: Situation) -> int:
    """场上已现的财神数（他人弃牌 + 全部副露中的财神）。"""
    count = 0
    for group in situation.discards:
        count += sum(1 for tile in group if tile == GOD)
    for meld in situation.all_melds:
        count += sum(1 for tile in meld.tiles if tile == GOD)
    return count


def extract(situation: Situation) -> list[float]:
    """把局面压成定长特征向量。

    刻意**不含**「当前累计得分」：快照里的 ``scores`` 是**本局**记分板，局中恒为 0，
    不含任何信息（平台把跨局累计放在 ranking 而非快照里）。
    """
    counts = situation.hand.counts
    meld_count = situation.hand.meld_count
    sets, partials, pair = shanten_module.quick_blocks(counts)
    real = list(counts)
    real[GOD] = 0

    values: list[float] = [
        float(shanten_module.quick_shanten(counts, meld_count)),
        float(sets),
        float(partials),
        float(pair),
        float(2 * sets + partials),
        float(counts[GOD]),
        float(sum(1 for amount in real if amount >= 2)),
        float(sum(1 for amount in real if amount == 1)),
        float(shanten_module.seven_pairs_shanten(counts) if meld_count == 0 else 99),
        float(tiles.total_tiles(counts)),
        float(meld_count),
        float(sum(1 for meld in situation.hand.melds if meld.is_chi)),
        float(sum(1 for meld in situation.hand.melds if meld.is_gang)),
        float(situation.table.wall_remaining),
        float(situation.table.draws_left),
        float(situation.table.progress),
        1.0 if situation.table.dealer_seat == situation.seat else 0.0,
        float(situation.table.round_no),
        1.0 if situation.god.catch_play else 0.0,
        1.0 if situation.is_restricted else 0.0,
    ]
    values.extend(float(_meld_count(situation, seat)) for seat in range(SEATS))
    values.extend(float(_discard_count(situation, seat)) for seat in range(SEATS))
    values.append(float(gods_seen(situation)))
    assert len(values) == FEATURE_COUNT, (len(values), FEATURE_COUNT)
    return values


__all__ = ["FEATURE_COUNT", "FEATURE_NAMES", "extract", "gods_seen"]
