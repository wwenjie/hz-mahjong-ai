"""对手听牌模型的**局面特征**（tasks.md 6B）。

与价值模型的关键差异：这里的特征描述的是**某一个对手的公开足迹**（他吃碰杠了什么、
弃了哪些牌、还剩多少暗手），观察者只能看到这些——这正是「对手是否听牌」这件事唯一
可用的推断依据。

合规红线：特征只来自公开信息 + 本人手牌，**不含任何对手暗牌**。

数据生成与线上推理共用本模块（避免 train-serve skew）。
"""

from __future__ import annotations

from collections.abc import Sequence

from majiang.rules import tiles
from majiang.rules.situation import Situation
from majiang.rules.tiles import GOD

SEATS = 4
MIDDLE_RANKS = frozenset({2, 3, 4, 5, 6, 7, 8})

FEATURE_NAMES: tuple[str, ...] = (
    "target_melds",
    "target_chi",
    "target_peng",
    "target_gang",
    "target_discards",
    "target_suit_concentration",
    "target_middle_share",
    "target_recent_same_suit",
    "target_discarded_god",
    "target_concealed",
    "target_is_dealer",
    "target_restricted",
    "target_discard_rate",
    "wall_remaining",
    "draws_left",
    "progress",
    "gods_seen",
    "table_melds",
    "my_concealed",
    "my_melds",
    "my_gods",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def _suit_concentration(discards: Sequence[int]) -> float:
    if not discards:
        return 0.0
    buckets = [0, 0, 0, 0]  # 万 / 筒 / 条 / 字
    for tile in discards:
        if tiles.is_number(tile):
            buckets[tiles.suit(tile)] += 1
        else:
            buckets[3] += 1
    return max(buckets) / len(discards)


def _middle_share(discards: Sequence[int]) -> float:
    if not discards:
        return 0.0
    middle = sum(
        1 for tile in discards if tiles.is_number(tile) and tiles.rank(tile) in MIDDLE_RANKS
    )
    return middle / len(discards)


def _recent_same_suit(discards: Sequence[int], window: int = 3) -> float:
    recent = discards[-window:]
    if len(recent) < 2:
        return 0.0
    suits = [tiles.suit(tile) if tiles.is_number(tile) else 3 for tile in recent]
    return float(max(suits.count(value) for value in set(suits)))


def extract(situation: Situation, target: int) -> list[float]:
    """从 ``situation``（观察者视角）提取目标座位 ``target`` 的特征。"""
    discards = situation.discards[target] if target < len(situation.discards) else ()
    melds = situation.melds_for(target)
    concealed = situation.hand_counts_for(target)
    hands = situation.hand  # 观察者本人暗牌

    meld_total = len(melds)
    elapsed_turns = max(1.0, situation.table.draws_made / SEATS)
    values: list[float] = [
        float(meld_total),
        float(sum(1 for meld in melds if meld.is_chi)),
        float(sum(1 for meld in melds if meld.is_peng)),
        float(sum(1 for meld in melds if meld.is_gang)),
        float(len(discards)),
        _suit_concentration(discards),
        _middle_share(discards),
        _recent_same_suit(discards),
        1.0 if any(tile == GOD for tile in discards) else 0.0,
        float(concealed),
        1.0 if situation.table.dealer_seat == target else 0.0,
        1.0 if situation.god.restricts(target) else 0.0,
        len(discards) / elapsed_turns,
        float(situation.table.wall_remaining),
        float(situation.table.draws_left),
        float(situation.table.progress),
        float(sum(1 for group in situation.discards for tile in group if tile == GOD)),
        float(len(situation.all_melds)),
        float(tiles.total_tiles(hands.counts)),
        float(hands.meld_count),
        float(hands.god_count),
    ]
    assert len(values) == FEATURE_COUNT, (len(values), FEATURE_COUNT)
    return values


def opponents_of(seat: int, seats: int = SEATS) -> tuple[int, ...]:
    return tuple(other for other in range(seats) if other != seat)


__all__ = ["FEATURE_COUNT", "FEATURE_NAMES", "extract", "opponents_of"]
