"""番型倍率与番型详情。

总番 = 分支因子 × 2^动作链次数 × (4 白板 ×2) × (爆头 ×2)

平台 §1.3 表格里的「杠开 ×2 / 杠爆 ×4 / 财飘 ×4 / 双财飘 ×8 / 三财飘 ×16」与上式
等价，不需要为每个组合单独建表——它们只是同一公式在不同动作链下的读数：

- 杠开 = 动作链 1 次（杠）
- 杠爆 = 爆头 × 链 1
- 财飘 = 爆头 × 飘 1 次；双财飘 / 三财飘即飘 2 / 3 次

文档给出的两个上限也吻合，可作为回归用例：
- 平胡分支：3 连杠 + 三财飘 + 爆头 + 4 白板 = 2^6 × 2 × 2 = 256
- 全局最大：三豪华七对 + 三财飘 + 4 白板 + 爆头 = 16 × 2^3 × 2 × 2 = 512
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import tiles
from . import win
from .tiles import GOD

CHAIN_MAX = 6
FOUR_GODS_TOTAL = 4
FOUR_GODS_BONUS = 2
BAOTOU_BONUS = 2

# 已由官方番型端点实测确认（tasks.md 2.14）：4 白板与爆头叠加，§1.3 爆头行写的
# 「正好 4 白板除外」不成立。实测：手牌 1w2w3w4w5w6w7w8w9w 加四张白板、摸东 →
# fan=4，detail=['平胡','4个白板','爆头']。开关保留以便应对平台口径变更。
BAOTOU_STACKS_WITH_FOUR_GODS = True

_PIAO_NAMES: dict[int, str] = {1: "财飘", 2: "双财飘", 3: "三财飘"}


def chain_label(gang_count: int, piao_count: int) -> str | None:
    """动作链的官方命名口径。

    与官方番型计算端点实测一致：纯杠为「杠开」/「连杠×N」，纯飘为「财飘」/
    「双财飘」/「三财飘」/「连飘×N」，杠飘混合则合并为「杠飘链×N」（N 为链总次数）。
    注意「杠爆」不是独立标签——爆头状态下杠开在 detail 里就是「杠开」+「爆头」两项。
    """
    chain = gang_count + piao_count
    if chain == 0:
        return None
    if piao_count == 0:
        return "杠开" if gang_count == 1 else f"连杠×{gang_count}"
    if gang_count == 0:
        return _PIAO_NAMES.get(piao_count, f"连飘×{piao_count}")
    return f"杠飘链×{chain}"


class FanError(ValueError):
    """番型入参非法。"""


@dataclass(frozen=True, slots=True)
class FanResult:
    hu: bool
    fan: int
    baotou: bool
    branch: int
    branch_name: str
    chain_count: int
    piao_count: int
    gang_count: int
    four_gods: bool
    detail: tuple[str, ...]


def compute_fan(
    waiting_counts: Sequence[int],
    draw: int,
    meld_count: int = 0,
    *,
    chain_count: int = 0,
    piao_count: int = 0,
    baotou: bool | None = None,
    is_gang_draw: bool = False,
    you_cai_bi_kao: bool = False,
    baotou_stacks_with_four_gods: bool = BAOTOU_STACKS_WITH_FOUR_GODS,
) -> FanResult:
    """计算一次自摸胡牌的番型。

    ``waiting_counts`` 是摸牌前的暗手牌（未摸牌张数），``draw`` 是本次摸到的牌。
    爆头按摸牌前的听牌态判定；4 白板按胡牌时的「手留白 + 链内飘出」判定。
    """
    if not 0 <= chain_count <= CHAIN_MAX:
        raise FanError(f"动作链次数应在 0–{CHAIN_MAX}，实际 {chain_count}")
    if not 0 <= piao_count <= chain_count:
        raise FanError(f"飘次数应不超过动作链次数，实际 piao={piao_count} chain={chain_count}")
    if not 0 <= draw < tiles.TILE_KINDS:
        raise FanError(f"非法摸牌牌索引: {draw}")

    gang_count = chain_count - piao_count
    drawn = list(waiting_counts)
    drawn[draw] += 1
    if drawn[GOD] + piao_count > FOUR_GODS_TOTAL:
        raise FanError(
            f"手留白 {drawn[GOD]} 加飘出 {piao_count} 超过全场财神总数 {FOUR_GODS_TOTAL}"
        )

    baotou_flag = win.is_baotou(waiting_counts, meld_count) if baotou is None else baotou
    branch = win.best_branch(drawn, meld_count)
    four_gods = drawn[GOD] + piao_count == FOUR_GODS_TOTAL

    if branch is None:
        return FanResult(
            hu=False,
            fan=0,
            baotou=baotou_flag,
            branch=0,
            branch_name="",
            chain_count=chain_count,
            piao_count=piao_count,
            gang_count=gang_count,
            four_gods=four_gods,
            detail=(),
        )

    if you_cai_bi_kao and drawn[GOD] > 0 and not (baotou_flag or is_gang_draw):
        return FanResult(
            hu=False,
            fan=0,
            baotou=baotou_flag,
            branch=branch.fan,
            branch_name=branch.name,
            chain_count=chain_count,
            piao_count=piao_count,
            gang_count=gang_count,
            four_gods=four_gods,
            detail=("有财必拷响：持财神时不可平胡",),
        )

    fan = branch.fan
    detail: list[str] = [branch.name]
    label = chain_label(gang_count, piao_count)
    if label is not None:
        detail.append(label)
    fan <<= chain_count
    if four_gods:
        fan *= FOUR_GODS_BONUS
        detail.append("4个白板")
    if baotou_flag and (baotou_stacks_with_four_gods or not four_gods):
        fan *= BAOTOU_BONUS
        detail.append("爆头")

    return FanResult(
        hu=True,
        fan=fan,
        baotou=baotou_flag,
        branch=branch.fan,
        branch_name=branch.name,
        chain_count=chain_count,
        piao_count=piao_count,
        gang_count=gang_count,
        four_gods=four_gods,
        detail=tuple(detail),
    )
