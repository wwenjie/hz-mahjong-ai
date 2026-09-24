"""官方口径对拍固化（tasks.md 2.13 / 2.14）。

以下用例于 2026-09-23 对官方平台（接入指南 v34）的 ``POST /portal/api/tools/fan-calc``
逐例实测，返回值原样固化于此，用于锁住规则引擎的判定口径。复跑对拍：

    uv run python tools/fan_calc_diff.py --cases 300

三处文档歧义由这批实测澄清：

1. **4 白板与爆头叠加**（§1.3 爆头行写的「正好 4 白板除外」不成立）。
2. **财神可作胡牌张**：摸到白板时按百搭补将成胡即算胡牌。
3. **动作链命名**：纯杠为「杠开」/「连杠×N」，纯飘为「财飘」/「双财飘」/「三财飘」/
   「连飘×N」，杠飘混合合并为「杠飘链×N」；「杠爆」不是独立标签，而是「杠开」+「爆头」。
"""

import pytest

from majiang.rules import fan as fan_module
from majiang.rules import score as score_module
from majiang.rules import tiles

from .helpers import counts_of

# (手牌, 摸牌, 动作链次数, 飘次数, 胡, 爆头, 总番, detail)
OFFICIAL_CASES: tuple[tuple[str, str, int, int, bool, bool, int, tuple[str, ...]], ...] = (
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", "5b", 0, 0, True, False, 1, ("平胡",)),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", "5b", 0, 0, True, True, 2, ("平胡", "爆头")),
    ("1w2w3w4w5w6w7w8w9w白白白白", "东", 0, 0, True, True, 4, ("平胡", "4个白板", "爆头")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", "白", 0, 0, True, False, 1, ("平胡",)),
    ("1w1w2w2w3w3w4w4w5w5w6w6w5b", "5b", 0, 0, True, False, 2, ("七对",)),
    ("1w1w1w1w2w2w3w3w4w4w5w5w6w", "6w", 0, 0, True, False, 4, ("豪华七对×1",)),
    ("1w1w1w1w2w2w2w2w3w3w4w4w5w", "5w", 0, 0, True, False, 8, ("豪华七对×2",)),
    ("1w1w1w1w2w2w2w2w3w3w3w3w白", "4w", 3, 3, True, True, 512,
     ("豪华七对×3", "三财飘", "4个白板", "爆头")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", "5b", 1, 0, True, False, 2, ("平胡", "杠开")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", "5b", 1, 0, True, True, 4, ("平胡", "杠开", "爆头")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", "5b", 2, 0, True, False, 4, ("平胡", "连杠×2")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", "5b", 6, 0, True, False, 64, ("平胡", "连杠×6")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", "5b", 2, 2, True, True, 8, ("平胡", "双财飘", "爆头")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", "5b", 3, 3, True, True, 32,
     ("平胡", "三财飘", "4个白板", "爆头")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", "5b", 4, 4, True, False, 32, ("平胡", "连飘×4", "4个白板")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", "5b", 4, 2, True, True, 32, ("平胡", "杠飘链×4", "爆头")),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", "9t", 0, 0, False, False, 0, ()),
)


@pytest.mark.parametrize(
    ("spec", "draw", "chain", "piao", "hu", "baotou", "fan", "detail"),
    OFFICIAL_CASES,
)
def test_matches_official_fan_calc(
    spec: str,
    draw: str,
    chain: int,
    piao: int,
    hu: bool,
    baotou: bool,
    fan: int,
    detail: tuple[str, ...],
) -> None:
    result = fan_module.compute_fan(
        counts_of(spec), tiles.parse(draw), 0, chain_count=chain, piao_count=piao
    )
    assert result.hu is hu
    assert result.baotou is baotou
    assert result.fan == fan
    assert result.detail == detail


def test_official_settlement_matches_at_maximum_fan() -> None:
    settlement = score_module.settle(512, 1)
    assert settlement.to_api_shape() == {
        "dealer_hu": {"win": 12288, "lose": [4096, 4096, 4096]},
        "nondealer_hu": {"win": 5120, "lose": [4096, 512, 512]},
    }


@pytest.mark.parametrize(
    ("chain", "piao", "expected"),
    [
        (0, 0, None),
        (1, 0, "杠开"),
        (2, 0, "连杠×2"),
        (6, 0, "连杠×6"),
        (1, 1, "财飘"),
        (2, 2, "双财飘"),
        (3, 3, "三财飘"),
        (4, 4, "连飘×4"),
        (3, 1, "杠飘链×3"),
        (4, 2, "杠飘链×4"),
        (6, 3, "杠飘链×6"),
    ],
)
def test_chain_labels_match_official(chain: int, piao: int, expected: str | None) -> None:
    assert fan_module.chain_label(chain - piao, piao) == expected


def test_god_may_serve_as_the_winning_tile() -> None:
    result = fan_module.compute_fan(
        counts_of("1w2w3w4w5w6w7w8w9w1b2b3b5b"), tiles.parse("白")
    )
    assert result.hu is True and result.fan == 1
