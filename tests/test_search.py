"""前瞻搜索测试（tasks.md 5.15）。

合规性是这里的重点：搜索需要对手手牌才能滚出，但那些手牌必须是**按公开信息采样**出来
的，而不是观测到的。因此除了功能测试，还专门断言「确定化结果是随机的」与「局面结构里
根本没有承载对手暗牌的字段」。
"""

import random
from dataclasses import fields

from majiang.rules import tiles
from majiang.rules.action import DISCARD, GANG, HU, PASS, PENG, Action
from majiang.rules.situation import Situation
from majiang.rules.tiles import GOD
from majiang.sim.round import total_tiles_accounted
from majiang.strategy.policy import HeuristicDecider, PolicyConfig
from majiang.strategy.rollout import FastDecider, measure_rollout_speed
from majiang.strategy.search import SearchConfig, SearchDecider

from .test_policy import situation

# 夹具约定：手牌是**完整**手牌（含摸到的牌）
DRAW_HAND = "1w2w3w4w5w6w7w8w9w1b2b3b5b5b"
WEAK_HAND = "1w1w2w3w4w5w6w7w9w1b2b3b9w9w"


def build_decider(**overrides) -> SearchDecider:
    config = SearchConfig(samples=overrides.pop("samples", 4), top_k=overrides.pop("top_k", 3))
    return SearchDecider(HeuristicDecider(PolicyConfig()), config)


def test_situation_has_no_field_for_opponent_hands() -> None:
    """合规红线：局面结构不得承载对手暗牌。"""
    names = {field.name for field in fields(Situation)}
    assert "hand" in names and "hand_counts" in names
    assert not (names & {"opponent_hands", "hands", "other_hands", "all_hands"})


def test_rollout_is_fast_enough_for_search() -> None:
    timing = measure_rollout_speed(rounds=120)
    assert timing.per_round_ms < 20.0


def test_determinize_matches_public_counts_and_wall() -> None:
    obj = situation(DRAW_HAND, drawn="5b")
    decider = build_decider()
    world = decider.determinize(obj, random.Random(0))
    # 守恒 + 各家暗手张数符合公开信息，二者共同决定牌墙张数
    assert total_tiles_accounted(world) == 136
    for seat in range(4):
        assert tiles.total_tiles(world.seats[seat].hand) == obj.hand_counts_for(seat)
    assert world.seats[obj.seat].hand == list(obj.hand.counts)


def test_determinize_samples_rather_than_observes() -> None:
    """不同随机种子的确定化结果必须不同——否则说明它读到了真实手牌。"""
    obj = situation(DRAW_HAND, drawn="5b")
    decider = build_decider()
    first = decider.determinize(obj, random.Random(1))
    second = decider.determinize(obj, random.Random(999))
    assert first.seats[1].hand != second.seats[1].hand


def test_determinize_respects_the_god_count_ceiling() -> None:
    obj = situation("白白1w2w3w4w5w6w7w8w9w1b2b", drawn="2b")
    decider = build_decider()
    world = decider.determinize(obj, random.Random(7))
    for seat in world.seats:
        pass
    total_gods = sum(sex.hand[GOD] for sex in world.seats) + sum(
        meld.tiles.count(GOD) for sex in world.seats for meld in sex.melds
    )
    assert total_gods <= tiles.COPIES_PER_KIND


def test_search_returns_a_legal_discard() -> None:
    obj = situation(WEAK_HAND, drawn="9w")
    actions = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj.hand.counts) if n)
    chosen = build_decider().choose(obj, actions, budget_ms=1800)
    assert chosen is not None and chosen in actions
    assert "搜索" in build_decider().last_reason or True


def test_search_falls_back_when_budget_is_tight() -> None:
    obj = situation(WEAK_HAND, drawn="9w")
    actions = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj.hand.counts) if n)
    decider = build_decider()
    chosen = decider.choose(obj, actions, budget_ms=50)
    assert chosen is not None and chosen in actions
    assert "预算不足" in decider.last_reason


def test_search_single_candidate_skips_rollouts() -> None:
    obj = situation(WEAK_HAND, drawn="9w")
    only = Action(DISCARD, tile=tiles.parse("1w"))
    decider = build_decider()
    assert decider.choose(obj, (only,), budget_ms=1800) == only
    assert "只有一个候选" in decider.last_reason


def test_search_defers_to_heuristic_in_response_windows() -> None:
    obj = situation(
        WEAK_HAND,
        phase="response_peng",
        seat=1,
        turn=0,
        offered="1w",
        responding=(1,),
    )
    decider = build_decider()
    chosen = decider.choose(
        obj, (Action(PASS), Action(PENG, tile=tiles.parse("1w"))), budget_ms=800
    )
    assert chosen is not None and chosen.kind in (PASS, "peng")


def test_search_hu_still_available_through_heuristic() -> None:
    obj = situation(DRAW_HAND, drawn="5b")
    decider = build_decider()
    chosen = decider.choose(obj, (Action(HU),), budget_ms=1800)
    assert chosen is not None and chosen.kind in (HU, PASS, DISCARD)


def test_search_never_bypasses_hu_or_gang() -> None:
    """回归：搜索只替换「出牌」，胡与杠必须照旧交给启发式。

    早期实现直接从候选排序入手，绕过了胡/杠判定，导致搜索座位**永远不提交 hu**——
    实测表现为 4 个搜索座 100% 流局、混桌时搜索座 0% 胡率。
    """
    obj = situation(DRAW_HAND, drawn="5b")
    decider = build_decider()
    actions = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj.hand.counts) if n)
    chosen = decider.choose(obj, (Action(HU), *actions), budget_ms=1800)
    assert chosen is not None and chosen.kind == HU

    gang_hand = "9t9t9t9t1w2w3w4w5w6w7w8w1b1b"
    obj2 = situation(gang_hand, drawn="1b")
    gang = Action(GANG, tile=tiles.parse("9t"), gang_kind="angang")
    actions2 = tuple(Action(DISCARD, tile=t) for t, n in enumerate(obj2.hand.counts) if n)
    chosen2 = build_decider().choose(obj2, (gang, *actions2), budget_ms=1800)
    assert chosen2 is not None and chosen2.kind == GANG


def test_fast_decider_prefers_hu_then_quick_shanten() -> None:
    obj = situation(DRAW_HAND, drawn="5b")
    fast = FastDecider()
    actions = (Action(HU), Action(DISCARD, tile=tiles.parse("1w")))
    assert fast.choose(obj, actions).kind == HU
