"""启发式决策测试。"""

from majiang.client.snapshot import Snapshot
from majiang.rules import tiles
from majiang.rules.action import CHI, DISCARD, GANG, HU, PASS, PENG, Action, legal_actions
from majiang.rules.tiles import GOD
from majiang.strategy import risk
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

MY_SEAT = 0
DEALER_SEAT = 1


def codes(spec: str) -> list[str]:
    out: list[str] = []
    pending = ""
    for char in spec:
        if pending:
            out.append(pending + char)
            pending = ""
        elif char.isdigit():
            pending = char
        else:
            out.append(char)
    assert not pending, spec
    return out


def situation(
    hand: str,
    *,
    phase: str = "draw",
    seat: int = MY_SEAT,
    turn: int = MY_SEAT,
    drawn: str | None = None,
    offered: str | None = None,
    responding: tuple[int, ...] = (),
    wall: int = 74,
    dealer: int = DEALER_SEAT,
    discards: dict[int, str] | None = None,
    melds: dict[int, list[dict]] | None = None,
    god: dict | None = None,
):
    """构造局面夹具。

    约定：``hand`` 是**完整的**手牌，若给了 ``drawn`` 则它必须已经包含在 ``hand`` 里
    ——平台就是「摸牌并入 my_hand，另用 drawn_tile 标注哪一张是刚摸的」。搞错这一点
    会让策略多减一张牌，手牌张数与副露数不再相容。
    """
    if drawn:
        assert drawn in codes(hand), "my_hand 必须包含摸到的牌"
    raw = {
        "game_id": "g_1",
        "phase": phase,
        "round_no": 1,
        "dealer": dealer,
        "turn": turn,
        "waited_seat": turn,
        "wall_remaining": wall,
        "discards": [
            codes((discards or {}).get(seat_index, "")) for seat_index in range(4)
        ],
        "melds": [(melds or {}).get(seat_index, []) for seat_index in range(4)],
        "hand_counts": [len(codes(hand))] * 4,
        "my_hand": codes(hand),
        "seat": seat,
        "last_discard": offered or "",
        "drawn_tile": drawn or "",
        "god": god
        or {"baotou": False, "chain_count": 0, "catch_play": False, "god_discarder_seat": -1},
        "scores": [0, 0, 0, 0],
    }
    if phase.startswith("response"):
        raw["responding_seats"] = list(responding)
        raw["window_deadline_ms"] = 0
    return Snapshot.parse(raw).to_situation()


def decide(situation_obj, actions=None, decider=None) -> Action:
    decider = decider or HeuristicDecider()
    actions = actions if actions is not None else (Action(DISCARD, tile=t) for t in _discardables(situation_obj))
    return decider.choose(situation_obj, tuple(actions), budget_ms=1500)


def _discardables(situation_obj) -> list[int]:
    return [tile for tile, amount in enumerate(situation_obj.hand.counts) if amount]


def test_discard_prefers_lower_shanten() -> None:
    obj = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b")
    decider = HeuristicDecider()
    chosen = decider.choose(
        obj, tuple(Action(DISCARD, tile=t) for t in _discardables(obj)), budget_ms=1500
    )
    assert chosen is not None and chosen.kind == DISCARD
    assert tiles.to_code(chosen.tile) == "9b"
    assert decider.last_detail["discards"]
    assert "向听=0" in decider.last_detail["discards"][0]


def test_never_discards_god_when_alternative_exists() -> None:
    obj = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b白", drawn="白")
    decider = HeuristicDecider()
    actions = tuple(Action(DISCARD, tile=t) for t in _discardables(obj))
    chosen = decider.choose(obj, actions, budget_ms=1500)
    assert chosen is not None and chosen.tile != GOD


def test_takes_concealed_gang_when_it_does_not_set_back() -> None:
    obj = situation("9t9t9t9t1w2w3w4w5w6w7w8w1b1b", drawn="1b")
    decider = HeuristicDecider()
    actions = tuple(Action(DISCARD, tile=t) for t in _discardables(obj)) + (
        Action(GANG, tile=tiles.parse("9t"), gang_kind="angang"),
    )
    chosen = decider.choose(obj, actions, budget_ms=1500)
    assert chosen is not None and chosen.kind == GANG
    assert "杠" in decider.last_reason


def test_gang_skipped_in_reserved_tail() -> None:
    obj = situation("9t9t9t9t1w2w3w4w5w6w7w8w1b1b", drawn="1b", wall=20)
    decider = HeuristicDecider()
    actions = tuple(Action(DISCARD, tile=t) for t in _discardables(obj)) + (
        Action(GANG, tile=tiles.parse("9t"), gang_kind="angang"),
    )
    chosen = decider.choose(obj, actions, budget_ms=1500)
    assert chosen is not None and chosen.kind == DISCARD


BAOTOU_WITH_TWO_GODS = "1w2w3w4w5w6w7w8w9w1b2b3b白白"


def test_piao_when_survival_is_high() -> None:
    obj = situation(BAOTOU_WITH_TWO_GODS, drawn="白", wall=74)
    decider = HeuristicDecider()
    chosen = decider.choose(
        obj,
        (Action(HU), *(Action(DISCARD, tile=t) for t in _discardables(obj))),
        budget_ms=1500,
    )
    assert chosen is not None and chosen.kind == DISCARD and chosen.tile == GOD
    assert decider.last_detail["survival"] > decider.last_detail["threshold"]
    assert "弃胡飘" in decider.last_reason


# 续飘成立、但摸到的牌**不是**财神的一手牌（40 万副随机手牌里筛出来的 6 个之一）：
# 6 对 + 2 张财神。waiting（打掉摸进的 6w）= 5 对 + 1 张单 + 2 张财神，是爆头；
# 再把一张财神也打掉（五对 + 单张 + 1 张财神）**仍是**爆头 —— 所以 `_piao_candidate`
# 认定「可以续飘」，而抓打圈内合法出牌只有刚摸到的 6w。
PIAO_BUT_DRAWN_NOT_GOD = "6w6w8w8w1b1b4b4b7b7b3t3t白白"


def test_piao_never_discards_god_inside_catch_play_circle() -> None:
    """规则合规（B' 的 D1）：抓打圈内只能打刚摸到的那张，弃胡飘不得自建「打财神」。

    原先该分支自建 `Action(DISCARD, tile=GOD)`、**绕过 `legal_actions`**，而这里摸的是
    `6w`（不是财神），合法集只有 `[discard:6w, hu]` ⇒ 真机必被 409 INVALID_ACTION 拒掉、
    白费一个出牌窗口。这道闸门的作用是保证提交的动作合法，不是调参。
    """
    obj = situation(
        PIAO_BUT_DRAWN_NOT_GOD,
        drawn="6w",
        wall=74,
        seat=1,
        turn=1,
        god={
            "baotou": False,
            "chain_count": 0,
            "catch_play": True,
            "god_discarder_seat": 0,  # 打财神的是他家，因此本座受抓打圈限制
        },
    )
    actions = legal_actions(obj)
    assert obj.is_restricted
    assert GOD not in [a.tile for a in actions if a.kind == DISCARD]
    decider = HeuristicDecider()
    chosen = decider.choose(obj, actions, budget_ms=1500)
    assert chosen is not None
    assert chosen in actions, f"决策器返回了合法集之外的动作: {chosen}"
    assert chosen.kind == HU  # 飘不成，退回到胡
    # 日志必须说清「被抓打圈挡住」而不是「没算过阈值」——真机靠 reason 复核行为
    assert "抓打圈" in decider.last_reason and decider.last_detail["piao_blocked_by"] == "catch-play"


def test_piao_still_allowed_when_the_drawn_tile_is_the_god() -> None:
    """抓打圈内**摸到的正好是财神**时，打财神是合法的——守卫不能一票否决这种情况。"""
    obj = situation(
        PIAO_BUT_DRAWN_NOT_GOD,
        drawn="白",
        wall=74,
        seat=1,
        turn=1,
        god={
            "baotou": False,
            "chain_count": 0,
            "catch_play": True,
            "god_discarder_seat": 0,
        },
    )
    actions = legal_actions(obj)
    assert obj.is_restricted and Action(DISCARD, tile=GOD) in actions
    decider = HeuristicDecider()
    chosen = decider.choose(obj, actions, budget_ms=1500)
    assert chosen is not None and chosen.kind == DISCARD and chosen.tile == GOD
    assert "弃胡飘" in decider.last_reason


def test_hu_when_survival_is_low() -> None:
    crowded = {1: "1w1w1w2w2w2w3w3w3w4w4w4w5w", 2: "1b1b1b2b2b2b3b3b3b4b4b4b5b", 3: "1t1t1t2t2t2t3t3t3t4t4t4t5t"}
    obj = situation(BAOTOU_WITH_TWO_GODS, drawn="白", wall=25, discards=crowded)
    decider = HeuristicDecider()
    chosen = decider.choose(
        obj,
        (Action(HU), *(Action(DISCARD, tile=t) for t in _discardables(obj))),
        budget_ms=1500,
    )
    assert chosen is not None and chosen.kind == HU
    assert "胡牌" in decider.last_reason


def test_final_mode_is_more_willing_to_piao() -> None:
    obj = situation(BAOTOU_WITH_TWO_GODS, drawn="白", wall=40)
    actions = (Action(HU), *(Action(DISCARD, tile=t) for t in _discardables(obj)))
    qualifier = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
    final = HeuristicDecider(PolicyConfig.for_mode(Mode.FINAL))
    qualifier.decide = None  # type: ignore[attr-defined]
    chosen_q = qualifier.choose(obj, actions, budget_ms=1500)
    chosen_f = final.choose(obj, actions, budget_ms=1500)
    assert qualifier.last_detail["threshold"] >= final.last_detail["threshold"]
    assert chosen_q is not None and chosen_f is not None


# 一手很散、且「南/西/北」三张单张字牌**完全等价**的牌（东东是对子，故东不同价）：
# 三张的向听、骨架构型、静态喂牌值全相同 ⇒ `total` 逐位相等，是纯粹的三方并列。
# 注意 `discards` 让**北**的已见张数最多（1 家打过两张），这才有可分辨的方向。
TIE_THREE_HONORS = "1w4w7w1b4b7b1t4t7t东东南西北"


def test_default_tiebreak_falls_back_to_the_lowest_index() -> None:
    """**把现状钉住**：并列时默认走 `sorted(reverse=True)` 的稳定性 ⇒ 永远选最小牌索引。

    这不是设计决定而是实现副作用——棋牌花色的索引顺序（万 0-8 / 筒 9-17 / 条 18-26 /
    字 27-33）让它变成一个**系统性的花色偏置**（B' 量到 万 0.54 / 筒 0.28 / 字 0.22 / 条 0.08）。
    这条测试的作用是：一旦有人无意改了并列行为，它先响。
    """
    obj = situation(TIE_THREE_HONORS, drawn="北", discards={1: "北北"})
    decider = HeuristicDecider()
    chosen = decider.choose(obj, legal_actions(obj), budget_ms=1000)
    assert chosen is not None and chosen.kind == DISCARD
    assert tiles.to_code(chosen.tile) == "南"
    # 三张确实同价（否则这条测试测的不是并列）
    totals = {
        row.split()[0]: row.split("合计=")[1] for row in decider.last_detail["discards"][:3]
    }
    assert totals["南"] == totals["西"] == totals["北"]


def test_safe_tiebreak_prefers_the_most_visible_of_equal_candidates() -> None:
    """`safe_tiebreak` 把并列规则显式写成「同分优先打已见张最多的那张」。

    依据是**安全性有确定方向**：一张已经被人打过两张的牌，别人手里的同牌更少 ⇒ 更不可能
    是他等的那张。它同时天然抹掉花色索引偏置（已见张数与花色无关）。
    """
    obj = situation(TIE_THREE_HONORS, drawn="北", discards={1: "北北"})
    config = PolicyConfig.for_mode(Mode.QUALIFIER, safe_tiebreak=True)
    chosen = HeuristicDecider(config).choose(obj, legal_actions(obj), budget_ms=1000)
    assert chosen is not None and chosen.kind == DISCARD
    assert tiles.to_code(chosen.tile) == "北"


def test_safe_tiebreak_cannot_override_a_non_tied_candidate() -> None:
    """**改动面的边界**：它只重排 `total` 完全相等的候选，压不过任何一项。

    这是它与已被判死的 `feed_visibility` 的关键差别——那个把 `(4−seen)/4` 乘进喂牌项**本身**，
    于是能越过形质改变排序。这里用手牌分辨力不同的两张牌验证：改 safe_tiebreak 不改变选择。
    """
    obj = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b", discards={1: "9b9b"})
    plain = HeuristicDecider()
    safe = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER, safe_tiebreak=True))
    a = plain.choose(obj, legal_actions(obj), budget_ms=1000)
    b = safe.choose(obj, legal_actions(obj), budget_ms=1000)
    assert a is not None and b is not None and a.tile == b.tile


def test_unified_score_uses_points_and_kills_the_feed_knob() -> None:
    """统一期望得分（B' 设计说明 / A 实现）的三条结构性断言。

    ① 前两项 `P_win×E_pay − (1−P_win)×P_opp×E_loss` **就是** `routes.evaluate` 的 `value`
       ⇒ 档位的 `total` 应当≈该 `value` 减喂牌代价，量纲=分；
    ② `value_weight` 不再参与（拍出来的 10 被去掉）；
    ③ `feed_weight` 不再参与 —— 改它对 `total` **完全无影响**（喂牌已从旋钮变推导量）。
    """
    obj = situation("2w3w4w5w6w7w8w9w5b6b2b2b9b1t", drawn="1t")
    plain = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER, unified_score=True))
    plain.choose(obj, legal_actions(obj), budget_ms=2000)
    rows = {row.split()[0]: row for row in plain.last_detail["discards"]}
    assert rows, "应给出候选明细"

    # ③ 换 feed_weight(3.0→9.0) 与 value_weight(10→1) 都不改变统一得分的排序
    for kwargs in ({"feed_weight": 9.0}, {"value_weight": 1.0}):
        tweaked = HeuristicDecider(
            PolicyConfig.for_mode(Mode.QUALIFIER, unified_score=True, **kwargs)
        )
        tweaked.choose(obj, legal_actions(obj), budget_ms=2000)
        assert tweaked.last_detail["discards"] == plain.last_detail["discards"], kwargs

    top = plain.last_detail["discards"][0]
    assert "合计=" in top


def test_unified_score_orders_candidates_differently_from_v3() -> None:
    """**空干预防线的单元测试部分**：两个档位的排序键必须真的不同（结构性、不靠运气）。

    ⚠ 真正的「改动量」判据是**分歧率**，它必须用真机决策点量（`tools/divergence_gate.py`，
    判据 <5% 即空干预）——**不要**用少数几个手工构造的局面去充当这道门：
    初版本测试就是拿 3 个手工局面断言「必须有分歧」，结果 3 个全相同（那些局面本来只有一个
    合理出牌），测试失败而改动其实有效。手工局面只能验**结构**，不能验**分布**。
    """
    v3 = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER, tiebreak="tenpai-only",
                                               wait_aware_tenpai=True))
    uni = HeuristicDecider(
        PolicyConfig.for_mode(Mode.QUALIFIER, tiebreak="tenpai-only", wait_aware_tenpai=True,
                              unified_score=True)
    )
    obj = situation("1w4w7w2w5w8w3b6b9b1t2t3t白白", drawn="白")
    actions = legal_actions(obj)
    v3.choose(obj, actions, budget_ms=2000)
    uni.choose(obj, actions, budget_ms=2000)
    # 排序键不同 ⇒ 候选的评分明细必然不同（v3 是 −10×向听+形质−喂牌；统一是分）
    assert v3.last_detail["discards"] != uni.last_detail["discards"]
    assert "合计=" in uni.last_detail["discards"][0]


def test_tenpai_only_tiebreak_keeps_wait_aware_but_yields_shanten1() -> None:
    """`tiebreak="tenpai-only"` 的边界：听牌态仍按可见听口选（v3 的机制），向听 ≥1 不重排。"""
    # 听牌态：与 v3 的听牌选牌一致
    tenpai = situation("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b")
    a = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER, tiebreak="tenpai-only",
                                               wait_aware_tenpai=True))
    b = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER, tiebreak="exact-ukeire",
                                               wait_aware_tenpai=True))
    pick_a = a.choose(tenpai, legal_actions(tenpai), budget_ms=2000)
    pick_b = b.choose(tenpai, legal_actions(tenpai), budget_ms=2000)
    assert pick_a is not None and pick_b is not None and pick_a.tile == pick_b.tile
    # 向听 ≥1：不再调用进张次排序（明细里没有 tiebreak 字段）
    high = situation("1w4w7w2w5w8w3b6b9b1t2t3t白白", drawn="白")
    only = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER, tiebreak="tenpai-only",
                                                  wait_aware_tenpai=True))
    only.choose(high, legal_actions(high), budget_ms=2000)
    assert "tiebreak" not in only.last_detail


def test_peng_accepted_when_meld_route_wins() -> None:    # 对子很少（七对很远）、副露更近：碰把向听 3 推到 2，副露路线期望反超七对
    obj = situation(
        "2w4w5w5w9w2b5b6b3t4t5t北中",
        phase="response_peng",
        seat=1,
        turn=0,
        offered="5w",
        responding=(1,),
    )
    decider = HeuristicDecider(PolicyConfig(route_aware=True))
    chosen = decider.choose(
        obj,
        (Action(PASS), Action(PENG, tile=tiles.parse("5w"))),
        budget_ms=800,
    )
    assert chosen is not None and chosen.kind == PENG
    assert "副露路线期望" in decider.last_reason


def test_peng_declined_when_pair_route_is_clearly_better() -> None:
    """6 对在手：七对已是 0 向听（番 2），碰完变 3 向听（番 1）——保留七对明显更优。"""
    obj = situation(
        "1w1w2w2w3w3w4w4w5w5w6w6w1b",
        phase="response_peng",
        seat=1,
        turn=0,
        offered="1w",
        responding=(1,),
    )
    decider = HeuristicDecider(PolicyConfig(route_aware=True))
    chosen = decider.choose(
        obj,
        (Action(PASS), Action(PENG, tile=tiles.parse("1w"))),
        budget_ms=800,
    )
    assert chosen is not None and chosen.kind == PASS
    assert "七对" in decider.last_reason


def test_route_calibration_makes_the_four_pair_case_near_a_tie() -> None:
    """4 对 + 1 面子时两条路线几乎等值——这是按路线标定后的结果，不再是旧表那样系统性偏向七对。

    实测表下：七对 2 向听 10.5%×番 2 = 0.210，碰后副露 1 向听 21.4%×番 1 = 0.214，
    仅差约 2%。共用一张路线无关表时七对被高估近一倍，会一路去做七对。
    """
    from majiang.strategy import routes

    obj = situation(
        "1w1w3w3w5w5w7w7w1b2b3b9b9t",
        phase="response_peng",
        seat=1,
        turn=0,
        offered="1w",
        responding=(1,),
    )
    pair, meld = routes.evaluate(obj, base_score=1)
    decider = HeuristicDecider(PolicyConfig(route_aware=True))
    plan = decider._meld_plan(obj, tiles.parse("1w"), Action(PENG, tile=tiles.parse("1w")))
    assert plan is not None
    melds, after_counts = plan
    _, after = routes.evaluate(obj, base_score=1, counts=after_counts, melds=melds)
    assert abs(after.value - pair.value) / max(pair.value, 1e-9) < 0.15


def test_peng_declined_on_pair_route() -> None:
    obj = situation(
        "1w1w2w2w3w3w4w4w5w5w6b6b6b9t",
        phase="response_peng",
        seat=1,
        turn=0,
        offered="6b",
        responding=(1,),
    )
    decider = HeuristicDecider()
    chosen = decider.choose(
        obj,
        (Action(PASS), Action(PENG, tile=tiles.parse("6b"))),
        budget_ms=800,
    )
    assert chosen is not None and chosen.kind == PASS
    assert "七对" in decider.last_reason or "无改善" in decider.last_reason


def test_chi_declined_when_no_improvement() -> None:
    obj = situation(
        "1w2w3w4w5w6w7w8w9w1b3b5b7b9t",
        phase="response_chi",
        seat=1,
        turn=0,
        offered="2b",
        responding=(1,),
    )
    decider = HeuristicDecider()
    chosen = decider.choose(
        obj,
        (Action(PASS), Action(CHI, tile=tiles.parse("2b"), tiles=(tiles.parse("1b"), tiles.parse("3b")))),
        budget_ms=800,
    )
    assert chosen is not None and chosen.kind == PASS


def test_response_without_membership_still_returns_something() -> None:
    obj = situation(
        "1w2w3w4w5w6w7w8w9w1b3b5b7b9t",
        phase="response_chi",
        seat=1,
        turn=0,
        offered="2b",
        responding=(2,),
    )
    decider = HeuristicDecider()
    assert decider.choose(obj, (Action(PASS),), budget_ms=800).kind == PASS


def test_missing_drawn_tile_still_allows_hu() -> None:
    obj = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b5b", drawn=None)
    decider = HeuristicDecider()
    chosen = decider.choose(obj, (Action(HU),), budget_ms=1500)
    assert chosen is not None and chosen.kind == HU


def test_risk_model_uses_only_public_information() -> None:
    early = situation("1w2w3w4w5w6w7w8w9w1b2b3b5b", drawn="5b", wall=80)
    late = situation(
        "1w2w3w4w5w6w7w8w9w1b2b3b5b",
        drawn="5b",
        wall=25,
        discards={1: "1w2w3w4w5w6w7w", 2: "1b2b3b4b5b6b7b", 3: "1t2t3t4t5t6t7t"},
    )
    assert risk.lap_survival(risk.assess(early)) > risk.lap_survival(risk.assess(late))


def test_config_for_mode_shares_base_settings() -> None:
    qualifier = PolicyConfig.for_mode(Mode.QUALIFIER, base_score=5)
    final = PolicyConfig.for_mode(Mode.FINAL, base_score=5)
    assert qualifier.base_score == final.base_score == 5
    assert final.piao_threshold_scale < qualifier.piao_threshold_scale
