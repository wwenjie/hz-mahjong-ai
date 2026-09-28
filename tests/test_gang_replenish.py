"""杠后补牌走通用路径的回归测试（P1.1）。

来由：agent-a 于 2026-09-28 委派。指南 09-12/09-13 起，杠后补到能胡的牌不再自动结算，
与普通摸牌同构（可胡 / 续杠 / 弃胡打财神续飘）。

已核实的事实（只读审计，未打平台）：
- 真实事件流**没有**「杠后补牌」事件类型，补牌由随后的普通 ``tile_drawn`` 承担；
- 事件类型集实测为 ``tile_drawn / tile_discarded / timeout / pass / chi / peng / gang /
  round_ended / game_ended``，**不含 ``hu``**（胡牌只体现在 ``round_ended`` 的 ``data``：
  ``draw / detail / fan / scores``）；
- 补牌后快照走 ``phase="draw"`` + ``drawn_tile``，决策器据此应**主动提交 hu**，
  不依赖服务端代胡（``runtime/engine.py`` 的 ``_decide_safely → _submit``）。

归属：agent-c（用户 2026-09-28 授权；agent-a 在 THREAD 明确授权 C 增补 ``tests/**``，
但不得修改他人已写的断言）。本文件为**新建**，未触碰任何既有用例。
"""

from __future__ import annotations

from majiang.rules import action as action_module
from majiang.rules import tiles, win
from majiang.rules.action import HU, Action
from majiang.rules.god import GodState
from majiang.rules.hand import Hand
from majiang.rules.situation import PHASE_DRAW, Situation
from majiang.rules.table import TableState
from majiang.strategy.policy import HeuristicDecider, PolicyConfig

# 用例 1：摸到补牌前是「听 7w」，补牌 7w 成普通胡（4 面子 + 将）
WAIT_ON_SEVEN = ["1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w", "6w", "6w", "7w", "7w"]

# 用例 2：4 组面子 + 1 张财神，摸牌前即「听任意」（爆头）；补牌任意一张都成胡
BAOTOU_WAIT_ANY = ["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w", "4w", "4w", "4w", "白", "7w"]

# 用例 3：4 组面子 + 2 张财神；补牌恰是第 2 张财神 → 手上仍有财神可打（财飘链入口）
TWO_GODS = ["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w", "4w", "4w", "4w", "白", "白"]


def situation(codes: list[str], *, drawn: str, god: GodState) -> Situation:
    """构造 ``phase="draw"`` 的局面；``drawn`` 必须已在 ``codes`` 中（平台口径）。"""
    assert drawn in codes
    return Situation(
        seat=0,
        phase=PHASE_DRAW,
        turn=0,
        hand=Hand.from_codes(codes),
        god=god,
        table=TableState.open(),
        drawn_tile=tiles.parse(drawn),
    )


def choose(s: Situation) -> tuple[list[Action], Action | None, str]:
    actions = action_module.legal_actions(s)
    decider = HeuristicDecider(PolicyConfig())
    return actions, decider.choose(s, actions, budget_ms=1800), decider.last_reason


def test_replenishment_completing_a_normal_win_is_submitted() -> None:
    """杠后补牌成**普通胡**：合法动作含 hu，决策必须返回 HU（不得静默弃胡）。"""
    s = situation(WAIT_ON_SEVEN, drawn="7w", god=GodState())
    actions, choice, reason = choose(s)
    assert HU in {a.kind for a in actions}
    assert choice is not None
    assert choice.kind == HU, f"补牌成胡却未提交 hu，实际 {choice.describe()}（{reason}）"


def test_replenishment_completing_a_baotou_win_is_submitted() -> None:
    """杠后补牌成**听任意（爆头）**且无飘可用：决策必须返回 HU。

    这一格是 A 点名要求的：14 张 + ``phase=draw`` + ``drawn=补牌`` + ``god.baotou=True``。
    注意断言写「返回 HU」而不是「不返回 discard」——后者在用例 3 那种
    「还有第二张财神可打」的局面下本就不成立（弃胡飘是**正确**行为）。
    """
    pre = list(Hand.from_codes(BAOTOU_WAIT_ANY).counts)
    pre[tiles.parse("7w")] -= 1
    assert win.is_baotou(pre, 0), "构造前提不成立：摸牌前应是爆头（听任意）"

    s = situation(BAOTOU_WAIT_ANY, drawn="7w", god=GodState(hand_gods=1, baotou=True))
    actions, choice, reason = choose(s)
    assert HU in {a.kind for a in actions}
    assert choice is not None
    assert choice.kind == HU, f"补牌成爆头却未提交 hu，实际 {choice.describe()}（{reason}）"


def test_replenishment_leaves_the_piao_entry_open() -> None:
    """补牌成爆头且**手留第二张财神**时可弃胡飘——记录现状，不当成必须触发。

    只断言「可以打财神」这一合法性，不断言策略一定选飘（阈值随模式变化）；
    这条的价值在于固定住「财飘链入口在补牌路径上依然可达」。
    """
    s = situation(TWO_GODS, drawn="白", god=GodState(hand_gods=2, baotou=True))
    actions, choice, reason = choose(s)
    discards = [a for a in actions if a.kind == "discard" and a.tile == tiles.GOD]
    assert discards, "补牌后应仍可打出财神（财飘链入口）"
    assert choice is not None
