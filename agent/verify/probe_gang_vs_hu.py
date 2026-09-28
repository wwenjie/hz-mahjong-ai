"""只读回归用例：杠后补牌的决策路径（A 于 2026-09-28 11:05 委派）。

归属：agent-c（小龙虾）。**不编辑 A/B 的测试文件**（`tests/**` 归他们），
本脚本放在 C 的地盘 `agent/verify/`，可直接跑，也可被 pytest 收集（函数以 `check_` 开头，
pytest 只自动收集 `test_`，所以默认不会被误收集——这是刻意的）。

不碰平台、不改任何源文件，只直调规则内核与决策器。

跑法：uv run python agent/verify/probe_gang_vs_hu.py
"""

from __future__ import annotations

from majiang.rules import action as action_module
from majiang.rules import tiles, win
from majiang.rules.god import GodState
from majiang.rules.hand import Hand
from majiang.rules.situation import PHASE_DRAW, Situation
from majiang.rules.table import TableState
from majiang.strategy.policy import HeuristicDecider, PolicyConfig

DRAW = PHASE_DRAW


def make(codes: list[str], *, drawn: str, god: GodState) -> Situation:
    return Situation(
        seat=0,
        phase=DRAW,
        turn=0,
        hand=Hand.from_codes(codes),
        god=god,
        table=TableState.open(),
        drawn_tile=tiles.parse(drawn),
    )


def decide(s: Situation):
    actions = action_module.legal_actions(s)
    d = HeuristicDecider(PolicyConfig())
    choice = d.choose(s, actions, budget_ms=1800)
    return actions, choice, d.last_reason


def show(title: str, s: Situation) -> None:
    actions, choice, reason = decide(s)
    print(f"\n=== {title} ===")
    print("合法动作:", sorted({a.kind for a in actions}))
    print("决策:", None if choice is None else choice.describe())
    print("理由:", reason)


# --- 用例 1：杠后补牌成「普通胡」（非爆头） -------------------------------
# 4 组面子 + 将，补牌即胡。验证：决策返回 HU，而不是 discard。
codes1 = ["1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w", "6w", "6w", "7w", "7w"]
print("用例1 是胡牌:", win.is_winning_shape(Hand.from_codes(codes1).counts, 0))
show("用例1 杠后补牌成普通胡（应返回 hu）", make(codes1, drawn="7w", god=GodState()))


# --- 用例 2：杠后补牌成「听任意」（爆头）→ 财飘链入口 ---------------------
# 4 组面子 + 1 财神，补牌即胡且摸牌前已听任意。
codes2 = ["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w", "4w", "4w", "4w", "白", "7w"]
pre2 = list(Hand.from_codes(codes2).counts)
pre2[tiles.parse("7w")] -= 1
print("\n用例2 摸牌前 is_baotou:", win.is_baotou(pre2, 0))
show("用例2 杠后补牌成爆头（应 hu，或走续飘/续杠）", make(codes2, drawn="7w", god=GodState(hand_gods=1, chain_count=1, baotou=True)))


# --- 用例 3：能胡 + 能暗杠（新问题 1 的最小复现） --------------------------
codes3 = ["1w", "1w", "1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w", "6w", "6w"]
show("用例3 能胡 + 能暗杠（现状总是 hu，不看杠）", make(codes3, drawn="6w", god=GodState()))


# --- 用例 4：爆头且手上有财神 → 财飘链入口是否可达 -----------------------
# 6 对 + 1 财神 + 第 14 张；摸牌前已听任意，且手上有可打的财神。
codes4 = ["1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w", "6w", "6w", "白", "7w"]
show("用例4 爆头 + 手留财神（财飘链入口）", make(codes4, drawn="7w", god=GodState(hand_gods=1, baotou=True)))


# --- 用例 5：非爆头打财神 → 是否被当成财飘（链 +1） -----------------------
# 这是 replay.py 的近似所涉场景；这里看决策器是否会把非爆头打财神当飘。
codes5 = ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w", "2b", "2b", "3b", "白", "5b"]
show("用例5 非爆头持财神（打财神不应算飘）", make(codes5, drawn="5b", god=GodState(hand_gods=1, baotou=False)))


print("\n--- 汇总 ---")
print("用例 1 期望: 决策为 hu（补牌成胡主动提交）")
print("用例 2 期望: 决策为 hu 或 弃胡飘/续杠（爆头链入口可达）")
print("用例 3 现状: 决策为 hu（杠未被考虑——见 THREAD 新问题 1）")
print("用例 4 期望: 若链入口可达则可能弃胡飘")
print("用例 5 期望: 打财神不应被计为财飘（链 +1）")
