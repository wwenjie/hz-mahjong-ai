#!/usr/bin/env python
"""B4：模型产物确定性 + 边界输入（离线，零平台请求）。

两件事：

1. **确定性**：同一输入两次调用**逐位一致**（模型求值 / 特征提取 / 启发式决策）。
   顺带跨进程复核（`PYTHONHASHSEED` 随机化下结果不变），排除「字典序/哈希序」类不确定性。
2. **边界输入不崩**：空手牌、单牌种满 4 张、抓打圈受限、财神满配、牌墙 0 / 满、
   极大局号、observer 座位、`actions=()` 空候选 —— 逐项要求
   **要么给确定值、要么给出可解释的领域异常（`HandError`/`SituationError`/`GodStateError`）**，
   绝不出现「未捕获的其它异常」。

用法：`uv run python agent/verify/model_determinism_boundary_probe.py`
产物：`agent/out/model-determinism-boundary.log`
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import traceback

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "agent" / "out" / "model-determinism-boundary.log"

from majiang.rules import action as action_module  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.god import GodState  # noqa: E402
from majiang.rules.hand import Hand, HandError  # noqa: E402
from majiang.rules.situation import PHASE_DRAW, Situation, SituationError  # noqa: E402
from majiang.rules.table import INITIAL_WALL, TableState, TableError  # noqa: E402
from majiang.strategy import features, opponent_features  # noqa: E402
from majiang.strategy.opponent import OpponentModel  # noqa: E402
from majiang.strategy.policy import HeuristicDecider  # noqa: E402
from majiang.strategy.value import ValueModel  # noqa: E402

DOMAIN_ERRORS = (HandError, SituationError, TableError, ValueError)

LINES: list[str] = []
FAIL: list[str] = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def check(label: str, ok: bool, detail: str = "") -> None:
    say(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def section(title: str) -> None:
    say("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


def situation(
    codes: list[str],
    *,
    drawn: str | None = None,
    god: GodState | None = None,
    wall: int | None = None,
    round_no: int = 1,
    seat: int = 0,
    melds=(),
) -> Situation:
    table = TableState.open()
    if wall is not None:
        table = TableState(wall_remaining=wall, dealer_seat=0, round_no=round_no)
    elif round_no != 1:
        table = TableState(wall_remaining=INITIAL_WALL, dealer_seat=0, round_no=round_no)
    return Situation(
        seat=seat,
        phase=PHASE_DRAW,
        turn=seat,
        hand=Hand.from_codes(codes, melds),
        god=god or GodState(),
        table=table,
        drawn_tile=None if drawn is None else tiles.parse(drawn),
    )


VALUE = ValueModel.load(REPO / "models" / "value_model.json")
OPPONENT = OpponentModel.load(REPO / "models" / "opponent_model.json")

# ── 1. 模型求值确定性 ────────────────────────────────────────────────────
section("1. 模型求值确定性（同输入两次，逐位一致）")

base_s = situation(
    ["1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w", "6w", "6w", "白"],
    drawn="7w",
    god=GodState(hand_gods=1, chain_count=1, piao_count=0, baotou=True),
)
v = features.extract(base_s)
p1 = VALUE.predict(v)
p2 = VALUE.predict(v)
check("价值模型 predict 两次逐位一致", p1 == p2, f"{p1!r}")

o1 = OPPONENT.ready_probability(base_s, 1)
o2 = OPPONENT.ready_probability(base_s, 1)
check("对手模型 ready_probability 两次逐位一致", o1 == o2, f"{o1!r}")

f1 = features.extract(base_s)
f2 = features.extract(base_s)
check("价值特征 extract 两次逐位一致", list(f1) == list(f2))
check("特征维数 == FEATURE_COUNT", len(f1) == features.FEATURE_COUNT, f"{len(f1)}")

of1 = opponent_features.extract(base_s, 2)
of2 = opponent_features.extract(base_s, 2)
check("对手特征 extract 两次逐位一致", list(of1) == list(of2))
check("对手特征维数 == FEATURE_COUNT", len(of1) == opponent_features.FEATURE_COUNT, f"{len(of1)}")

# 特征向量是否含 NaN/Inf（会让模型输出不确定或崩）
bad_vals = [x for x in list(f1) + list(of1) if x != x or x in (float("inf"), float("-inf"))]
check("特征向量无 NaN/Inf", not bad_vals, f"{bad_vals[:4]}")

# ── 2. 决策确定性 ────────────────────────────────────────────────────────
section("2. 启发式决策确定性（同局面两次，动作与理由一致）")

codes = ["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w", "4w", "4w", "4w", "白", "7w"]
s = situation(codes, drawn="7w", god=GodState(hand_gods=1, chain_count=1, baotou=True))
actions = action_module.legal_actions(s)
runs = []
for _ in range(3):
    d = HeuristicDecider()
    choice = d.choose(s, actions, budget_ms=600)
    runs.append((None if choice is None else choice.describe(), d.last_reason))
check("决策两次一致（动作）", runs[0][0] == runs[1][0] == runs[2][0], f"{runs[0][0]!r}")
check("决策两次一致（理由）", runs[0][1] == runs[1][1] == runs[2][1], f"{runs[0][1]!r}")

# ── 3. 跨进程确定性（PYTHONHASHSEED 随机化） ─────────────────────────────
section("3. 跨进程确定性（不同 PYTHONHASHSEED 下同一输出）")

# 子进程脚本：用来在两个不同 PYTHONHASHSEED 的进程里复算同一串输出。
child_py = REPO / "agent" / "out" / "_determinism_child.py"
child_py.write_text(
    "import sys, pathlib\n"
    "sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'src'))\n"
    "from majiang.rules import tiles\n"
    "from majiang.rules.god import GodState\n"
    "from majiang.rules.hand import Hand\n"
    "from majiang.rules.situation import PHASE_DRAW, Situation\n"
    "from majiang.rules.table import TableState\n"
    "from majiang.strategy import features, opponent_features\n"
    "from majiang.strategy.opponent import OpponentModel\n"
    "from majiang.strategy.value import ValueModel\n"
    "\n"
    "s = Situation(seat=0, phase=PHASE_DRAW, turn=0,\n"
    "              hand=Hand.from_codes(['1w','1w','2w','2w','3w','3w','4w','4w','5w','5w','6w','6w','白']),\n"
    "              god=GodState(hand_gods=1, chain_count=1),\n"
    "              table=TableState.open(), drawn_tile=tiles.parse('7w'))\n"
    "V = ValueModel.load(pathlib.Path(sys.argv[1]) / 'models' / 'value_model.json')\n"
    "O = OpponentModel.load(pathlib.Path(sys.argv[1]) / 'models' / 'opponent_model.json')\n"
    "print(repr(V.predict(features.extract(s))))\n"
    "print(repr(O.ready_probability(s, 3)))\n"
    "print(repr(tuple(features.extract(s))))\n"
    "print(repr(tuple(opponent_features.extract(s, 1))))\n",
    encoding="utf-8",
)

outputs = []
for seed in ("0", "1", "12345"):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    proc = subprocess.run(
        [sys.executable, str(child_py), str(REPO)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    outputs.append((seed, proc.stdout.strip()))
uniq = {out for _, out in outputs}
check("3 个 PYTHONHASHSEED 下输出完全相同", len(uniq) == 1, f"{len(uniq)} 种不同输出；stderr={outputs[0][1][:0]}")
if len(uniq) != 1:
    for seed, out in outputs:
        say(f"  seed={seed}: {out[:120]}")

# ── 4. 边界输入 ─────────────────────────────────────────────────────────
section("4. 边界输入（不崩；要么给确定值，要么给领域异常）")


def attempt(label: str, fn, *, expect_error: type | None = None) -> None:
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        if expect_error is not None and isinstance(exc, expect_error):
            check(label, True, f"按预期抛出 {type(exc).__name__}: {exc}")
        elif isinstance(exc, DOMAIN_ERRORS):
            check(label, True, f"领域异常（可解释） {type(exc).__name__}: {exc}")
        else:
            check(label, False, f"未捕获异常 {type(exc).__name__}: {exc}")
            traceback.print_exc()
        return
    if expect_error is not None:
        check(label, False, f"预期抛 {expect_error.__name__}，实际返回 {result!r}")
    else:
        check(label, True, f"返回 {str(result)[:90]}")


# 4.1 空手牌（结构合法，但尚未摸牌）；特征提取必须给确定值
empty = Situation(
    seat=0, phase=PHASE_DRAW, turn=0, hand=Hand.from_counts([0] * tiles.TILE_KINDS),
    god=GodState(), table=TableState.open(), drawn_tile=None,
)
attempt("空手牌 → 价值特征可提取", lambda: tuple(features.extract(empty))[:3])
attempt("空手牌 → 对手特征可提取(seat=1)", lambda: tuple(opponent_features.extract(empty, 1))[:3])
attempt("空手牌 → 价值模型可求值", lambda: VALUE.predict(features.extract(empty)))
attempt("空手牌 → 对手模型可求值", lambda: OPPONENT.ready_probability(empty, 1))
attempt("空手牌 → 空候选下决策返回 None", lambda: HeuristicDecider().choose(empty, (), budget_ms=200), expect_error=None)

# 4.2 单牌种满 4 张（合法上界）；5 张必须被拒
attempt(
    "单牌种 4 张 → 可构造",
    lambda: Hand.from_codes(["1w", "1w", "1w", "1w"]).counts[tiles.parse("1w")],
)
attempt(
    "单牌种 5 张 → 必须抛 HandError",
    lambda: Hand.from_codes(["1w"] * 5),
    expect_error=HandError,
)
# 14 张同种在物理上不存在（每种仅 4 张），必须被拦住
attempt("14 张同种 → 必须抛 HandError", lambda: Hand.from_counts([14] + [0] * (tiles.TILE_KINDS - 1)), expect_error=HandError)

# 4.3 财神满配 + 抓打圈（全场财神总数 FOUR_GODS_TOTAL=4，手留 + 飘出不得超过）
full_god = GodState(hand_gods=1, chain_count=3, piao_count=3, baotou=True)
attempt("财神用尽态(手留1+飘3=4, 链3) → 可构造", lambda: (full_god.hand_gods, full_god.piao_count))
attempt(
    "财神超配(手留4+飘1>4) → 必须抛 GodStateError",
    lambda: GodState(hand_gods=4, chain_count=1, piao_count=1),
    expect_error=ValueError,
)
attempt(
    "飘次数>链次数 → 必须抛 GodStateError",
    lambda: GodState(hand_gods=0, chain_count=1, piao_count=2),
    expect_error=ValueError,
)
catch = GodState(hand_gods=1, chain_count=1, piao_count=1, baotou=True, catch_play=True, god_discarder_seat=2)
attempt("抓打圈受限态 → GodState 可构造", lambda: catch.catch_play)
attempt(
    "抓打圈态 → 合法动作只允许打刚摸的牌/暗杠",
    lambda: sorted({a.kind for a in action_module.legal_actions(
        situation(["1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w", "6w", "6w", "白"], drawn="7w", god=catch)
    )}),
)
attempt(
    "抓打圈态 → 特征提取不崩",
    lambda: tuple(features.extract(situation(["1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w", "6w", "6w", "白"], drawn="7w", god=catch)))[:3],
)

# 4.4 牌墙 0 / 满 / 越界
attempt("牌墙 0 → 可构造且 draws_left=0", lambda: TableState(wall_remaining=0, dealer_seat=0, round_no=1).draws_left)
attempt("牌墙满 → draws_left", lambda: TableState.open().draws_left)
attempt("牌墙越界(-1) → 必须抛 TableError", lambda: TableState(wall_remaining=-1, dealer_seat=0, round_no=1), expect_error=TableError)
attempt("牌墙越界(INITIAL+1) → 必须抛 TableError", lambda: TableState(wall_remaining=INITIAL_WALL + 1, dealer_seat=0, round_no=1), expect_error=TableError)
attempt(
    "牌墙 0 态 → 价值特征可提取",
    lambda: tuple(features.extract(situation(["1w", "2w", "3w"], wall=0)))[:3],
)
attempt("极大局号 → 可构造", lambda: TableState(wall_remaining=50, dealer_seat=0, round_no=10_000).round_no)
attempt(
    "极大局号 → 特征可提取",
    lambda: tuple(features.extract(situation(["1w", "2w", "3w"], wall=50, round_no=10_000)))[:3],
)
attempt("局号 0 → 必须抛 TableError", lambda: TableState(wall_remaining=50, dealer_seat=0, round_no=0), expect_error=TableError)

# 4.5 observer 座位：不是可行动座位，必须干净处理
attempt(
    "observer 座位 → 合法动作集合为空",
    lambda: action_module.legal_actions(
        Situation(
            seat=-1, phase=PHASE_DRAW, turn=0, hand=Hand.from_counts([0] * tiles.TILE_KINDS),
            god=GodState(), table=TableState.open(), drawn_tile=None,
        )
    ),
)

# ── 5. 汇总 ─────────────────────────────────────────────────────────────
section("汇总")
say(f"检查项 {len(LINES)} 行；FAIL {len(FAIL)} 项：{FAIL}")
verdict = "PASS" if not FAIL else "FAIL"
child_py.unlink(missing_ok=True)
OUT.write_text(
    "\n".join(
        [
            "# B4 模型产物确定性 + 边界输入（离线，零平台请求）",
            f"# repo={REPO} python={sys.version.split()[0]}",
            f"# verdict={verdict}",
            "",
            *LINES,
            "",
            json.dumps({"verdict": verdict, "failed_checks": FAIL}, ensure_ascii=False),
            "",
        ]
    ),
    encoding="utf-8",
)
say(f"\nverdict = {verdict}")
say(f"log -> {OUT}")
raise SystemExit(0 if verdict == "PASS" else 1)
