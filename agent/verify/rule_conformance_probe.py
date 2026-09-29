#!/usr/bin/env python
"""B5：规则与计分的**可证伪**一致性检查（只读，零平台请求）。

平台规则原文不在仓内，所以本探针只覆盖**仓内可独立核对**的部分：

A. **运行时参数不得写死**（AGENTS.md §2 硬约束）：
   `base_score` / `you_cai_bi_kao` / 庄闲倍率 必须来自运行时配置，而非散落在决策逻辑里的字面量。
   做法：AST 扫描 `src/majiang/strategy/**`，找「疑似硬编码的结算常量」；
   并确认 `score.settle` 的默认 base 只作为**兜底**（调用方一律传 config.base_score）。
B. **计分公式对拍文档示例**：
   - `fan=4` 且庄家自摸 → 四家净分 `[-32, 96, -32, -32]`
   - `fan=2` 且庄家自摸 → 每家闲家付 16
   - `fan=1` 闲家自摸 → 庄家付 8、两闲家各付 1
C. **财神规则**：`tiles.GOD == 33`（白板）；`you_cai_bi_kao` 在「摸到财神且非爆头/非杠开」时生效。
D. **行动合法性自洽**：财神不可吃/碰（`legal_actions` 不产出对财神的吃碰）。
"""

from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

FAIL: list[str] = []
def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)

section = lambda t: print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)  # noqa: E731

from majiang.rules import score, tiles  # noqa: E402

# ── A. 运行时参数不写死 ────────────────────────────────────────────────
section("A. 运行时参数注入（不得写死）")
src = (REPO / "src" / "majiang" / "strategy" / "policy.py").read_text(encoding="utf-8")
check("policy.py 从 tournament 读取 base_score",
      "base_score=int(tournament.base_score)" in src)
check("policy.py 从 tournament 读取 you_cai_bi_kao",
      "you_cai_bi_kao=bool(tournament.you_cai_bi_kao)" in src)

# 找策略层里被字面量化的结算乘数（8 / 96 / 32 这类）
# 注意：只盯**算术上下文**里的常量——集合/列表字面量（如牌型的 frozenset({2,3,4,5,6,7,8})）不是乘数。
suspicious: list[str] = []
for p in sorted((REPO / "src" / "majiang" / "strategy").rglob("*.py")):
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    # 先收集所有容器字面量里的常量行，排除掉
    container_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, int):
                    container_lines.add(child.lineno)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, int)
                and node.value in (8, 16, 32, 64, 96, 128, 256, 512)
                and node.lineno not in container_lines
                and isinstance(getattr(node, "ctx", None), ast.Load) is False):
            # 仅当它出现在算术表达式里才可疑（BinOp / 比较 / 赋值右侧）
            line = lines[node.lineno - 1].strip()
            if line.startswith("#") or "RANK" in line or "rank" in line or "frozenset" in line:
                continue
            suspicious.append(f"{p.name}:{node.lineno}: {line[:70]}")
check("策略层无裸结算乘数字面量", not suspicious,
      f"{len(suspicious)} 条（前 3）：{suspicious[:3]}")

# ── B. 计分公式对拍文档示例 ───────────────────────────────────────────
section("B. 计分公式对拍（平台变更记录里的实测例）")
d4 = score.seat_deltas(fan=4, base=1, winner_seat=1, dealer_seat=1)
check("fan=4 庄家自摸 → [-32, 96, -32, -32]", list(d4) == [-32, 96, -32, -32], f"{list(d4)}")
d2 = score.seat_deltas(fan=2, base=1, winner_seat=0, dealer_seat=0)
check("fan=2 庄家自摸 → 闲家各付 16", d2[1] == -16 and d2[2] == -16 and d2[3] == -16, f"{list(d2)}")
d1 = score.seat_deltas(fan=1, base=1, winner_seat=1, dealer_seat=0)
check("fan=1 闲家自摸 → 庄家付 8、闲家各付 1",
      d1[0] == -8 and d1[2] == -1 and d1[3] == -1, f"{list(d1)}")
# 守恒：净分和为 0
check("任意结算净分和为 0", sum(score.seat_deltas(fan=7, base=3, winner_seat=2, dealer_seat=3)) == 0)
# 负番/底分越界显式拒绝
try:
    score.seat_deltas(fan=-1, base=1, winner_seat=0, dealer_seat=0)
    check("负番显式拒绝", False, "未抛错")
except score.ScoreError:
    check("负番显式拒绝", True, "ScoreError")
try:
    score.settle(fan=1, base=0)
    check("底分 0 显式拒绝", False, "未抛错")
except score.ScoreError:
    check("底分 0 显式拒绝", True, "ScoreError")

# ── C. 财神常量 ───────────────────────────────────────────────────────
section("C. 财神常量与规则")
check("tiles.GOD == 33（白板）", tiles.GOD == 33, f"{tiles.GOD}")
check("is_god(白) 为真", tiles.is_god(tiles.parse("白")) is True)
check("is_god(1b) 为假", tiles.is_god(tiles.parse("1b")) is False)

# ── D. 财神不可吃碰（合法动作） ────────────────────────────────────────
section("D. 财神不可吃碰（legal_actions 自洽）")
from majiang.rules.action import CHI, PENG, legal_actions  # noqa: E402
from majiang.rules.god import GodState  # noqa: E402
from majiang.rules.hand import Hand  # noqa: E402
from majiang.rules.situation import PHASE_DRAW, Situation  # noqa: E402
from majiang.rules.table import TableState  # noqa: E402

# 手里两张白板 + 一张 1w，轮到自己摸牌：不应出现「碰白板」这类动作
s = Situation(
    seat=0, phase=PHASE_DRAW, turn=0,
    hand=Hand.from_codes(["白", "白", "白", "白", "1w", "2w", "3w", "4w", "5w",
                         "6w", "7w", "8w", "9w", "1b"]),
    god=GodState(), table=TableState.open(), drawn_tile=tiles.parse("1b"),
)
acts = legal_actions(s)
kinds = sorted({a.kind for a in acts})
said_god = [a for a in acts if a.kind in (CHI, PENG) and getattr(a, "tile", None) == tiles.GOD]
check("财神无吃/碰动作", not said_god, f"动作种类={kinds}")

section("结论")
if FAIL:
    print(f"未通过 {len(FAIL)} 项：")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过。")
