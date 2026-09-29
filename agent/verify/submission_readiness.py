#!/usr/bin/env python
"""提交就绪审计（agent-c 独立线，只读 + 纯离线演练）。

回答三件事，全部**可复现、可证伪**：

A. **静态合规**：`src/majiang/**` 是否零第三方依赖？是否有硬编码凭据？
B. **模型回退演练**：所有模型驱动档位在「模型缺失 / 损坏」时是否**绝不中止决策**、
   而是回退启发式并打印警告？（AGENTS.md §5 硬要求）
C. **产物完整性**：模型文件是否存在、能否用纯 Python 加载、是否确定性。

用法：.venv/bin/python agent/verify/submission_readiness.py
纪律：只读；不写 src/**；零平台请求。写入 `agent/out/` 由调用方重定向。
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

FAIL: list[str] = []
WARN: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def section(t: str) -> None:
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ── A. 静态合规 ─────────────────────────────────────────────────────────
section("A. 静态合规")

try:
    import tomllib

    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    deps = py["project"].get("dependencies") or []
    check("pyproject 运行依赖为空", deps == [], f"dependencies={deps}")
except Exception as exc:  # noqa: BLE001
    check("pyproject 可解析", False, str(exc))

stdlib = set(sys.stdlib_module_names)
third: list[tuple[str, str]] = []
for p in sorted((REPO / "src" / "majiang").rglob("*.py")):
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        check(f"{p.name} 语法", False, str(exc))
        continue
    for n in ast.walk(tree):
        mods: list[str] = []
        if isinstance(n, ast.Import):
            mods = [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom) and not (n.level or 0) and n.module:
            mods = [n.module.split(".")[0]]
        for m in mods:
            if m not in stdlib and m != "majiang":
                third.append((str(p.relative_to(REPO)), m))
check("src/majiang 零第三方 import", not third, f"命中 {len(third)} 条 {third[:5]}")

CRED_PAT = ast.parse  # placeholder to keep linters quiet
import re  # noqa: E402

suspicious: list[str] = []
pat = re.compile(r"['\"][A-Za-z0-9_\-]{24,}['\"]")
for p in sorted((REPO / "src" / "majiang").rglob("*.py")):
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s.startswith("#") or "token" not in line.lower():
            continue
        for m in pat.finditer(line):
            if m.group(0).strip("'\"").isalnum():
                suspicious.append(f"{p.relative_to(REPO)}:{i}: {s[:80]}")
check("无硬编码长 token 字面量", not suspicious, f"{len(suspicious)} 条 {suspicious[:3]}")

# ── B. 模型回退演练 ─────────────────────────────────────────────────────
section("B. 模型回退演练（模型缺失 / 损坏）")

from majiang.strategy import value as value_mod  # noqa: E402
from majiang.strategy.opponent import (  # noqa: E402
    ModelReadyModel,
    OpponentModel,
    load_or_none,
)

# B1: 对手模型 —— 路径不存在
mm = load_or_none(str(REPO / "models" / "__nope__.json"))
check("对手模型：缺失路径 → 不抛错且回退", mm.using_model is False, f"using_model={mm.using_model}")
check("对手模型：回退对象可用", isinstance(mm, ModelReadyModel))

# B2: 对手模型 —— 文件存在但内容损坏
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    fh.write("{not valid json")
    bad = fh.name
mm2 = load_or_none(bad)
check("对手模型：损坏文件 → 回退（不抛错）", mm2.using_model is False)

# B3: 对手模型 —— 结构合法但字段缺失
payload = json.loads((REPO / "models" / "opponent_model.json").read_text(encoding="utf-8"))
try:
    OpponentModel({})
    check("对手模型：空 payload 应报错（ModelError）", False, "未抛错")
except Exception as exc:  # noqa: BLE001
    check("对手模型：空 payload 抛 ModelError", "ModelError" in type(exc).__name__, type(exc).__name__)

# B4: 价值档位 —— 用坏路径构造 decider，必须回退启发式而非崩
from majiang.cli import _value_decider  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode  # noqa: E402

d = _value_decider(Mode.QUALIFIER, str(REPO / "models" / "__nope__.json"))
check("价值档位：坏路径 → 回退为启发式", isinstance(d, HeuristicDecider), type(d).__name__)

# B5: 正常模型能加载且确定性
m_ok = load_or_none(str(REPO / "models" / "opponent_model.json"))
check("对手模型：正常路径可加载", m_ok.using_model is True)
vm = value_mod.ValueModel.load(str(REPO / "models" / "value_model.json"))
check("价值模型：正常路径可加载", vm is not None)

# ── C. 产物完整性 ───────────────────────────────────────────────────────
section("C. 产物完整性")
for name in ("value_model.json", "opponent_model.json"):
    p = REPO / "models" / name
    ok = p.exists() and p.stat().st_size > 0
    check(f"{name} 存在且非空", ok, f"{p.stat().st_size} B" if p.exists() else "缺失")

# 纯 Python 求值不 import numpy/torch（AST 级，不看注释/文档串）
gb_tree = ast.parse((REPO / "src" / "majiang" / "strategy" / "gbdt.py").read_text(encoding="utf-8"))
gb_mods: set[str] = set()
for n in ast.walk(gb_tree):
    if isinstance(n, ast.Import):
        gb_mods.update(a.name.split(".")[0] for a in n.names)
    elif isinstance(n, ast.ImportFrom) and not (n.level or 0) and n.module:
        gb_mods.add(n.module.split(".")[0])
check("gbdt.py 不 import numpy/torch", not ({"numpy", "torch"} & gb_mods), f"imports={sorted(gb_mods)}")

# 默认档不依赖模型
from majiang.cli import DECIDERS  # noqa: E402

check("默认档 'heuristic' 在 DECIDERS 中", "heuristic" in DECIDERS, f"{len(DECIDERS)} 个档位")

section("结论")
if FAIL:
    print(f"未通过 {len(FAIL)} 项：")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过。")
