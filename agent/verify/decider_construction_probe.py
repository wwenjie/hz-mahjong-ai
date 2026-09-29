#!/usr/bin/env python
"""B3：全部决策器档位的构造演练 + 模型故障注入（离线，零平台请求）。

回答两件事：

1. **构造即验证**：`DECIDERS` 里每一个档位 × 两种 mode 都能构造出来、不抛异常。
   （捕获「某个档位在模型缺失时会崩」这类只在启动路径上暴露的问题。）
2. **故障注入**：把对手/价值模型替换成
   - 不存在的路径
   - 损坏的 JSON
   - 合法 JSON 但空 payload
   - 特征维数不符 / 版本字段不符
   逐项验证：**绝不抛错、退回启发式**（AGENTS.md §5 硬要求）。

用法：.venv/bin/python agent/verify/decider_construction_probe.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import traceback

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

FAIL: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


section = lambda t: print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)  # noqa: E731

from majiang.cli import DECIDERS, _value_decider  # noqa: E402
from majiang.strategy.opponent import load_or_none  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig  # noqa: E402

# ── 1. 全部档位构造 ──────────────────────────────────────────────────────
section(f"1. 构造全部档位（{len(DECIDERS)} 个 × 2 mode）")
bad: list[str] = []
for name, factory in DECIDERS.items():
    for mode in Mode:
        try:
            d = factory(mode)
            assert d is not None
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{name}[{mode.value}]: {type(exc).__name__}: {exc}")
            traceback.print_exc()
check("所有档位可构造", not bad, f"{len(bad)} 个失败：{bad[:4]}")

# ── 2. 模型故障注入 ────────────────────────────────────────────────────
section("2. 模型故障注入（对手模型 / 价值模型）")

good_opp = json.loads((REPO / "models" / "opponent_model.json").read_text(encoding="utf-8"))
good_val = json.loads((REPO / "models" / "value_model.json").read_text(encoding="utf-8"))
print(f"  基线 payload keys: opponent={sorted(good_opp)[:8]} ... value={sorted(good_val)[:8]} ...")


def write_tmp(obj_or_text) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        if isinstance(obj_or_text, str):
            fh.write(obj_or_text)
        else:
            json.dump(obj_or_text, fh)
        return fh.name


# 构造各种坏 payload
variants: dict[str, object] = {
    "损坏 JSON": "{not json at all",
    "空 payload": {},
}

# 特征维数不符：改 n_features 相关字段（逐名尝试，避免猜错键名）
for key in ("n_features", "feature_count", "expected_features", "n_features_in"):
    if key in good_opp:
        v = json.loads(json.dumps(good_opp))
        v[key] = 999
        variants[f"对手特征维数不符({key}=999)"] = v
        break

# 版本不符
for key in ("version", "model_version", "MODEL_VERSION"):
    if key in good_opp:
        v = json.loads(json.dumps(good_opp))
        v[key] = "bogus-version-xyz"
        variants[f"对手版本不符({key})"] = v
        break

print(f"  注入变体：{list(variants)}")

# 2a. 对手模型：坏 payload → load_or_none 必须不抛错、.using_model False
for label, payload in variants.items():
    path = write_tmp(payload)
    try:
        mm = load_or_none(path)
        ok = mm.using_model is False
        check(f"对手[{label}] → 回退不抛错", ok, f"using_model={mm.using_model}")
    except Exception as exc:  # noqa: BLE001
        check(f"对手[{label}] → 回退不抛错", False, f"抛了 {type(exc).__name__}: {exc}")

# 不存在的路径 / 空路径
for label, path in (("不存在的路径", str(REPO / "models" / "__nope__.json")), ("空路径", "")):
    try:
        mm = load_or_none(path)
        check(f"对手[{label}] → 回退不抛错", mm.using_model is False)
    except Exception as exc:  # noqa: BLE001
        check(f"对手[{label}] → 回退不抛错", False, f"抛了 {type(exc).__name__}: {exc}")

# 2b. risk 档位：用坏模型构造 HeuristicDecider，必须不抛错且仍可用
for label, payload in variants.items():
    path = write_tmp(payload)
    try:
        d = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER), risk_model=load_or_none(path))
        check(f"risk档[{label}] → 构造成功", isinstance(d, HeuristicDecider))
    except Exception as exc:  # noqa: BLE001
        check(f"risk档[{label}] → 构造成功", False, f"抛了 {type(exc).__name__}: {exc}")

# 2c. value 档位：坏 payload → _value_decider 回退启发式
for label, payload in variants.items():
    path = write_tmp(payload)
    try:
        d = _value_decider(Mode.QUALIFIER, path)
        check(f"value档[{label}] → 回退启发式", isinstance(d, HeuristicDecider), type(d).__name__)
    except Exception as exc:  # noqa: BLE001
        check(f"value档[{label}] → 回退启发式", False, f"抛了 {type(exc).__name__}: {exc}")

section("结论")
if FAIL:
    print(f"未通过 {len(FAIL)} 项：")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过。")
