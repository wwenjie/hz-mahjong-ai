#!/usr/bin/env python
"""B4：模型产物确定性 + 边界输入（离线，零平台请求）。

1. **确定性**：同一组特征两次前向，输出**逐位一致**（纯 Python、无随机、无网络）。
2. **值域**：分类模型输出 ∈ [0,1]；回归模型输出有限（非 NaN/inf）。
3. **边界局面**：构造若干极端但仍然合法的局面，验证
   - `opponent_features.extract` / `features.extract` 不抛错、维数正确；
   - 模型前向不抛错；
   - 非法局面（暗手 >4 张同种等）由 `Hand` **显式拒绝**（不静默吞掉）。
4. **规模**：对局初始 13/14 张手牌、空弃牌、无副露等都能走通。

用法：.venv/bin/python agent/verify/model_determinism_probe.py
"""

from __future__ import annotations

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

from majiang.client.snapshot import Snapshot  # noqa: E402
from majiang.rules.hand import Hand  # noqa: E402
from majiang.strategy import features, opponent_features  # noqa: E402
from majiang.strategy.opponent import OpponentModel  # noqa: E402
from majiang.strategy.value import ValueModel  # noqa: E402

def codes(spec: str) -> list[str]:
    out, pending = [], ""
    for ch in spec:
        if ch.isspace():
            continue
        if pending:
            out.append(pending + ch); pending = ""
        elif ch.isdigit():
            pending = ch
        else:
            out.append(ch)
    assert not pending, spec
    return out


def make(hand_spec: str, *, seat=0, discards=None, melds=None, god=None,
         wall=74, drawn=None, phase="draw", last_discard=""):
    raw = {
        "game_id": "g_1", "phase": phase, "round_no": 1, "dealer": 1, "turn": seat,
        "waited_seat": seat, "wall_remaining": wall,
        "discards": [codes((discards or {}).get(i, "")) for i in range(4)],
        "melds": [(melds or {}).get(i, []) for i in range(4)],
        "hand_counts": [len(codes(hand_spec))] * 4,
        "my_hand": codes(hand_spec), "seat": seat,
        "last_discard": last_discard, "drawn_tile": drawn or "",
        "god": god or {"baotou": False, "chain_count": 0, "catch_play": False, "god_discarder_seat": -1},
        "scores": [0, 0, 0, 0],
    }
    return Snapshot.parse(raw).to_situation()

# ── 1. 确定性 ─────────────────────────────────────────────────────────
section("1. 模型产物确定性（同输入两次前向逐位一致）")
opp = OpponentModel.load(REPO / "models" / "opponent_model.json")
val = ValueModel.load(REPO / "models" / "value_model.json")
sit = make("1w2w3w4w5w6w7w8w9w5b6b2b2b9b", drawn="9b")

feat_o = opponent_features.extract(sit, 1)
run1 = [opp.predict(feat_o) for _ in range(3)]
check("对手模型：3 次前向逐位一致", run1[0] == run1[1] == run1[2], f"{run1[0]!r}")
check("对手模型：输出 ∈ [0,1]", 0.0 <= run1[0] <= 1.0, f"{run1[0]:.6f}")

feat_v = features.extract(sit)
rv = [val.predict(feat_v) for _ in range(3)]
check("价值模型：3 次前向逐位一致", rv[0] == rv[1] == rv[2], f"{rv[0]!r}")
import math  # noqa: E402
check("价值模型：输出有限（非 NaN/inf）", math.isfinite(rv[0]), f"{rv[0]:.4f}")

# ── 2. 特征维数 ───────────────────────────────────────────────────────
section("2. 特征维数")
check("opponent 特征维数 = FEATURE_COUNT",
      len(feat_o) == opponent_features.FEATURE_COUNT, f"{len(feat_o)}")
check("value 特征维数 = FEATURE_COUNT",
      len(feat_v) == features.FEATURE_COUNT, f"{len(feat_v)}")
check("opponent 特征无 NaN",
      all(math.isfinite(float(x)) for x in feat_o))
check("value 特征无 NaN", all(math.isfinite(float(x)) for x in feat_v))

# ── 3. 边界局面 ───────────────────────────────────────────────────────
section("3. 边界局面（合法但极端）")
bounds = {
    "14 张全不同（最杂）": "1w2w3w4w5w6w7w8w9w1b2b3b1t2t",
    "含 4 张财神（白板）": "白白白白1w2w3w4w5w6w7w8w9w",
    "刻子满手（14 张同种×4）": "1w1w1w1w2w2w2w2w3w3w3w3w9t9t",
    "听牌态（13 张）": "1w2w3w4w5w6w7w8w9w5b6b7b9b",
    "清一色": "1w1w1w2w2w2w3w3w3w4w4w4w5w5w",
}
for label, spec in bounds.items():
    try:
        s = make(spec)
        fo = opponent_features.extract(s, 1)
        fv = features.extract(s)
        po = float(opp.predict(fo))
        pv = float(val.predict(fv))
        ok = (len(fo) == opponent_features.FEATURE_COUNT
              and len(fv) == features.FEATURE_COUNT
              and 0.0 <= po <= 1.0 and math.isfinite(pv))
        check(f"边界[{label}] 走通", ok, f"P(听)={po:.4f} V={pv:.3f}")
    except Exception as exc:  # noqa: BLE001
        check(f"边界[{label}] 走通", False, f"{type(exc).__name__}: {exc}")

# ── 4. 非法输入必须显式拒绝（不静默） ──────────────────────────────────
section("4. 非法输入显式拒绝")
try:
    Hand.from_counts(tuple([5] + [0] * 33))  # 5 张同种：超过每色 4 张
    check("Hand 拒绝 5 张同种", False, "未抛错（静默吞掉）")
except Exception as exc:  # noqa: BLE001
    check("Hand 拒绝 5 张同种", True, type(exc).__name__)
try:
    Hand.from_counts(tuple([1] * 40))
    check("Hand 拒绝错误维数", False, "未抛错")
except Exception as exc:  # noqa: BLE001
    check("Hand 拒绝错误维数", True, type(exc).__name__)

section("结论")
if FAIL:
    print(f"未通过 {len(FAIL)} 项：")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过。")
