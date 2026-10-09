"""决定性实验：把 6 个「口味分歧」点跑过多个臂，看谁的选择翻向用户直觉。
臂：v5(1步,exact-ukeire) / two-ply / v5-twoply / v5-cand5-twoply / search
目的：区分「引擎 bug」vs「目标函数分歧（1步 vs 2步/改良）」。
"""
from __future__ import annotations
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, _HERE)
from majiang.rules import tiles as T
from majiang.rules.action import legal_actions
from majiang.cli import make_decider
from majiang.strategy.policy import Mode
from replay_report import _situation_from_view

CASES = {
    "seq7":   ("report_20261008_233042_seq7",   "2b",  ["6t", "8t"]),
    "seq31":  ("report_20261009_000104_seq31",  "9b",  ["9w"]),
    "seq65":  ("report_20261009_002905_seq65",  "1w",  ["6b"]),
    "seq71":  ("report_20261009_102559_seq71",  "2t",  ["7w"]),
    "seq141": ("report_20261009_114836_seq141", "8w",  ["5w", "3t", "5t"]),
    "seq150": ("report_20261009_131257_seq150", "1b",  ["9b"]),
}
ARMS = ["v5", "two-ply", "v5-twoply", "v5-cand5-twoply"]

for tag, (rid, v5t, human) in CASES.items():
    doc = json.loads(open(os.path.join(_ROOT, "webapp", "reports", rid + ".json"), encoding="utf-8").read())
    sit = _situation_from_view((doc.get("decision") or {}).get("situation") or {})
    acts = legal_actions(sit)
    print("=" * 88)
    print(f"{tag}  手牌={' '.join(T.to_codes(T.tiles_of(sit.hand.counts)))}  "
          f"melds={[m.kind+':'+''.join(T.to_codes(m.tiles)) for m in sit.hand.melds]}")
    print(f"    v5实选={v5t}   用户主张={human}")
    for arm in ARMS:
        try:
            d = make_decider(arm, Mode.QUALIFIER)
            ch = d.choose(sit, list(acts), budget_ms=3000)
            got = T.to_code(ch.tile) if ch and ch.tile is not None else ch.describe()
        except Exception as e:
            got = f"ERR:{type(e).__name__}:{e}"
        mark = " ←与v5同" if got == v5t else (" ←翻向用户!" if got in human else "")
        print(f"    {arm:20s} → {got}{mark}")
