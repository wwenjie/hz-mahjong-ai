"""独立复算 6 个分歧点：用**本仓引擎** shanten/ukeire 核对检索简报的裁决。
输出每个候选弃牌的 (向听, 进张枚数, 进张种类, 进张明细)。
"""
from __future__ import annotations
import glob, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, _HERE)

from majiang.rules import tiles as T, shanten as sh
from majiang.rules.action import legal_actions, DISCARD
from majiang.cli import make_decider
from majiang.strategy.policy import Mode
from replay_report import _situation_from_view

SEQS = [
    "report_20261009_000104_seq31",
    "report_20261009_114836_seq141",
    "report_20261008_233042_seq7",
    "report_20261009_112221_seq124",
    "report_20261009_131257_seq150",
    "report_20261009_102559_seq71",
]

for rid in SEQS:
    path = os.path.join(_ROOT, "webapp", "reports", rid + ".json")
    doc = json.loads(open(path, encoding="utf-8").read())
    view = (doc.get("decision") or {}).get("situation") or {}
    sit = _situation_from_view(view)
    d = make_decider("v5", Mode.QUALIFIER)
    acts = legal_actions(sit)
    chosen = d.choose(sit, acts, budget_ms=3000)
    print("=" * 100)
    print(f"{rid}  (seq={rid.split('seq')[-1]})")
    print(f"  手牌: {' '.join(T.to_codes(T.tiles_of(sit.hand.counts)))}  "
          f"melds={[m.kind+':'+''.join(T.to_codes(m.tiles)) for m in sit.hand.melds]}  "
          f"drawn={None if sit.drawn_tile is None else T.to_code(sit.drawn_tile)}  "
          f"总张数={T.total_tiles(sit.hand.counts)}")
    print(f"  v5实选: {chosen.describe() if chosen else None}")
    vis = sh.visible_counts(sit.hand.counts, [m.tiles for m in sit.all_melds], sit.discards)
    rows = []
    for a in [a for a in acts if a.kind == DISCARD]:
        counts = list(sit.hand.counts)
        counts[a.tile] -= 1
        mc = sit.hand.meld_count
        try:
            s = sh.shanten(counts, mc)
        except Exception as e:
            s = f"err:{type(e).__name__}"
        try:
            uk = sh.ukeire(counts, mc, visible=vis)
            tot = sum(n for _, n in uk)
        except Exception as e:
            uk, tot = (), f"err:{type(e).__name__}"
        rows.append((T.to_code(a.tile), s, len(uk) if isinstance(uk, tuple) else uk, tot,
                     ' '.join(f"{T.to_code(t)}x{n}" for t, n in uk) if isinstance(uk, tuple) else ""))
    rows.sort(key=lambda r: (r[1] if isinstance(r[1], int) else 99, -(r[3] if isinstance(r[3], int) else 0)))
    print(f"  {'牌':>4} {'向听':>4} {'种数':>4} {'枚数':>4}  进张明细")
    for code, s, k, tot, det in rows[:10]:
        print(f"  {code:>4} {str(s):>4} {str(k):>4} {str(tot):>4}  {det}")
print("=" * 100)
