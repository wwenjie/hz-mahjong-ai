"""决定性审计：对每份报障快照，打印
  - v5.choose 实选（应逐位复现网页）
  - 逐候选 total（降序）
  - 两条路线的 EV（pair / meld，routes.evaluate）
  - 精确进张
并标注：实选 vs (最高total / 最高EV / 最高进张) 是否一致。
"""
from __future__ import annotations
import glob, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, _HERE)

from majiang.rules import tiles as T, shanten as sh
from majiang.rules.action import legal_actions, DISCARD
from majiang.cli import make_decider
from majiang.strategy import routes
from majiang.strategy.policy import Mode
from replay_report import _situation_from_view

REPORT_DIR = os.path.join(_ROOT, "webapp", "reports")
seen = set()
for path in sorted(glob.glob(os.path.join(REPORT_DIR, "*.json"))):
    doc = json.loads(open(path, encoding="utf-8").read())
    dec = doc.get("decision") or {}; view = dec.get("situation") or {}
    if not view: continue
    sit = _situation_from_view(view); mc = sit.hand.meld_count
    key = (tuple(sorted(T.tiles_of(sit.hand.counts))), sit.drawn_tile, sit.offered_tile,
           tuple((m.kind, tuple(m.tiles)) for m in sit.hand.melds))
    if key in seen: continue
    seen.add(key)
    d = make_decider("v5", Mode.QUALIFIER)
    acts = legal_actions(sit)
    chosen = d.choose(sit, acts, budget_ms=3000)
    web = (dec.get("suggestion") or {}).get("action") or {}
    vis = sh.visible_counts(sit.hand.counts, [m.tiles for m in sit.all_melds], sit.discards)
    rows = []
    for a in [a for a in acts if a.kind == DISCARD]:
        s = d._score_discard(sit, a)
        c = list(sit.hand.counts); c[s.tile] -= 1
        try:
            pr, mr = routes.evaluate(sit, base_score=d.config.base_score, counts=c)
        except Exception as e:
            pr = mr = None
        try:
            u = sum(x for _, x in sh.ukeire(c, mc, visible=vis))
        except Exception:
            u = -1
        rows.append((s, pr, mr, u))
    rows.sort(key=lambda r: r[0].total, reverse=True)
    print("=" * 92)
    print(f"{doc['id']}  {doc.get('created_at')}  seq={dec.get('decision_seq')}")
    print("cmt:", (doc.get("comment") or "")[:66].replace("\n", " "))
    print(f"hand={' '.join(T.to_codes(sorted(T.tiles_of(sit.hand.counts))))} "
          f"melds={[m.kind+':'+''.join(T.to_codes(m.tiles)) for m in sit.hand.melds]} "
          f"drawn={None if sit.drawn_tile is None else T.to_code(sit.drawn_tile)} "
          f"offered={None if sit.offered_tile is None else T.to_code(sit.offered_tile)}")
    print(f"v5网页={web.get('tile') or web.get('kind')}   v5复算={chosen.describe()}")
    print(f"{'牌':>4} {'向听':>3} {'total':>8} {'pairEV':>7} {'meldEV':>7} {'进张':>5}")
    for s, pr, mr, u in rows[:8]:
        pv = f"{pr.value:.2f}" if pr else "-"
        mv = f"{mr.value:.2f}" if mr else "-"
        print(f"{T.to_code(s.tile):>4} {s.shanten:>3} {s.total:>8.2f} {pv:>7} {mv:>7} {u:>5}")
    if not rows:
        print("  （无弃牌候选：实选为 pass/hu 等非出牌动作 ⇒ 跳过三问）")
        print(f"  实选={chosen.describe()}")
        continue
    # 三问：实选是否=最高total / EV / 进张
    best_total = T.to_code(rows[0][0].tile)
    best_ev = max(rows, key=lambda r: max(r[1].value if r[1] else -9e9, r[2].value if r[2] else -9e9))
    best_uke = max(rows, key=lambda r: r[3])
    ch = chosen.describe()
    print(f"  最高total={best_total} | 最高routeEV={T.to_code(best_ev[0].tile)} "
          f"| 最高进张={T.to_code(best_uke[0].tile)}({best_uke[3]}) | 实选={ch}")
print(f"\n共 {len(seen)} 个不同决策点")
