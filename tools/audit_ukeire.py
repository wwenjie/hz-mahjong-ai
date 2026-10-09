"""对报障快照：算每张候选的「精确进张」（ukeire），看同分平局靠什么定夺。"""
from __future__ import annotations
import glob, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from majiang.rules import tiles as T
from majiang.rules.action import legal_actions, DISCARD
from majiang.rules import shanten as sh
from majiang.rules import win
from majiang.strategy.policy import _current_shanten
from replay_report import _situation_from_view

REPORT_DIR = os.path.join(_ROOT, "webapp", "reports")


def ukeire_after(counts, meld_count, visible):
    """返回 (kinds, copies, 到听牌? ) 。用精确 shanten.ukeire。"""
    entries = sh.ukeire(counts, meld_count, visible=visible)
    kinds = len(entries)
    copies = sum(c for _, c in entries)
    return kinds, copies


def main():
    ids = sorted(glob.glob(os.path.join(REPORT_DIR, "*.json")))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for path in ids:
        if only and only not in os.path.basename(path):
            continue
        doc = json.loads(open(path, encoding="utf-8").read())
        dec = doc.get("decision") or {}
        view = dec.get("situation") or {}
        if not view:
            continue
        sit = _situation_from_view(view)
        mc = sit.hand.meld_count
        visible = sh.visible_counts(
            sit.hand.counts,
            [m.tiles for m in sit.all_melds],
            sit.discards,
        )
        acts = [a for a in legal_actions(sit) if a.kind == DISCARD]
        # engine score
        from majiang.cli import make_decider
        from majiang.strategy.policy import Mode
        d = make_decider("v5", Mode.QUALIFIER)
        rows = []
        for a in acts:
            s = d._score_discard(sit, a)
            counts = list(sit.hand.counts); counts[s.tile] -= 1
            try:
                k, c = ukeire_after(counts, mc, visible)
            except Exception as e:
                k, c = -1, -1
            try:
                after_sh = sh.shanten_any(counts, mc)
            except Exception:
                after_sh = None
            rows.append((s, k, c, after_sh))
        rows.sort(key=lambda r: r[0].total, reverse=True)
        print("=" * 84)
        print(f"{doc['id']}  {doc.get('created_at')}")
        print("comment:", (doc.get("comment") or "")[:70].replace("\n", " "))
        print(f"hand: {' '.join(T.to_codes(sorted(T.tiles_of(sit.hand.counts))))}  "
              f"melds={[m.kind+':'+''.join(T.to_codes(m.tiles)) for m in sit.hand.melds]}")
        print(f"{'牌':>4} {'向听':>4} {'形质':>6} {'合计':>8} {'喂牌':>5} | {'进张种':>6} {'进张张':>6}")
        for s, k, c, ash in rows:
            print(f"{T.to_code(s.tile):>4} {s.shanten:>4} {s.blocks:>6.2f} {s.total:>8.2f} "
                  f"{s.feed_cost:>5.2f} | {k:>6} {c:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
