"""审计报障快照：逐候选 dump 出牌评分（只读，不改引擎）。

对 webapp/reports 下每份快照，重建 Situation，枚举全部合法 discard，
打印每张的 (shanten, blocks代理, shape_value, feed, total)，让「为什么选它」透明。
"""
from __future__ import annotations
import glob, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from majiang.rules import tiles as T
from majiang.rules.action import legal_actions, DISCARD
from majiang.rules import shanten as sh
from majiang.cli import make_decider
from majiang.strategy.policy import Mode
sys.path.insert(0, _HERE)
from replay_report import _situation_from_view

REPORT_DIR = os.path.join(_ROOT, "webapp", "reports")


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
        d = make_decider("v5", Mode.QUALIFIER)
        acts = [a for a in legal_actions(sit) if a.kind == DISCARD]
        rows = []
        for a in acts:
            s = d._score_discard(sit, a)
            rows.append(s)
        rows.sort(key=lambda x: x.total, reverse=True)
        print("=" * 78)
        print(f"{doc['id']}  {doc.get('created_at')}")
        print("comment:", (doc.get("comment") or "")[:80].replace("\n", " "))
        print(f"hand: {' '.join(T.to_codes(sorted(T.tiles_of(sit.hand.counts))))}  "
              f"melds={[m.kind+':'+''.join(T.to_codes(m.tiles)) for m in sit.hand.melds]}")
        print(f"drawn={None if sit.drawn_tile is None else T.to_code(sit.drawn_tile)}  "
              f"wall={sit.table.wall_remaining}")
        picked = (dec.get("suggestion") or {}).get("action") or {}
        print(f"v5网页={picked.get('tile')}   复算头名={T.to_code(rows[0].tile)}")
        print(f"{'牌':>4} {'向听':>4} {'blocks':>7} {'合计':>8}   {'喂牌':>5} {'路线':>5}")
        for s in rows:
            mark = " <== v5" if T.to_code(s.tile) == T.to_code(rows[0].tile) else ""
            print(f"{T.to_code(s.tile):>4} {s.shanten:>4} {s.blocks:>7.2f} {s.total:>8.2f}   "
                  f"{s.feed_cost:>5.2f} {s.route:>5}{mark}")
        # also dump quick_blocks / shape_value directly
        print("-- 结构量 --")
        for s in rows:
            counts = list(sit.hand.counts); counts[s.tile] -= 1
            qb = sh.quick_blocks(counts)
            try:
                sv = sh.shape_value(counts, sit.hand.meld_count)
            except Exception as e:
                sv = f"ERR:{e}"
            mc = sh.shape_mix(counts, sit.hand.meld_count)
            if isinstance(sv, float):
                print(f"  {T.to_code(s.tile):>4} quick_blocks={qb} shape_value={sv:.2f} shape_mix(runs,pairs,kanchan)={mc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
