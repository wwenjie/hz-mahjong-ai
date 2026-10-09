"""对 6 个「口味分歧」报障点做分项分解：
打印每个弃牌候选的 (向听, block_value=形质, feed, god_penalty, total)。
目的：看孤张胜出是哪一项造成的。"""
from __future__ import annotations
import glob, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, _HERE)

from majiang.rules import tiles as T
from majiang.rules.action import legal_actions, DISCARD
from majiang.cli import make_decider
from majiang.strategy.policy import Mode
from replay_report import _situation_from_view

SEQS = {
    "report_20261008_233042_seq7": "v5=2b; 用户:6t8t卡张只等7t(该打8t?), 质疑打2b",
    "report_20261009_000104_seq31": "v5=9b; 用户/豆包:该打9w(舍79w卡张,留344t+67b)",
    "report_20261009_002905_seq65": "v5=1w; 用户:肯定打6b",
    "report_20261009_102559_seq71": "v5=2t; 用户:打7w(保留1t2t2t3t,听口更广)",
    "report_20261009_114836_seq141": "v5=8w; 用户/豆包:该打5w/3t/5t(孤张)",
    "report_20261009_131257_seq150": "v5=1b; DeepSeek:该打9b",
}

for rid, note in SEQS.items():
    path = os.path.join(_ROOT, "webapp", "reports", rid + ".json")
    doc = json.loads(open(path, encoding="utf-8").read())
    dec = doc.get("decision") or {}
    view = dec.get("situation") or {}
    sit = _situation_from_view(view)
    d = make_decider("v5", Mode.QUALIFIER)
    acts = legal_actions(sit)
    discs = [a for a in acts if a.kind == DISCARD]
    chosen = d.choose(sit, acts, budget_ms=3000)
    print("=" * 100)
    print(f"{rid}")
    print(f"  备注: {doc.get('comment','')[:60]}")
    print(f"  提示: {note}")
    print(f"  手牌: {' '.join(T.to_codes(T.tiles_of(sit.hand.counts)))}  "
          f"melds={[m.kind+':'+''.join(T.to_codes(m.tiles)) for m in sit.hand.melds]}  "
          f"drawn={None if sit.drawn_tile is None else T.to_code(sit.drawn_tile)}")
    print(f"  实选: {chosen.describe() if chosen else None}")
    print(f"  {'牌':>4} {'向听':>3} {'形质':>7} {'喂牌':>6} {'财神':>5} {'total':>8}")
    rows = []
    for a in discs:
        s = d._score_discard(sit, a)
        rows.append((T.to_code(a.tile), s.shanten, s.blocks, s.feed_cost, s.god_penalty, s.total))
    rows.sort(key=lambda r: r[5], reverse=True)
    for code, sh_, blk, feed, god, total in rows:
        print(f"  {code:>4} {sh_:>3} {blk:>7.2f} {feed:>6.2f} {god:>5.0f} {total:>8.2f}")
print("=" * 100)
