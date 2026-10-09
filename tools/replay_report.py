"""从网页「报障快照」离线复现决策点，并重跑引擎查因。

用法::

    .venv/bin/python tools/replay_report.py                     # 列全部报障
    .venv/bin/python tools/replay_report.py --id report_...      # 复现某一份
    .venv/bin/python tools/replay_report.py --last              # 复现最新一份

为什么要它：网页上点「报障」会把**当时的完整局面 + 引擎建议 + 你的备注**落成
``webapp/reports/*.json``。引擎对给定局面是**确定性**的 ⇒ 事后在本地把快照
重建成 ``Situation``、重跑引擎（及候选修复档），即可逐位复现当时的建议、
不必让用户把牌局挂在页面上。

快照来源：``webapp/session.py::GameSession.snapshot_report``。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from majiang.rules import tiles as T  # noqa: E402
from majiang.rules.action import legal_actions  # noqa: E402
from majiang.rules.god import GodState  # noqa: E402
from majiang.rules.hand import Hand  # noqa: E402
from majiang.rules.melds import parse_melds  # noqa: E402
from majiang.rules.situation import Situation  # noqa: E402
from majiang.rules.table import TableState  # noqa: E402
from majiang.cli import make_decider  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

REPORT_DIR = os.path.join(_ROOT, "webapp", "reports")


def _situation_from_view(view: dict) -> Situation:
    """把 ``situation_view`` 的 JSON 视图重建回引擎 ``Situation``。

    视图字段见 ``webapp/session.py::situation_view``；这里只做逆映射，
    不引入任何新语义（缺字段就按引擎默认值）。
    """
    my_melds = parse_melds(view.get("my_melds") or ())
    hand = Hand.from_codes(view.get("my_hand") or [], melds=my_melds)
    g = view.get("god") or {}
    god = GodState(
        hand_gods=int(g.get("hand_gods", 0)),
        chain_count=int(g.get("chain_count", 0)),
        piao_count=int(g.get("piao_count", 0)),
        baotou=bool(g.get("baotou", False)),
        catch_play=bool(g.get("catch_play", False)),
        god_discarder_seat=int(g.get("god_discarder_seat", -1)),
    )
    scores = view.get("scores")
    table = TableState(
        wall_remaining=int(view.get("wall_remaining", 0) or 0),
        dealer_seat=int(view.get("dealer_seat", 0) or 0),
        round_no=int(view.get("round_no", 1) or 1),
        scores=tuple(scores) if scores else (),
        rounds_total=int(view.get("rounds_total", 0) or 0),
    )
    discards = tuple(T.parse_all(group or ()) for group in (view.get("discards") or ()))
    melds = tuple(
        parse_melds(group or ()) for group in (view.get("melds") or ())
    )
    offered = view.get("offered_tile")
    drawn = view.get("drawn_tile")
    return Situation.from_parts(
        seat=int(view["seat"]),
        phase=str(view["phase"]),
        turn=int(view.get("turn", view["seat"])),
        hand=hand,
        god=god,
        table=table,
        responding_seats=tuple(int(s) for s in (view.get("responding_seats") or ())),
        offered_tile=None if offered is None else T.parse(offered),
        drawn_tile=None if drawn is None else T.parse(drawn),
        discards=discards,
        melds=melds,
        hand_counts=tuple(int(c) for c in (view.get("hand_counts") or ())),
    )


def _fmt_action(a) -> str:
    return a.describe() if a is not None else "<无>"


def load_reports() -> list[str]:
    return sorted(glob.glob(os.path.join(REPORT_DIR, "*.json")))


def pick(ids: list[str], args: argparse.Namespace) -> str | None:
    if not ids:
        return None
    if args.id:
        hit = [p for p in ids if os.path.basename(p).startswith(args.id)]
        return hit[0] if hit else None
    return ids[-1]  # --last / 默认取最新


def replay(path: str, arms: list[str]) -> int:
    doc = json.loads(open(path, encoding="utf-8").read())
    dec = doc.get("decision") or {}
    view = dec.get("situation") or {}
    sess = doc.get("session") or {}
    print("=" * 70)
    print(f"报障快照 : {doc.get('id')}  ({doc.get('created_at')})")
    print(f"用户备注 : {doc.get('comment') or '（空）'}")
    print(f"场次      : human_seat={sess.get('human_seat')} rounds={sess.get('rounds')} "
          f"mode={sess.get('mode')} seed={sess.get('seed')}")
    if not view:
        print("!! 快照不含决策点局面（situation），无法重放（可能是旧版快照）。")
        return 2
    print(f"决策点    : phase={dec.get('phase')} seq={dec.get('decision_seq')}")
    decider_used = (sess.get("decider") or "v5")
    print(f"网页时 {decider_used:<3} : {json.dumps((dec.get('suggestion') or {}).get('action'), ensure_ascii=False)}"
          f"  理由={((dec.get('suggestion') or {}).get('reason'))!r}")
    print("-" * 70)

    sit = _situation_from_view(view)
    actions = legal_actions(sit)
    print("合法动作  :", ", ".join(a.describe() for a in actions) or "（无）")
    print(f"手牌      : {' '.join(T.to_codes(sorted(T.tiles_of(sit.hand.counts))))}")
    print(f"副露      : {[m.kind + ':' + ''.join(T.to_codes(m.tiles)) for m in sit.hand.melds]}")
    print(f"被响应牌  : {None if sit.offered_tile is None else T.to_code(sit.offered_tile)}"
          f"   摸到的牌: {None if sit.drawn_tile is None else T.to_code(sit.drawn_tile)}")
    print("-" * 70)

    mode = Mode.QUALIFIER if (sess.get("mode") or "qualifier") == "qualifier" else Mode.FINAL
    for name in arms:
        try:
            d = make_decider(name, mode)
            chosen = d.choose(sit, actions, budget_ms=3000)
            reason = str(getattr(d, "last_reason", "") or "")
            print(f"  [{name:14s}] → {_fmt_action(chosen)}   ({reason})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name:14s}] !! {type(exc).__name__}: {exc}")
    print("=" * 70)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="复现网页报障快照并重跑引擎")
    ap.add_argument("--id", default=None, help="快照 id（可前缀匹配）")
    ap.add_argument("--last", action="store_true", help="取最新一份")
    ap.add_argument("--arms", default="v5",
                    help="要重跑的档位，逗号分隔（默认 v5；可加 v7-chibest,v7-keepchi,v7-keeppairs）")
    ap.add_argument("--list", action="store_true", help="只列出报障快照")
    args = ap.parse_args()

    ids = load_reports()
    if args.list or not ids:
        if not ids:
            print(f"（{REPORT_DIR} 下暂无报障快照）")
            return 1
        for p in ids:
            doc = json.loads(open(p, encoding="utf-8").read())
            print(f"  {os.path.basename(p):40s}  {doc.get('created_at','')}  "
                  f"{(doc.get('comment') or '')[:40]}")
        return 0

    path = pick(ids, args)
    if path is None:
        print(f"未找到匹配 {args.id!r} 的快照")
        return 1
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    return replay(path, arms)


if __name__ == "__main__":
    raise SystemExit(main())
