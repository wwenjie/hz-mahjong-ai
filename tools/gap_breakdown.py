"""速度差距的细分解：**按巡目的累计到听率** + **副露率**（我们 vs 强 bot），同一把尺子。

延续 `tools/compare_tenpai_quality.py`（那里给出「闲局慢 0.41 巡、庄局慢 0.70 巡」）。
本工具回答「**差在哪儿**」：
1. 累计到听率曲线（第 1..12 巡末，各家至少听牌一次的累计比例）——看我们是从第几巡开始掉队；
2. 平均副露数/局（副露是提速的主要手段，我们在线上的副露率只有强 bot 的一半）；
3. 到听者中「听牌前已副露」的比例——检验「副露⇒更快」这条机制在我们身上是否成立。

用法::
    .venv/bin/python tools/gap_breakdown.py --rooms 1200
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import shanten as sh  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
SEATS = 4
TOP_BOTS = (
    "玄武", "三杯猫", "铳一色", "歪比巴卜", "爆头研究所", "Astra", "腾蛇", "康陶应雀",
    "凤凰", "glm-flash", "麒麟", "白虎", "豆包", "Nomad", "双白平胡", "Kimi", "菜菜子",
    "走马", "今晚打老虎", "晴总总",
)


def is_top(name: str) -> bool:
    return any(key in name for key in TOP_BOTS)


def hands_from_span(span):
    codes_list = list(getattr(span, "start_hands", ()) or ())
    if len(codes_list) != SEATS or any(c is None for c in codes_list):
        return None
    hands = [[0] * tiles.TILE_KINDS for _ in range(SEATS)]
    for seat, codes in enumerate(codes_list):
        for code in codes:
            hands[seat][tiles.parse(code)] += 1
    return hands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="速度差距细分解：累计到听率 + 副露率")
    ap.add_argument("--rooms", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(files) > args.rooms:
        files = sorted(random.sample(files, args.rooms))

    first_turn: dict[str, list[int]] = collections.defaultdict(list)   # 首次到听巡目
    meld_count: dict[str, list[int]] = collections.defaultdict(list)   # 每局副露数（按座位组）
    melded_before: dict[str, list[int]] = collections.defaultdict(list)  # 到听时已副露？（1/0）
    rounds = 0

    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        seat_names = [str(s.get("name", "?"))[:14] for s in doc.get("seats") or []]
        group_of = {
            seat: ("我们" if seat == mine else ("强 bot" if is_top(seat_names[seat]) else "其他"))
            for seat in range(SEATS)
        }
        for span in replay.round_spans(doc):
            hands = hands_from_span(span)
            if hands is None:
                continue
            rounds += 1
            melds = [0] * SEATS
            first: dict[int, tuple[int, int]] = {}   # seat -> (巡, 听牌时副露数)
            draws_total = 0
            for event in span.events:
                kind = event.get("type")
                seat = event.get("seat")
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                data = event.get("data") or {}
                if kind == "tile_drawn" and isinstance(seat, int) and tile is not None:
                    hands[seat][tile] += 1
                    draws_total += 1
                elif kind == "tile_discarded" and isinstance(seat, int) and tile is not None:
                    if hands[seat][tile] > 0:
                        hands[seat][tile] -= 1
                    if seat not in first:
                        try:
                            if sh.shanten(hands[seat], melds[seat]) == 0:
                                first[seat] = (max(1, draws_total // 4), melds[seat])
                        except Exception:  # noqa: BLE001
                            pass
                elif kind in ("chi", "peng", "gang", "minggang", "ming_gang") and isinstance(
                    seat, int
                ):
                    melds[seat] += 1
                    if kind == "chi":
                        run = [
                            t for t in (replay._tile_of(c) for c in (data.get("tiles") or ()))  # noqa: SLF001
                            if t is not None
                        ]
                        for t in [x for x in run if x != tile]:
                            if hands[seat][t] > 0:
                                hands[seat][t] -= 1
                    elif kind == "peng" and tile is not None:
                        for _ in range(2):
                            if hands[seat][tile] > 0:
                                hands[seat][tile] -= 1
            for seat in range(SEATS):
                group = group_of[seat]
                meld_count[group].append(melds[seat])
                if seat in first:
                    turn, melded = first[seat]
                    first_turn[group].append(turn)
                    melded_before[group].append(1 if melded > 0 else 0)

    print(f"扫描 {len(files)} 房 / {rounds} 局")
    print(f"\n【每局副露数（含吃碰杠）】")
    for group in ("我们", "强 bot", "其他"):
        vals = meld_count.get(group) or []
        if vals:
            print(f"  {group:<8} n={len(vals):<6} 均值 {sum(vals) / len(vals):.3f}")
    print(f"\n【累计到听率：截至第 T 巡，该组至少听牌一局的累计比例】")
    horizons = list(range(2, 13))
    print("  组       " + "".join(f"{h:>6}" for h in horizons))
    for group in ("我们", "强 bot", "其他"):
        vals = first_turn.get(group) or []
        if not vals:
            continue
        row = "".join(f"{sum(1 for v in vals if v <= h) / len(vals) * 100:>5.1f}%" for h in horizons)
        print(f"  {group:<8} {row}")
    print(f"\n【到听时「已副露」的比例】")
    for group in ("我们", "强 bot", "其他"):
        vals = melded_before.get(group) or []
        if vals:
            print(f"  {group:<8} {sum(vals) / len(vals):.1%}  （n={len(vals)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
