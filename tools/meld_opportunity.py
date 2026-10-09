"""副露差距归因：**吃/碰机会数 vs 机会→副露 转化率**（我们 vs 强 bot），四家精确手牌。

背景（2026-10-09 14:10）：我们每局副露 **0.629** vs 强 bot **1.218**（只有 52%），
到听慢 0.41–0.70 巡；但「放宽闸门（向听不变也吃）」的 EV 是 **−0.908/触发**（t −5.09）。
⇒ 本工具回答中间那个问题：**是「机会少」还是「机会来了却不吃」**。
- 若**机会数**就少 ⇒ 问题在搭子/路线（七手承诺、财神使用让我们不去做搭子）
- 若**转化率**低 ⇒ 问题在闸门判据本身

口径：对每次他家弃牌，算**下家**（吃牌窗口只给 `(discarder+1)%4`）的
「吃机会 = `chi_combinations` 非空」「碰机会 = 手里 ≥2 张」；再看该座位下一次动作是否
就是吃/碰。财神（白）不能被吃碰杠 ⇒ 跳过。**两边同一把尺子**（都从 `start_hands` 精确重建）。

用法::
    .venv/bin/python tools/meld_opportunity.py --rooms 1200
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

from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import chi_combinations  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
SEATS = 4
GOD = tiles.GOD
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
    ap = argparse.ArgumentParser(description="吃/碰机会 vs 转化率")
    ap.add_argument("--rooms", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(files) > args.rooms:
        files = sorted(random.sample(files, args.rooms))

    chi_opp: dict[str, int] = collections.Counter()
    chi_taken: dict[str, int] = collections.Counter()
    peng_opp: dict[str, int] = collections.Counter()
    peng_taken: dict[str, int] = collections.Counter()
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
            # 待判定的机会：discarder -> [(种类, 座位)]
            pending: dict[int, list[tuple[str, int]]] = {}
            for event in span.events:
                kind = event.get("type")
                seat = event.get("seat")
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                data = event.get("data") or {}
                if not isinstance(seat, int):
                    continue
                # 先结算上一张弃牌留下的机会：若本事件就是该座位吃/碰，则算「吃掉/碰掉」
                if kind in ("chi", "peng") and seat in pending:
                    for group, target in pending.pop(seat):
                        if group == "吃" and kind == "chi":
                            chi_taken[group_of[seat]] += 1
                        if group == "碰" and kind == "peng":
                            peng_taken[group_of[seat]] += 1
                if kind == "tile_drawn" and tile is not None:
                    hands[seat][tile] += 1
                elif kind == "tile_discarded" and tile is not None:
                    if hands[seat][tile] > 0:
                        hands[seat][tile] -= 1
                    pending = {}
                    chi_seat = (seat + 1) % SEATS
                    if tile != GOD:
                        if chi_combinations(hands[chi_seat], tile):
                            chi_opp[group_of[chi_seat]] += 1
                            pending.setdefault(chi_seat, []).append(("吃", tiles.TILE_KINDS))
                        for other in range(SEATS):
                            if other != seat and hands[other][tile] >= 2:
                                peng_opp[group_of[other]] += 1
                                pending.setdefault(other, []).append(("碰", tiles.TILE_KINDS))
                elif kind == "chi":
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

    print(f"扫描 {len(files)} 房 / {rounds} 局")
    print(f"\n{'组':<8}{'吃机会/局':>10}{'吃/局':>8}{'吃转化率':>10}{'碰机会/局':>11}{'碰/局':>8}{'碰转化率':>10}")
    for group in ("我们", "强 bot", "其他"):
        co, ct = chi_opp[group], chi_taken[group]
        po, pt = peng_opp[group], peng_taken[group]
        print(f"{group:<8}{co / max(1, rounds):>10.3f}{ct / max(1, rounds):>8.3f}"
              f"{ct / max(1, co):>10.1%}{po / max(1, rounds):>11.3f}{pt / max(1, rounds):>8.3f}"
              f"{pt / max(1, po):>10.1%}")
    print("\n判读：")
    print("  · 若「吃机会/局」我们≈对手、而「转化率」明显低 ⇒ 问题在**闸门判据**（该改判据）")
    print("  · 若「吃机会/局」我们就低 ⇒ 问题在**搭子/路线**（手牌没做成可吃形态）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
