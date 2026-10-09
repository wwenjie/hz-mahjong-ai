"""早期构建差距：**按巡目的平均向听**曲线（我们 vs 强 bot vs 其他），四家精确手牌。

背景（2026-10-09 14:10/15:00）：到听差距**从第 2 巡就出现**（累计到听率 12.3% vs 15.5%），
而「吃」与「碰」的放宽闸门两条轴都已关闭（−0.790 / 见碰的对拍）⇒ 差距很可能在**早期打牌层**
（我们怎么在头几巡构建手牌），不是副露决策。

口径：与 `gap_breakdown.py` 相同（`start_hands` + 真实事件流维护四家手牌），
每次弃牌后算该座位向听（`shanten`，张数按 13−3×副露），按巡目（全场摸牌数//4）与座位分组平均。
**只统计到该座位和牌/流局前的巡目**（和牌后不再计），并单独给「到听前一巡」的对照。

用法::
    .venv/bin/python tools/shanten_progress.py --rooms 1200
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics
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
    ap = argparse.ArgumentParser(description="按巡目的平均向听（我们 vs 强 bot）")
    ap.add_argument("--rooms", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--max-turn", type=int, default=10)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(files) > args.rooms:
        files = sorted(random.sample(files, args.rooms))

    curve: dict[str, dict[int, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    meld_curve: dict[str, dict[int, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
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
        group_base = {
            seat: ("我们" if seat == mine else ("强 bot" if any(
                k in seat_names[seat] for k in TOP_BOTS) else "其他"))
            for seat in range(SEATS)
        }

        for span in replay.round_spans(doc):
            hands = hands_from_span(span)
            if hands is None:
                continue
            # **庄闲分离**：庄家多摸一张（14 张）⇒ 打完第一张后向听天然更低；我们只有 25% 的
            # 庄局样本、对手 75% ⇒ 不分层会把这个结构性优势算成「对手更快」。
            dealer = int(getattr(span, "dealer", -1))
            by_seat = {seat: f"{group_base[seat]}({'庄' if seat == dealer else '闲'})"
                       for seat in range(SEATS)}
            rounds += 1
            melds = [0] * SEATS
            draws_total = 0
            done: set[int] = set()   # 已和牌（或已无意义）
            for event in span.events:
                kind = event.get("type")
                seat = event.get("seat")
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                data = event.get("data") or {}
                if not isinstance(seat, int):
                    continue
                if kind == "tile_drawn" and tile is not None:
                    hands[seat][tile] += 1
                    draws_total += 1
                elif kind == "tile_discarded" and tile is not None:
                    if hands[seat][tile] > 0:
                        hands[seat][tile] -= 1
                    turn = draws_total // 4
                    if seat not in done and 1 <= turn <= args.max_turn:
                        try:
                            value = sh.shanten(hands[seat], melds[seat])
                        except Exception:  # noqa: BLE001
                            value = None
                        if value is not None:
                            curve[by_seat[seat]][turn].append(value)
                            meld_curve[by_seat[seat]][turn].append(melds[seat])
                elif kind in ("chi", "peng", "gang", "minggang", "ming_gang"):
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

    print(f"扫描 {len(files)} 房 / {rounds} 局")
    print(f"\n【按巡目的**平均向听**（越小越快）】")
    header = "  组       " + "".join(f"{t:>6}" for t in range(1, args.max_turn + 1))
    print(header)
    for group in ("我们(闲)", "强 bot(闲)", "其他(闲)", "我们(庄)", "强 bot(庄)"):
        row = ""
        for t in range(1, args.max_turn + 1):
            vals = curve[group].get(t) or []
            row += f"{statistics.mean(vals):>6.2f}" if vals else "     -"
        print(f"  {group:<8} {row}")
    print(f"\n【按巡目的**平均副露数**】")
    print(header)
    for group in ("我们(闲)", "强 bot(闲)", "其他(闲)", "我们(庄)", "强 bot(庄)"):
        row = ""
        for t in range(1, args.max_turn + 1):
            vals = meld_curve[group].get(t) or []
            row += f"{statistics.mean(vals):>6.2f}" if vals else "     -"
        print(f"  {group:<8} {row}")
    print("\n注：向听用 `shanten`（含财神百搭）；样本为该巡目仍在该局中的观测数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
