"""打牌层缺口：**每次出牌是否达成了该手牌的最小向听**（我们 vs 强 bot），四家精确手牌。

**为什么需要**（2026-10-09 15:30）：
- 到听速度差距稳定存在（闲-闲每巡慢 0.10→0.21 向听），**第 1 巡就有 +0.10**（那时几乎无副露）
  ⇒ 不能全归因于副露；
- 闸门原因普查（16,887 窗口）显示我们副露少的原因是 **`equal-not-allowed`（向听不变）**，
  而「向听不变也吃/碰」已证亏/中性 ⇒ 副露少本身合理；
- ⇒ 剩下要量的就是**打牌层**：打出这一张之后，是否是该手牌可达的最小向听？

口径：对每个座位的每次弃牌，`actual = shanten(打后)`、`best = min over 可打牌 shanten(打后)`
（同一手牌、同一副露数），记 `gap = actual − best`（0 = 达成最优）。
只用**不依赖任何决策器**的算术 ⇒ 四家同一把尺子、无隐藏信息。

用法::
    .venv/bin/python tools/discard_optimality.py --rooms 1200
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
    ap = argparse.ArgumentParser(description="出牌是否达成最小向听（我们 vs 强 bot）")
    ap.add_argument("--rooms", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--max-turn", type=int, default=6)
    args = ap.parse_args(argv)

    pool = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(pool) > args.rooms:
        pool = sorted(random.sample(pool, args.rooms))

    gaps: dict[str, list[int]] = collections.defaultdict(list)
    gaps_by_turn: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    rounds = 0

    for path in pool:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        names = [str(s.get("name", "?"))[:14] for s in doc.get("seats") or []]
        base = {
            seat: ("我们" if seat == mine
                   else ("强 bot" if any(k in names[seat] for k in TOP_BOTS) else "其他"))
            for seat in range(SEATS)
        }
        for span in replay.round_spans(doc):
            hands = hands_from_span(span)
            if hands is None:
                continue
            dealer = int(getattr(span, "dealer", -1))
            group = {seat: f"{base[seat]}({'庄' if seat == dealer else '闲'})" for seat in range(SEATS)}
            rounds += 1
            melds = [0] * SEATS
            draws_total = 0
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
                    turn = draws_total // 4
                    hand = hands[seat]
                    if hand[tile] > 0:
                        hand[tile] -= 1
                    if 1 <= turn <= args.max_turn and sum(hand) == 13 - 3 * melds[seat]:
                        try:
                            actual = sh.shanten(hand, melds[seat])
                            # **`best_shanten` 就是「打任意一张后的最小向听」**（它内部枚举 34 张）：
                            # 直接用它比手写 34 次枚举快 17 倍（这是本工具的第一个版本跑不动的根因）。
                            hand[tile] += 1            # 还原成「打之前」的 14 张
                            best = sh.best_shanten(hand, melds[seat])
                            hand[tile] -= 1
                        except Exception:  # noqa: BLE001
                            actual = best = None
                        if actual is not None and best is not None:
                            gap = actual - best
                            gaps[group[seat]].append(gap)
                            gaps_by_turn[(group[seat], turn)].append(gap)
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

    print(f"扫描 {len(pool)} 房 / {rounds} 局")
    print(f"\n【出牌向听缺口 gap = 实际向听 − 该手牌可达最小向听（0 = 最优）】")
    for group in ("我们(闲)", "强 bot(闲)", "其他(闲)", "我们(庄)", "强 bot(庄)"):
        values = gaps.get(group) or []
        if not values:
            continue
        suboptimal = sum(1 for v in values if v > 0)
        print(f"  {group:<12} N={len(values):<6} 均值 {statistics.mean(values):.3f}  "
              f"非最优占比 {suboptimal / len(values):.1%}  最坏 {max(values)}")
    print(f"\n【按巡目的「非最优占比」】")
    header = "  组           " + "".join(f"{t:>8}" for t in range(1, args.max_turn + 1))
    print(header)
    for group in ("我们(闲)", "强 bot(闲)", "其他(闲)"):
        row = ""
        for turn in range(1, args.max_turn + 1):
            values = gaps_by_turn.get((group, turn)) or []
            row += f"{sum(1 for v in values if v > 0) / len(values):>8.1%}" if values else "       -"
        print(f"  {group:<12} {row}")
    print("\n注：只统计「打后恰好 13−3×副露 张」的正常弃牌点；`shanten` 含财神百搭，四家同尺。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
