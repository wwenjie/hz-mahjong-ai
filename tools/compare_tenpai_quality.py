"""诊断：**我们 vs 对手**的「首次听牌巡目 / 听口质量 / 胡率」——回答「为什么我们胡得少」。

**做法（不用模拟、不引入近似）**：房文件 `blocks[].start_hands` 给出**四家精确起手**，
按事件流逐张维护四家手牌/副露/牌河 ⇒ 每次弃牌后即时算该座位的向听；首次到 0 时记录
**巡目**（全场摸牌数/4）与**听口质量**（`_wait_copies` 可见张数 + `winning_draws` 种数）。
口径与引擎一致：**每个座位只看自己的手牌 + 公开信息**（副露 + 牌河），
所以「我们」和「对手」两边是**同一把尺子**。

输出：按「我们 / 对手」分组，并可按对手名细分（默认取出现最多的几个强 bot）。

用法::
    .venv/bin/python tools/compare_tenpai_quality.py --rooms 2000
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
from majiang.rules import win as win_mod  # noqa: E402
from majiang.rules.melds import Meld  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import _wait_copies  # noqa: E402

OUR = "u_a7f7c67bb14a"
SEATS = 4

# 与 1b/Stage A 同名单的头部 bot（用于把「对手」拆成强/弱两档——否则「对手慢」可能只是弱 bot 拉低）
TOP_BOTS = (
    "玄武", "三杯猫", "铳一色", "歪比巴卜", "爆头研究所", "Astra", "腾蛇", "康陶应雀",
    "凤凰", "glm-flash", "麒麟", "白虎", "豆包", "Nomad", "双白平胡", "Kimi", "菜菜子",
    "走马", "今晚打老虎", "晴总总",
)


def is_top(name: str) -> bool:
    return any(key in name for key in TOP_BOTS)


def hands_from_span(span) -> list[list[int]] | None:
    codes_list = list(getattr(span, "start_hands", ()) or ())
    if len(codes_list) != SEATS or any(c is None for c in codes_list):
        return None
    hands = [[0] * tiles.TILE_KINDS for _ in range(SEATS)]
    for seat, codes in enumerate(codes_list):
        for code in codes:
            hands[seat][tiles.parse(code)] += 1
    return hands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="我们 vs 对手：首次听牌巡目 / 听口质量 / 胡率")
    ap.add_argument("--rooms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    if args.rooms and len(files) > args.rooms:
        files = sorted(random.sample(files, args.rooms))

    # 每条记录：('我们'|对手名, 首次听牌巡目, 听口张数, 听口种数)
    records: list[tuple[str, int, int, int]] = []
    counters: collections.Counter = collections.Counter()
    names = collections.Counter()

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
        for name in seat_names:
            if name != seat_names[mine]:
                names[name] += 1
        rounds = {int(r.get("round_no", 0) or 0): r for r in (doc.get("rounds") or [])}
        for span in replay.round_spans(doc):
            hands = hands_from_span(span)
            if hands is None:
                counters["起始手牌不完整"] += 1
                continue
            melds: list[list[tuple[int, ...]]] = [[] for _ in range(SEATS)]
            discards: list[list[int]] = [[] for _ in range(SEATS)]
            draws_total = 0
            first: dict[int, tuple[int, int, int]] = {}
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
                    # 弃牌后即时评估该座位
                    if seat not in first:
                        try:
                            shanten = sh.shanten(hands[seat], len(melds[seat]))
                        except Exception:  # noqa: BLE001
                            shanten = None
                        if shanten == 0:
                            visible = sh.visible_counts(
                                hands[seat],
                                [m for group in melds for m in group],
                                discards,
                            )
                            copies = _wait_copies(hands[seat], len(melds[seat]), visible, tile)
                            try:
                                kinds = len(win_mod.winning_draws(hands[seat], len(melds[seat])))
                            except ValueError:
                                kinds = 0
                            if copies is not None:
                                first[seat] = (max(1, draws_total // 4), copies, kinds)
                                counters["听牌点"] += 1
                    discards[seat].append(tile)
                elif kind in ("chi", "peng", "gang", "minggang", "ming_gang") and isinstance(
                    seat, int
                ):
                    if kind == "chi":
                        run = [
                            t for t in (replay._tile_of(c) for c in (data.get("tiles") or ()))  # noqa: SLF001
                            if t is not None
                        ]
                        used = [t for t in run if t != tile]
                        for t in used:
                            if hands[seat][t] > 0:
                                hands[seat][t] -= 1
                        melds[seat].append(tuple(sorted(run)))
                    elif kind == "peng" and tile is not None:
                        for _ in range(2):
                            if hands[seat][tile] > 0:
                                hands[seat][tile] -= 1
                        melds[seat].append((tile, tile, tile))
            dealer = int(getattr(span, "dealer", -1))
            for seat, (turn, copies, kinds) in first.items():
                role = "庄" if seat == dealer else "闲"
                if seat == mine:
                    group = f"我们({role})"
                else:
                    base = "对手(强 bot)" if is_top(seat_names[seat]) else "对手(其他)"
                    group = f"{base}({role})"
                records.append((group, turn, copies, kinds))
            counters["局"] += 1

    print(f"扫描 {len(files)} 房 / {counters['局']} 局；听牌记录 {len(records)} 条"
          f"（起始手牌不完整跳过的局 {counters['起始手牌不完整']}）")
    for group in ("我们(闲)", "对手(强 bot)(闲)", "对手(其他)(闲)", "我们(庄)", "对手(强 bot)(庄)"):
        rows = [r for r in records if r[0] == group]
        if not rows:
            continue
        turns = [r[1] for r in rows]
        copies = [r[2] for r in rows]
        kinds = [r[3] for r in rows]
        print(f"\n【{group}】n={len(rows)}")
        print(f"  首次听牌巡目：均值 {statistics.mean(turns):.2f}  中位 {statistics.median(turns)}")
        print(f"  听口张数：均值 {statistics.mean(copies):.2f}  中位 {statistics.median(copies)}")
        print(f"  听口种数：均值 {statistics.mean(kinds):.2f}  中位 {statistics.median(kinds)}")
        print(f"  听口 ≤4 张占比 {sum(1 for c in copies if c <= 4) / len(copies):.1%}"
              f"；≤2 张 {sum(1 for c in copies if c <= 2) / len(copies):.1%}")
        print(f"  张数 × 种数 均值 {statistics.mean(c * k for c, k in zip(copies, kinds)):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
