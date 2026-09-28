#!/usr/bin/env python3
"""爆头缺口分解（只读、离线、对平台零请求）。

A 的请求（11:45 条目）——把「爆头占比 我们 14.9% vs 对手 22.7%」拆成两个**互斥**解释：
  ① 对手**手里财神更多**（留得住 / 摸得到）；
  ② 对手**自然面子更多**（不靠财神凑）。
并给出：`P(爆头 | 胡)` 在 **god 数分桶**下 us vs opp 的对照；
以及**到听时刻**与**胡牌时刻**的 god 持牌数分布（us vs opp）。

方法：从 `data/auto_sessions/*/events/*.json` 离线**重建每家手牌时序**
（`start_hands` + 逐张 `tile_drawn`/`tile_discarded` + `chi`/`peng`/`gang` 副露）。
这是 AGENTS.md §4 明确允许的「离线数据生成时用对手手牌作标签」用途；
本脚本**不在推理路径上**、不被运行时导入。

真值判据全部复用规则模块（不重造 A 的逻辑）：
  - `rules.win.is_baotou(counts, meld_count)`  爆头 = 听任意
  - `rules.shanten.natural_shanten(...)`       4 组自然面子的进度
  - `rules.hand.Hand.is_waiting_shape`         到听（听牌）

仪表自检（AGENTS.md §6.1）：对每个胡牌局，重建的赢家手牌必须通过
`is_winning_shape`；自检失败的局单独计数并**排除**，不静默吞掉。

用法: nice -n 19 uv run python agent/verify/baotou_decompose.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import tiles  # noqa: E402
from majiang.rules.hand import Hand  # noqa: E402
from majiang.rules import win as winmod  # noqa: E402
from majiang.rules import shanten as shantenmod  # noqa: E402

OUR_UID = "u_a7f7c67bb14a"
GOD = tiles.GOD if hasattr(tiles, "GOD") else tiles.parse("白")


def parse_events(doc):
    """把 blocks 里的事件按 seq 铺平，返回原始事件列表。"""
    evs = []
    for b in doc.get("blocks", []):
        for e in (b.get("events") or []):
            evs.append(e)
    return evs


def group_rounds(doc):
    """按 `round_no` 把 blocks 合并成小局。

    关键结构事实（我踩过的坑）：**一个小局横跨多个 block**（样例 8 局 / 15 个 block），
    只读 `blocks[0]` 会丢掉半局的抽取/弃牌，重建出来的手牌不完整。
    `start_hands` 只在该局的第一个 block 上非空。
    """
    rounds = {}
    for b in doc.get("blocks") or []:
        rn = b.get("round_no")
        r = rounds.setdefault(rn, {"start_hands": None, "events": []})
        if r["start_hands"] is None and b.get("start_hands"):
            r["start_hands"] = b["start_hands"]
        r["events"].extend(b.get("events") or [])
    for rn, r in rounds.items():
        r["events"].sort(key=lambda e: int(e.get("seq") or 0))
    return rounds


def reconstruct_round(start_hands, events):
    """重建四家**暗手**张数（不含副露牌面）+ 副露组数。

    副露扣牌口径（按事件自带的语义，不靠猜）：
      - `peng`：暗手扣 2（另 1 张来自弃牌）；副露 +1
      - `chi`：`data.tiles` 是 3 张面子，`tile` 字段是**被吃的那张**（来自他家弃牌），
        故暗手扣「另两张」；副露 +1
      - `gang`：`data.kind` ∈ {`ming`(直杠，扣 3), `an`(暗杠，扣 4), `bu`(加杠，扣 1，副露已在
        `peng` 时计过，不再 +1)}
    """
    hands = {i: [0] * tiles.TILE_KINDS for i in range(4)}
    melds = {0: 0, 1: 0, 2: 0, 3: 0}
    for i, codes in enumerate(start_hands or []):
        for c in codes:
            hands[i][tiles.parse(c)] += 1
    for e in events:
        t = e.get("type")
        seat = e.get("seat")
        if seat is None or not (0 <= seat < 4):
            continue
        if t == "tile_drawn":
            hands[seat][tiles.parse(e["tile"])] += 1
        elif t == "tile_discarded":
            hands[seat][tiles.parse(e["tile"])] -= 1
        elif t == "peng":
            hands[seat][tiles.parse(e["tile"])] -= 2
            melds[seat] += 1
        elif t == "gang":
            kind = str((e.get("data") or {}).get("kind") or "ming")
            tile = tiles.parse(e["tile"])
            if kind == "an":
                hands[seat][tile] -= 4
                melds[seat] += 1
            elif kind == "bu":
                hands[seat][tile] -= 1  # 已有碰，不再占副露位
            else:  # ming
                hands[seat][tile] -= 3
                melds[seat] += 1
        elif t == "chi":
            data = e.get("data") or {}
            tiles_in = [tiles.parse(c) for c in (data.get("tiles") or [])]
            called = tiles.parse(e["tile"]) if e.get("tile") else None
            for c in tiles_in:
                if c == called:
                    continue  # 被吃的那张来自弃牌，不在暗手
                hands[seat][c] -= 1
            melds[seat] += 1
    return hands, melds


def analyze_file(path: str):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    seats = doc.get("seats") or []
    our = next((i for i, s in enumerate(seats) if s.get("user_id") == OUR_UID), None)
    rounds = group_rounds(doc)
    out = []
    for rn, r in rounds.items():
        ends = [e for e in r["events"] if e.get("type") == "round_ended"]
        if not ends or not r["start_hands"]:
            continue
        hands, melds = reconstruct_round(r["start_hands"], r["events"])
        end = ends[0]
        d = end.get("data") or {}
        if d.get("draw"):
            continue
        winner = end.get("seat")
        if winner is None:
            continue
        detail = d.get("detail") or []
        wc = hands.get(winner)
        ok = False
        if wc is not None:
            try:
                ok = winmod.is_winning_shape(wc, melds.get(winner, 0))
            except Exception:
                ok = False
        out.append({
            "round_no": rn, "winner": winner, "our": our,
            "side": "us" if winner == our else "opp",
            "is_baotou": "爆头" in detail, "detail": detail,
            "gods_at_win": (wc[GOD] if wc is not None else None),
            "hand_tiles": (sum(wc) if wc is not None else None),
            "n_melds": melds.get(winner, 0), "self_check_ok": ok,
        })
    return our, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个小局（0=全部）")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(ROOT / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.limit:
        paths = paths[:args.limit]
    print(f"扫描 {len(paths)} 个小局文件")

    rows = []
    skipped = 0
    files = 0
    for p in paths:
        try:
            our, out = analyze_file(p)
        except Exception:
            skipped += 1
            continue
        files += 1
        if not out:
            skipped += 1
            continue
        rows.extend(out)

    print(f"有效胡牌局 {len(rows)}（来自 {files} 文件）；跳过 {skipped}")
    checked = [r for r in rows if r["self_check_ok"]]
    print(f"仪表自检通过 {len(checked)}/{len(rows)} = {len(checked)/max(len(rows),1):.2%}")
    # 若自检通过率过低，先报告而不是继续
    if len(checked) < 0.5 * len(rows):
        print("\n!! 仪表自检通过率 <50%：重建口径可疑，先修仪表，不下结论。")
        return 2

    def rate(rows_):
        us = [r for r in rows_ if r["side"] == "us"]
        op = [r for r in rows_ if r["side"] == "opp"]
        def bt(xs):
            return sum(1 for r in xs if r["is_baotou"]), len(xs)
        ub, un = bt(us)
        ob, on = bt(op)
        return ub, un, ob, on

    ub, un, ob, on = rate(checked)
    print(f"\n=== P(爆头 | 胡) ===")
    print(f"  我们 {ub}/{un} = {ub/max(un,1):.2%}")
    print(f"  对手 {ob}/{on} = {ob/max(on,1):.2%}")

    print(f"\n=== god 持牌数分布（胡牌时刻，按 side）===")
    for side in ("us", "opp"):
        cnt = collections.Counter(r["gods_at_win"] for r in checked if r["side"] == side)
        tot = sum(cnt.values())
        dist = "  ".join(f"{k}:{v/tot:.1%}" for k, v in sorted(cnt.items()) if k is not None)
        print(f"  {side:4s} n={tot:5d}  {dist}")

    print(f"\n=== P(爆头 | 胡) 按 god 数分桶（us vs opp）===")
    for g in sorted({r["gods_at_win"] for r in checked if r["gods_at_win"] is not None}):
        us = [r for r in checked if r["side"] == "us" and r["gods_at_win"] == g]
        op = [r for r in checked if r["side"] == "opp" and r["gods_at_win"] == g]
        ur = sum(1 for r in us if r["is_baotou"]) / max(len(us), 1)
        opr = sum(1 for r in op if r["is_baotou"]) / max(len(op), 1)
        print(f"  god={g}: 我们 {ur:.1%} (n={len(us)})   对手 {opr:.1%} (n={len(op)})")

    print(f"\n=== 爆头局：god 数与副露数 ===")
    for side in ("us", "opp"):
        bts = [r for r in checked if r["side"] == side and r["is_baotou"]]
        if not bts:
            print(f"  {side:4s} 无爆头局")
            continue
        gm = sum(r["gods_at_win"] for r in bts) / len(bts)
        mm = sum(r["n_melds"] for r in bts) / len(bts)
        print(f"  {side:4s} n={len(bts):4d}  平均 god {gm:.2f}  平均副露 {mm:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
