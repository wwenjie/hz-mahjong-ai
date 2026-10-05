#!/usr/bin/env python3
"""bot 财神时序测量（A 00:35 交办，B' 03:48 顺带认领）。

数据实况（03:50 勘察）：
- 事件文件：data/auto_sessions/*/events/*.json；blocks[] 每块 = 一局的一巡段
  （block 级有 round_no / dealer / start_hands；事件在 block["events"]）
- tile 是字符串："2w"/"6b"/"3t"/"白"（财神 = "白"）
- tile_drawn / tile_discarded：seat 与 tile 在事件顶层（不在 data 里）
- round_ended：ev["seat"]=胡牌者；data 有 fan / scores / round_no / draw
- 没有 round_started 事件（start_hands 在 block 级）；没有 turn 字段
  ⇒ 巡目用「该 block 内该座位之前的 discard 数」近似

口径：
- 入手：start_hands 含 "白"（开局），或 tile_drawn tile=="白"（摸进）
- 打出：tile_discarded tile=="白"
- 结局：同 block 内 round_ended 的 winner/fan/scores

判据二分（A 00:35）：
- 早打换速度 ⇒ D 路径不同
- 同囤但更早做爆头形 ⇒ C 做牌路径
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter

GOD = "白"


def analyze_file(path: str) -> list[dict]:
    try:
        doc = json.load(open(path))
    except Exception:
        return []
    game_id = doc.get("game_id", "")
    records = []
    for blk in doc.get("blocks", []):
        round_no = blk.get("round_no")
        start_hands = blk.get("start_hands") or []
        # 每座财神状态
        st = {s: {"start": False, "drawn_turn": None, "discarded_turn": None} for s in range(4)}
        for s, hand in enumerate(start_hands):
            if hand and GOD in hand:
                st[s]["start"] = True
        # 巡目近似：每个座位已打出的牌数
        disc_count = Counter()
        winner = None
        fan = None
        scores = None
        for ev in blk.get("events", []):
            t = ev.get("type")
            seat = ev.get("seat")
            tile = ev.get("tile")
            if t == "tile_drawn" and tile == GOD and seat is not None:
                if st[seat]["drawn_turn"] is None and not st[seat]["start"]:
                    st[seat]["drawn_turn"] = disc_count[seat]
            elif t == "tile_discarded" and seat is not None:
                if tile == GOD and st[seat]["discarded_turn"] is None:
                    st[seat]["discarded_turn"] = disc_count[seat]
                disc_count[seat] += 1
            elif t == "round_ended":
                winner = ev.get("seat")
                d = ev.get("data") or {}
                fan = d.get("fan")
                scores = d.get("scores")
        # 结算本巡段
        for s in range(4):
            if not st[s]["start"] and st[s]["drawn_turn"] is None:
                continue  # 本局没摸到过财神
            rec = {
                "game_id": game_id,
                "round_no": round_no,
                "seat": s,
                "from_start": st[s]["start"],
                "discard_turn": st[s]["discarded_turn"],  # None = 整局未打
                "won": (winner == s),
                "fan": fan if winner == s else None,
                "score": scores[s] if scores else None,
            }
            records.append(rec)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", default="agent/out/bot-god-timing.txt")
    args = ap.parse_args()

    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[: args.limit]
    records = []
    for f in files:
        records.extend(analyze_file(f))
    print(f"文件 {len(files)}，财神记录 {len(records)}", flush=True)

    if not records:
        print("无记录", flush=True)
        return 1

    n = len(records)
    discarded = [r for r in records if r["discard_turn"] is not None]
    held = [r for r in records if r["discard_turn"] is None]
    n_dis = len(discarded)
    n_held = len(held)

    # 打出时序
    turns = sorted(r["discard_turn"] for r in discarded)
    med_turn = turns[len(turns) // 2] if turns else None
    mean_turn = sum(turns) / len(turns) if turns else None

    # 打出后结局 vs 未打结局
    dis_won = sum(1 for r in discarded if r["won"])
    held_won = sum(1 for r in held if r["won"])
    dis_scores = [r["score"] for r in discarded if r["score"] is not None]
    held_scores = [r["score"] for r in held if r["score"] is not None]
    dis_fans = [r["fan"] for r in discarded if r["fan"] is not None]
    held_fans = [r["fan"] for r in held if r["fan"] is not None]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    lines = [
        "=" * 64,
        "bot 财神时序测量（A 00:35 交办，B' 03:48 认领）",
        "口径：tiles=字符串，财神='白'；巡目=该座位此前 discard 数；结局=同巡段 round_ended",
        "=" * 64,
        f"样本：{len(files)} 文件 / {n} 条财神记录",
        "",
        f"打出财神：{n_dis} ({n_dis/n*100:.1f}%)　整局未打：{n_held} ({n_held/n*100:.1f}%)",
        "",
        "打出时序（该座位第几张牌打出财神）：",
        f"  中位 {med_turn}　均值 {mean_turn:.1f}　范围 {turns[0]}-{turns[-1]}" if turns else "  （无打出记录）",
        "",
        f"{'':<14}{'胡率':>10}{'均番(胡时)':>12}{'每手均分':>12}",
        f"{'打出财神':<14}{(f'{dis_won/n_dis*100:.1f}%' if n_dis else 'N/A'):>10}"
        f"{(f'{_mean(dis_fans):.2f}' if dis_fans else 'N/A'):>12}"
        f"{(f'{_mean(dis_scores):+.2f}' if dis_scores else 'N/A'):>12}",
        f"{'整局未打':<14}{(f'{held_won/n_held*100:.1f}%' if n_held else 'N/A'):>10}"
        f"{(f'{_mean(held_fans):.2f}' if held_fans else 'N/A'):>12}"
        f"{(f'{_mean(held_scores):+.2f}' if held_scores else 'N/A'):>12}",
        "",
        "判据对照（A 00:35）：",
        f"  打出 turn 中位 = {med_turn}（<5 ⇒ 早打换速度 ⇒ D 路径不同）",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
