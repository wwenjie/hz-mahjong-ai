"""胡牌结构对拍：**我方 vs 三家**在真机同一批局里的胡牌方式（爆头 / 平胡 / 番数）。

**为什么需要它**（2026-10-10 13:5x A）：B' 早期记过「爆头率只有 bot 的 1/3」（`agent/out/s2-fan-gap.txt`），
但那是另一条线的读数。本条用 `data/auto_sessions` 的 `round_ended.data.detail` **直接复核**，
并把口径钉死在同一批局内的同座对比（避免跨批次/场强差）。

杭州麻将里 **爆头 = 听任意**（财神百搭路线的终点），一次爆头的分值远高于平胡，
故「爆头率」很可能就是「打不过」的主因之一——这条如果成立，比出牌层任何细调都值钱。

用法::
    .venv/bin/python tools/baotou_census.py --rooms 1500
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUR = "u_a7f7c67bb14a"


def main() -> int:
    ap = argparse.ArgumentParser(description="胡牌结构对拍（爆头率）")
    ap.add_argument("--rooms", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    files = sorted(random.sample(files, min(args.rooms, len(files))))

    ours = collections.Counter()
    theirs = collections.Counter()
    fan = {"ours": 0, "theirs": 0}
    wins = {"ours": 0, "theirs": 0}
    rounds = 0
    rooms = 0
    keywords = ("爆头", "财飘", "七对", "平胡", "自摸", "杠", "碰碰", "清一色")
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        rooms += 1
        mine = ids.index(OUR)
        for block in doc.get("blocks") or []:
            for event in block.get("events") or []:
                if event.get("type") != "round_ended":
                    continue
                data = event.get("data") or {}
                detail = [str(x) for x in (data.get("detail") or [])]
                seat = event.get("seat")
                is_draw = bool(data.get("draw"))
                rounds += 1
                if is_draw or not isinstance(seat, int) or seat < 0:
                    continue
                target = "ours" if seat == mine else "theirs"
                wins[target] += 1
                fan[target] += int(data.get("fan") or 0)
                ours["全部"] += 1 if target == "ours" else 0
                theirs["全部"] += 1 if target == "theirs" else 0
                for word in keywords:
                    if any(word in item for item in detail):
                        if target == "ours":
                            ours[word] += 1
                        else:
                            theirs[word] += 1
                if not detail:
                    (ours if target == "ours" else theirs)["(无 detail)"] += 1

    rounds = max(1, rounds)
    print(f"真机抽样 {len(files)} 房 → 命中我方 {rooms} 房 / {rounds} 局")
    print(f"  我方胡 {wins['ours']}（{wins['ours'] / rounds:.1%} 局）"
          f"  三家合计胡 {wins['theirs']}（每家 {wins['theirs'] / 3 / rounds:.1%} 局）")
    print(f"  平均番：我方 {fan['ours'] / max(1, wins['ours']):.3f}"
          f" vs 三家 {fan['theirs'] / max(1, wins['theirs']):.3f}")
    print("  结构占比（分母＝各自的胡牌数）：")
    for word in keywords:
        o, t = ours[word], theirs[word]
        if not o and not t:
            continue
        o_r = o / max(1, wins["ours"])
        t_r = t / max(1, wins["theirs"])
        ratio = f"{t_r / o_r:.2f}×" if o_r else "∞"
        print(f"    {word:<6} 我方 {o:>6}（{o_r:6.1%}）  三家 {t:>6}（{t_r:6.1%}）"
              f"  **三家/我方 {ratio}**")
    if ours["(无 detail)"] or theirs["(无 detail)"]:
        print(f"    (无 detail) 我方 {ours['(无 detail)']} / 三家 {theirs['(无 detail)']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
