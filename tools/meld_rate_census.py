"""副露率对拍：**真机场上三家**与我方在这同一批局里各副露多少？

**为什么需要它**（2026-10-10 A）：`tools/ab_test.py` 的默认场地是「另三座 = baseline」，
即我们自己的复制品。而我们的策略副露少 ⇒ **任何「放宽吃/碰闸门」的假设都在一个几乎不
副露的场地上被测**。`ab_test --field` 的 docstring 早已写下这条（「真机对手副露 1.093/局，
而我们自己的策略只 0.591/局」），本工具把它**在同一批真机局上直接量化**：同一局里
我方（座位可变）与三家对手的副露计数，避免跨批次口径差。

用法::
    .venv/bin/python tools/meld_rate_census.py --rooms 2000
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

from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
MELD_TYPES = ("peng", "chi", "gang")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="副露率：真机场上三家 vs 我方（同批局）")
    ap.add_argument("--rooms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", default="", help="逐房明细 JSONL（可选）")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    files = sorted(random.sample(files, min(args.rooms, len(files))))

    ours = collections.Counter()
    theirs = collections.Counter()
    # 座位偏移统计：`(seat - mine) % 4` —— 抵消「我方总在某些座」的偏置
    by_offset = [collections.Counter() for _ in range(4)]
    rounds = 0
    hits = 0
    detail: list[dict] = []

    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        hits += 1
        room_ours = 0
        room_theirs = 0
        room_rounds = 0
        for _block, (state, events) in enumerate(replay.iter_rounds(doc)):
            rounds += 1
            room_rounds += 1
            for event in events:
                kind = str(event.get("type", ""))
                seat = event.get("seat")
                if kind in MELD_TYPES and isinstance(seat, int):
                    by_offset[(seat - mine) % 4][kind] += 1
                    if seat == mine:
                        ours[kind] += 1
                        room_ours += 1
                    else:
                        theirs[kind] += 1
                        room_theirs += 1
                replay.apply_event(state, event)
        detail.append({
            "room": doc.get("room_id"),
            "rounds": room_rounds,
            "ours": room_ours,
            "theirs": room_theirs,
        })

    rounds = max(1, rounds)
    total_ours = sum(ours.values())
    total_theirs = sum(theirs.values())
    print(f"真机抽样 {len(files)} 房 → 命中我方 {hits} 房 / {rounds} 局")
    print(f"  我方 副露 {total_ours:>6}  {total_ours / rounds:.3f}/局   {dict(ours)}")
    print(f"  三家 副露 {total_theirs:>6}  {total_theirs / rounds:.3f}/局（每家 "
          f"{total_theirs / (rounds * 3):.3f}）   {dict(theirs)}")
    print(f"  **比值：三家/我方 = {total_theirs / max(1, total_ours):.2f}×**")
    print("  按座位偏移（0 = 我方，1/2/3 = 下家/对家/上家）每局每座副露：")
    for offset in range(4):
        count = sum(by_offset[offset].values())
        unit = rounds if offset == 0 else rounds
        print(f"    偏移 {offset}: {count:>6}（{count / unit:.3f}/局）"
              f"{'  ← 我方' if offset == 0 else ''}  {dict(by_offset[offset])}")
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for row in detail:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"明细: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
