"""精确进张次排序的**适用面**普查：`ukeire_max_shanten` 挡住了多少决策？

**要问的问题**：冠军档 `v5/v7` 的 `ukeire_max_shanten=3` ⇒ 当最小向听 **> 3** 时，
`_break_ties_by_ukeire` 直接 `return None` ⇒ 排序只剩「(向听, 形质, 喂牌, 索引)」，
而**形质在 40% 的点上并列**（B' 实测）⇒ 那种局面**实际是按喂牌先验在选牌**。
诊断说过「我们每巡向听从第 1 巡就落后对手 0.10~0.21」——第 1 巡正是向听最高的时候。

**本工具量三件**：
1. 我方出牌点按 `top_shanten`（最小向听层）的分布；
2. 按「本局我方第几次出牌」的分布（前几巡形态）；
3. `last_detail` 里是否出现 `tiebreak` 键（＝精确进张层**真的跑了**）⇒ 得到「被 4+ 挡掉」的占比。

用法::
    .venv/bin/python tools/ukeire_gate_census.py --rooms 300 --jobs 6
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

from majiang.cli import make_decider  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def scan(files: list[str], arm: str) -> dict:
    dec = make_decider(arm, Mode.QUALIFIER)
    stats: collections.Counter = collections.Counter()
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for _block, (state, events) in enumerate(replay.iter_rounds(doc)):
            nth = 0  # 本局我方第几次出牌（1 起）
            stats["局"] += 1
            for event in events:
                if (event.get("type") != "tile_discarded" or event.get("seat") != mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                nth += 1
                hand = list(state.seats[mine].hand)
                if sum(hand) % 3 != 2:
                    replay.apply_event(state, event)
                    continue
                try:
                    sit = state.situation_for(mine, phase=PHASE_DRAW, drawn=None)
                    actions = legal_actions(sit)
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                discards = [a for a in actions if a.kind == DISCARD]
                if len(discards) < 2:
                    replay.apply_event(state, event)
                    continue
                try:
                    dec.choose(sit, actions, budget_ms=2000)
                    detail = dict(getattr(dec, "last_detail", {}) or {})
                    scores = [dec._score_discard(sit, a) for a in discards]  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                top = min(item.shanten for item in scores)
                stats["出牌点"] += 1
                bucket = min(top, 6)
                stats[f"向听{bucket}"] += 1
                stats["nth" + ("1" if nth == 1 else "2-4" if nth <= 4 else "5+")] += 1
                ran = "tiebreak" in detail
                stats["精确层跑了" if ran else "精确层未跑"] += 1
                if not ran:
                    stats[f"未跑|向听{bucket}"] += 1
                    stats["未跑|" + ("1st" if nth == 1 else "2-4th" if nth <= 4 else "5th+")] += 1
                else:
                    stats["跑了|" + ("1st" if nth == 1 else "2-4th" if nth <= 4 else "5th+")] += 1
                replay.apply_event(state, event)
    return dict(stats)


def _worker(payload):
    return scan(*payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="精确进张次排序适用面普查")
    ap.add_argument("--rooms", type=int, default=300)
    ap.add_argument("--arm", default="v7")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunks = [files[index::args.jobs] for index in range(args.jobs)]
        stats: collections.Counter = collections.Counter()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for part in pool.map(_worker, [(chunk, args.arm) for chunk in chunks]):
                stats.update(part)
    else:
        stats = collections.Counter(scan(files, args.arm))

    points = max(1, stats["出牌点"])
    print(f"档位 {args.arm}：{stats['局']} 局 / {stats['出牌点']} 个出牌点")
    print("  按最小向听：")
    for bucket in range(7):
        key = f"向听{bucket}"
        if stats[key]:
            print(f"    {key} {stats[key]:>7}（{stats[key] / points:.1%}）"
                  f"  其中精确层未跑 {stats[f'未跑|向听{bucket}']}")
    print(f"  按我方第几次出牌：1st {stats['nth1']} / 2-4th {stats['nth2-4']} / 5+ {stats['nth5+']}")
    print(f"  **精确进张层跑了 {stats['精确层跑了']}（{stats['精确层跑了'] / points:.1%}）、"
          f"未跑 {stats['精确层未跑']}（{stats['精确层未跑'] / points:.1%}）**")
    print(f"    未跑细分：1st {stats['未跑|1st']} / 2-4th {stats['未跑|2-4th']} / 5th+ {stats['未跑|5th+']}")
    print(f"    跑了细分：1st {stats['跑了|1st']} / 2-4th {stats['跑了|2-4th']} / 5th+ {stats['跑了|5th+']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
