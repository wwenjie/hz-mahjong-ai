"""空干预防线：**任意两个档位在真机决策点上的出牌分歧率**（B' 的门 3，先于 A/B）。

**为什么必须有它**：本项目今天吃过两次教训——
`seen-tiebreak` 靠它 10 分钟内被判掉（分歧率 ≈0 ⇒ 改动没落地）；
而 `shape-blocks` 的分歧 19.5% 才算真动手。**A/B 之前先问「它到底改了多少出牌」，
比 A/B 之后猜「正号是哪儿来的」便宜一个数量级。**

用法::

    uv run python tools/divergence_gate.py --arms unified,tenpai-wait --rooms 40 --limit 300
    uv run python tools/divergence_gate.py --arms unified,tenpai-wait --tenpai-only

判据（我们三方对齐后的口径）：**分歧率 <5% ⇒ 直接判空干预、不进 A/B**。
`--tenpai-only` 只在**我们实际打出后成听**的点上比——听口类改动必须看这一档
（非听牌点上 `winning_draws` 为空，听口那一列是退化的）。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

from majiang.cli import make_decider
from majiang.rules import win
from majiang.rules.action import DISCARD, legal_actions
from majiang.sim import replay
from majiang.strategy.policy import Mode

OUR = "u_a7f7c67bb14a"


def main() -> int:
    parser = argparse.ArgumentParser(description="两档位在真机决策点上的出牌分歧率")
    parser.add_argument("--arms", default="unified,tenpai-wait", help="两个档位名，逗号分隔")
    parser.add_argument("--rooms", type=int, default=40, help="按**房**抽样的房数（不是文件数）")
    parser.add_argument("--limit", type=int, default=300, help="最多比多少个决策点")
    parser.add_argument("--tenpai-only", action="store_true", help="只统计实际打出后成听的点")
    args = parser.parse_args()

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    if len(names) != 2:
        raise SystemExit("--arms 需要恰好两个档位名")
    deciders = [make_decider(name, Mode.QUALIFIER) for name in names]

    rooms = sorted(p for p in glob.glob("data/auto_sessions/*/events") if glob.glob(f"{p}/*.json"))
    chosen = rooms[:: max(1, len(rooms) // args.rooms)][: args.rooms]
    files: list[str] = []
    for room in chosen:
        files.extend(sorted(glob.glob(f"{room}/*.json")))

    same = differ = skipped = tenpai_points = 0
    swaps = collections.Counter()
    examples: list[str] = []
    for path in files:
        if same + differ >= args.limit:
            break
        try:
            doc = json.loads(open(path, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for state, events in replay.iter_rounds(doc):
            for event in events:
                if same + differ >= args.limit:
                    break
                if event.get("type") != "tile_discarded" or event.get("seat") != mine:
                    replay.apply_event(state, event)
                    continue
                if not state.opened or (event.get("data") or {}).get("catch_play"):
                    replay.apply_event(state, event)
                    continue
                hand = list(state.seats[mine].hand)
                if sum(hand) % 3 != 2:
                    replay.apply_event(state, event)
                    continue
                situation = state.situation_for(mine)
                actions = legal_actions(situation)
                if len([a for a in actions if a.kind == DISCARD]) < 2:
                    replay.apply_event(state, event)
                    continue
                if args.tenpai_only:
                    melds = situation.hand.meld_count
                    after = list(hand)
                    after[replay._tile_of(event.get("tile"))] -= 1  # type: ignore[index]
                    try:
                        if not win.winning_draws(after, melds):
                            replay.apply_event(state, event)
                            continue
                    except ValueError:
                        replay.apply_event(state, event)
                        continue
                    tenpai_points += 1
                picks = [d.choose(situation, actions, budget_ms=2000) for d in deciders]
                if any(p is None or p.kind != DISCARD for p in picks):
                    skipped += 1
                    replay.apply_event(state, event)
                    continue
                if picks[0].tile == picks[1].tile:
                    same += 1
                else:
                    differ += 1
                    if len(examples) < 3:
                        examples.append(
                            f"    {names[0]}={picks[0].describe()} "
                            f"vs {names[1]}={picks[1].describe()}"
                        )
                    swaps[(picks[0].tile, picks[1].tile)] += 1
                replay.apply_event(state, event)

    total = same + differ
    if not total:
        print("无样本")
        return 1
    rate = differ / total
    print(f"档位 {names[0]} vs {names[1]}：比较了 {total} 个决策点"
          f"{'（仅听牌点 ' + str(tenpai_points) + ' 个）' if args.tenpai_only else ''}")
    print(f"  相同 {same}（{same / total:.1%}）  **不同 {differ}（{rate:.1%}）**")
    print(f"  跳过 {skipped} 个（有档位没给出出牌）")
    for line in examples:
        print(line)
    print(f"\n判据：分歧率 <5% ⇒ 判**空干预**、不进 A/B；当前 {rate:.1%} ⇒ "
          f"{'**过门**' if rate >= 0.05 else '**判空干预**'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
