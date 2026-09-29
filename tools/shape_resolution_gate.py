"""v4 的机制门：**弃牌结构 + 形质分辨率**（验 `docs/ops.md` 里换档时写下的可证伪预测）。

**为什么需要它**：v4 的采纳依据是自对弈 A/B（n=10、合并名次分 +0.2732、t+4.66），
但 `shape_value` 的机制门**没有独立复算**——这是本仓第一次「A/B 先行、机制待验」的换档。
所以换档时我把两条**可证伪的预测**写进了 `docs/ops.md`：

1. **弃牌结构应向强 bot 画像靠拢**：中张占比从 ~19.6% 上升（强 bot 29.7%；
   逐房配对差 −11.2pp ± 0.4 是当前最大的指纹差之一）。
2. **同向听内的形质分辨率应上升**：`quick_blocks` 的裁剪恒饱和 ⇒ 同向听候选
   77~93% 完全并列，v4 若真在给形质项恢复分辨力，**并列度必须下降**。

本仪器在**真机决策点**上同时问 v3 与 v4，逐点比这两件事。
**它不动任何策略**，只回答「换了 v4 之后机制动没动」——若两条都没动，
说明 A/B 的正号另有来源，需按 `ops.md` 回头重审本档。

用法::

    uv run python tools/shape_resolution_gate.py --rooms 25 --limit 300
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re

from majiang.cli import make_decider
from majiang.rules import tiles
from majiang.rules.action import DISCARD, legal_actions
from majiang.sim import replay
from majiang.strategy.policy import Mode

OUR = "u_a7f7c67bb14a"
TOTAL_RE = re.compile(r"合计=(-?[\d.]+)")


def tile_class(tile: int) -> str:
    """字 / 边 / 中——与 `risk.visible_need` 的分类同源（它决定了喂牌代价）。"""
    if not tiles.is_number(tile):
        return "字"
    return "中" if 3 <= tiles.rank(tile) <= 7 else "边"


def top_ties(decider) -> int:
    """本决策点上「与最高分并列」的候选数（只看 `last_detail` 落盘的前 4 名）。"""
    rows = (getattr(decider, "last_detail", {}) or {}).get("discards") or []
    totals = [float(m.group(1)) for row in rows if (m := TOTAL_RE.search(row))]
    if not totals:
        return 0
    best = max(totals)
    return sum(1 for value in totals if abs(value - best) < 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="v4 的机制门：弃牌结构 + 形质分辨率")
    parser.add_argument("--rooms", type=int, default=25, help="按**房**抽样的房数")
    parser.add_argument("--limit", type=int, default=300, help="最多比多少个决策点")
    parser.add_argument("--baseline", default="tenpai-wait", help="对照档位（默认 v3）")
    parser.add_argument("--treatment", default="v4", help="受测档位（默认 v4）")
    args = parser.parse_args()

    base = make_decider(args.baseline, Mode.QUALIFIER)
    treat = make_decider(args.treatment, Mode.QUALIFIER)

    rooms = sorted(p for p in glob.glob("data/auto_sessions/*/events") if glob.glob(f"{p}/*.json"))
    chosen = rooms[:: max(1, len(rooms) // args.rooms)][: args.rooms]
    files: list[str] = []
    for room in chosen:
        files.extend(sorted(glob.glob(f"{room}/*.json")))

    mix = {args.baseline: collections.Counter(), args.treatment: collections.Counter()}
    tie_hist = {args.baseline: collections.Counter(), args.treatment: collections.Counter()}
    points = differ = 0
    for path in files:
        if points >= args.limit:
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
                if points >= args.limit:
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
                picks = []
                for decider in (base, treat):
                    choice = decider.choose(situation, actions, budget_ms=2000)
                    if choice is None or choice.kind != DISCARD:
                        picks = []
                        break
                    picks.append(choice)
                    mix[decider is base and args.baseline or args.treatment][
                        tile_class(choice.tile)
                    ] += 1
                    tie_hist[decider is base and args.baseline or args.treatment][
                        top_ties(decider)
                    ] += 1
                if not picks:
                    replay.apply_event(state, event)
                    continue
                points += 1
                if picks[0].tile != picks[1].tile:
                    differ += 1
                replay.apply_event(state, event)

    if not points:
        print("无样本")
        return 1

    def share(counter: collections.Counter) -> str:
        total = sum(counter.values()) or 1
        return "  ".join(
            f"{cls} {counter[cls] / total:5.1%}" for cls in ("字", "边", "中")
        )

    print(f"决策点 {points} 个（按房抽 {len(chosen)} 房）；{args.treatment} 与 {args.baseline} "
          f"分歧 {differ}（{differ / points:.1%}）\n")
    print("① 弃牌结构（强 bot 画像＝字30.6 / 边39.7 / **中29.7**；我们 v3 时期＝字33.5 / 边46.9 / 中19.6）")
    for name in (args.baseline, args.treatment):
        print(f"   {name:14s} {share(mix[name])}")
    delta = (
        mix[args.treatment]["中"] / (sum(mix[args.treatment].values()) or 1)
        - mix[args.baseline]["中"] / (sum(mix[args.baseline].values()) or 1)
    )
    print(f"   ⇒ 中张占比变化 {delta:+.2%}（预测①要求**上升**；升到 29.7% 需要 +10pp）")

    print("\n② 同向听内的形质分辨率（并列度：与最高分并列的候选数，越大越坏）")
    for name in (args.baseline, args.treatment):
        hist = tie_hist[name]
        total = sum(hist.values()) or 1
        joined = "  ".join(f"{k}名 {hist[k] / total:5.1%}" for k in sorted(hist))
        mean_ties = sum(k * v for k, v in hist.items()) / total
        print(f"   {name:14s} {joined}   平均 {mean_ties:.2f}")
    mean_base = sum(k * v for k, v in tie_hist[args.baseline].items()) / (
        sum(tie_hist[args.baseline].values()) or 1
    )
    mean_treat = sum(k * v for k, v in tie_hist[args.treatment].items()) / (
        sum(tie_hist[args.treatment].values()) or 1
    )
    print(f"   ⇒ 平均并列度变化 {mean_treat - mean_base:+.2f}（预测②要求**下降**）")

    print(
        "\n判读：两条预测都动 ⇒ v4 的机制故事成立；"
        "**两条都没动 ⇒ A/B 的正号另有来源，按 docs/ops.md 回头重审本档**。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
