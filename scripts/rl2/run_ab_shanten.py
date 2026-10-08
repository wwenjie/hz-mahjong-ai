#!/usr/bin/env python
"""A/B 对拍：向听最小化规则弃牌基线 vs v5 教师。

动机（2026-10-03 01:48 决策）：winner BC 弃牌头 A/B 7.2% vs 教师 30%，
且 GPU 训练在本环境三次 epoch 1 后被静默杀死（两次卡死在 ep2 batch 100）。
但评测脚本对 hu/gang/peng/chi 已有规则兜底 ⇒ A/B 差距 100% 来自弃牌质量。

本脚本提供一个**不训练的基线**：
- 弃牌：打使向听数最小的牌；同向听按 quick_blocks 形质值次排序（v5 热路径同款）
- 响应：沿用 A/B 脚本内置规则兜底（能胡/杠/碰/吃必做）

用途：
1. 规则基线若接近 v5 ⇒ BC 要超过它才算"学到东西"，否则学习路线继续证伪
2. 给 BC 弃牌头一个可比较的非学习参照系

用法：
    PYTHONPATH=src:/home/wuwenjie01/majiang_ai/src .venv/bin/python \\
        scripts/run_ab_shanten.py --matches 40 --rounds 8 --seed 20261003
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wuwenjie01/majiang_ai/src")

from majiang.rules import shanten as shanten_module  # noqa: E402


class ShantenDiscardDecider:
    """向听最小化弃牌 + 全规则响应兜底。"""

    def __init__(self) -> None:
        self.name = "shanten-discard"
        self.last_reason = ""
        self.last_detail = {}

    def configure(self, tournament) -> None:  # noqa: ARG002
        pass

    def choose(self, situation, actions, *, budget_ms: int = 0):
        # 规则兜底（与 run_ab_v5.BCDecider 完全同序）：胡 > 杠 > 碰 > 吃
        for want in ("hu", "gang", "peng", "chi"):
            for action in actions:
                if action.kind == want:
                    self.last_reason = f"rule:{want}"
                    return action

        # 弃牌：向听最小化，ties 用 quick_blocks 的形质值
        seat = situation.seat
        hand = situation.hand
        counts = [int(hand.counts[t]) for t in range(34)]
        meld_count = len(situation.melds_for(seat))
        memo: dict = {}

        discard_actions = [a for a in actions if a.kind == "discard" and a.tile is not None]
        if not discard_actions:
            return actions[0]

        # 弃牌时点手牌为「摸牌后多一张」状态（14 张），用 shanten_any 求最小向听
        current = shanten_module.shanten_any(counts, meld_count, memo=memo)
        best = None
        best_key = None
        for a in discard_actions:
            t = a.tile
            if not (0 <= t < 34) or counts[t] <= 0:
                continue
            counts[t] -= 1
            # 打出一张后回到未摸牌张数，可用 shanten
            s = shanten_module.shanten(counts, meld_count, memo=memo)
            # quick_blocks 返回 (面子, 搭子, 对子) 类形质；用作次排序
            try:
                qb = shanten_module.quick_blocks(counts)
                shape = sum(qb) if isinstance(qb, tuple) else qb
            except Exception:
                shape = 0
            counts[t] += 1
            key = (s, -shape)
            if best_key is None or key < best_key:
                best_key = key
                best = a
        self.last_reason = f"shanten:{current}->{best_key[0] if best_key else '?'}"
        self.last_detail = {"baseline_shanten": current}
        return best if best is not None else discard_actions[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A/B：向听规则弃牌 vs v5 教师")
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args(argv)

    from majiang.sim.batch import run_batch
    from majiang.strategy.policy import Mode
    from majiang.strategy.versions import build

    def make_rule():
        return ShantenDiscardDecider()

    def make_v5():
        return build("v5", Mode.QUALIFIER)

    factories = [make_rule, make_v5, make_v5, make_v5]
    labels = ["rule", "v5", "v5", "v5"]

    print(f"对拍: {args.matches} 场 × {args.rounds} 局，seed={args.seed}", flush=True)
    print("  座位 0: 向听规则弃牌（含胡/杠/碰/吃规则兜底）", flush=True)
    print("  座位 1-3: v5 教师", flush=True)
    t0 = time.time()
    result = run_batch(
        factories,
        matches=args.matches,
        rounds=args.rounds,
        base_score=1,
        seed=args.seed,
        labels=labels,
        dealer_rotation=True,
    )
    dt = time.time() - t0
    print(f"\n用时: {dt:.1f}s\n", flush=True)

    # 与 run_ab_v5 同款输出
    print(f"结果: 场数 {result.matches}, 局数 {result.rounds}, 流局率 {result.flow_rate:.1%}", flush=True)
    for stat in result.seats:
        print(
            f"  {stat.label:>12s} 总得分 {stat.total_score:>6d}  "
            f"名次分 {stat.place_points:>5d}  白板 {stat.god_count:>4d}  "
            f"胡率 {stat.win_rate:>5.1%}  均分 {stat.average_score:>7.1f}",
            flush=True,
        )

    out = Path("records") / f"ab_shanten_rule_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps({
        "matches": result.matches,
        "rounds": result.rounds,
        "flow_rate": result.flow_rate,
        "seed": args.seed,
        "seats": [{
            "label": s.label,
            "total_score": s.total_score,
            "place_points": s.place_points,
            "god_count": s.god_count,
            "win_rate": s.win_rate,
            "average_score": s.average_score,
            "wins": s.wins,
            "rounds": s.rounds,
        } for s in result.seats],
    }, ensure_ascii=False, indent=2))
    print(f"\n记录已保存: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
