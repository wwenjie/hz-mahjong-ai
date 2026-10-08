#!/usr/bin/env python
"""Hybrid A/B：expert_ft 模型 + 胡/碰判断交给 v5 规则引擎。

动机：BC 模型存在「该胡不胡」问题；v5 规则引擎对胡/碰的判断经过实战验证。
搭配口径（2026-10-08 用户指定）：
- 响应决策点若可选 hu/peng：
  1. 先问 v5 —— v5 选 hu/peng → 直接执行（该胡就胡、该碰就碰）
  2. v5 不选 → 把 hu/peng 从候选里剔除，剩余动作交模型决定（chi/gang/pass 仍归模型）
- 其余决策（discard、无 hu/peng 的响应）全走模型。

模型侧决策器直接复用 run_ab_v7.BCV7Decider（与方案1 逐字一致）。

用法::

    PYTHONPATH=src python scripts/run_ab_v7_hybrid.py \
        --model runs/bc_v7_expert_ft.pt --seed 20261003 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # from run_ab_v7 import

import torch  # noqa: E402

from run_ab_v7 import BCV7Decider  # noqa: E402

HU_PENG = ("hu", "peng")


class HybridDecider:
    """expert_ft 模型 + v5 规则引擎接管胡/碰。"""

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        self.model_decider = BCV7Decider(model_path, device)
        from majiang.strategy.versions import build
        from majiang.strategy.policy import Mode
        self.v5 = build("v5", Mode.QUALIFIER)
        self.name = f"hybrid[{Path(model_path).stem}+v5:hu,peng]"
        self.last_reason = ""

    def configure(self, tournament):
        for d in (self.model_decider, self.v5):
            if hasattr(d, "configure"):
                d.configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        kinds = {a.kind for a in actions}
        if any(k in kinds for k in HU_PENG):
            v5_choice = self.v5.choose(situation, actions, budget_ms=budget_ms)
            if v5_choice.kind in HU_PENG:
                self.last_reason = f"v5:{v5_choice.kind}"
                return v5_choice
            # v5 不要胡/碰 → 剔除 hu/peng，剩余交模型
            remaining = [a for a in actions if a.kind not in HU_PENG]
            if remaining:
                chosen = self.model_decider.choose(situation, remaining, budget_ms=budget_ms)
                self.last_reason = f"model:{chosen.kind}(v5否决hu/peng)"
                return chosen
            self.last_reason = f"v5:{v5_choice.kind}"
            return v5_choice
        return self.model_decider.choose(situation, actions, budget_ms=budget_ms)


def run_ab_hybrid(*, model_path, matches: int, rounds: int, seed: int, device: str = "cuda"):
    """与 run_ab_v7 相同协议：hybrid 坐 seat 0 vs 3×v5，庄家轮换。"""
    from majiang.sim.batch import run_batch
    from majiang.strategy.versions import build
    from majiang.strategy.policy import Mode

    def make_hybrid():
        return HybridDecider(model_path, device)

    def make_v5():
        return build("v5", Mode.QUALIFIER)

    factories = [make_hybrid, make_v5, make_v5, make_v5]
    labels = ["hybrid", "v5", "v5", "v5"]

    return run_batch(
        factories, matches=matches, rounds=rounds, base_score=1, seed=seed,
        labels=labels, dealer_rotation=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid A/B：expert_ft + v5 接管胡/碰")
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    print(f"模型: {args.model}（胡/碰 → v5 规则引擎）")
    print(f"对拍: {args.matches} 场 × {args.rounds} 局 vs v5 教师，seed={args.seed}")
    print()

    t0 = time.perf_counter()
    result = run_ab_hybrid(
        model_path=args.model, matches=args.matches, rounds=args.rounds,
        seed=args.seed, device=args.device,
    )
    dt = time.perf_counter() - t0

    print(f"用时: {dt:.1f}s")
    print()
    print("结果:")
    print(f"  场数: {result.matches}, 局数: {result.rounds}, 流局率: {result.flow_rate:.1%}")
    print()
    for stat in result.seats:
        print(
            f"  {stat.label:>12s} 总得分 {stat.total_score:>6d}  "
            f"名次分 {stat.place_points:>5d}  白板 {stat.god_count:>4d}  "
            f"胡率 {stat.win_rate:>5.1%}  均分 {stat.average_score:>7.1f}"
        )
    print()

    out_dir = Path("records")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ab_hybrid_{Path(args.model).stem}_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "protocol": "hybrid_v5_hu_peng",
            "matches": args.matches, "rounds": args.rounds, "seed": args.seed,
            "elapsed_s": dt,
            "seats": [{"label": s.label, "total_score": s.total_score, "place_points": s.place_points,
                        "god_count": s.god_count, "wins": s.wins, "rounds": s.rounds} for s in result.seats],
        }, f, indent=2)
    print(f"记录已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
