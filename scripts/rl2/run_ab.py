#!/usr/bin/env python
"""A/B 对拍：BC 模型 vs heuristic。

用法::

    PYTHONPATH=src uv run python scripts/run_ab.py --model runs/bc_v0.pt --matches 40 --rounds 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model import MahjongTransformer
from nnrl2.obs import situation_to_obs


class BCDecider:
    """BC 模型决策器。"""

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        self.device = device
        ckpt = torch.load(model_path, map_location=device)
        args = ckpt.get("args", {})
        self.model = MahjongTransformer(
            d_model=args.get("d_model", 128),
            nhead=args.get("nhead", 8),
            num_layers=args.get("num_layers", 4),
            dim_feedforward=args.get("dim_feedforward", 512),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.name = f"bc[{Path(model_path).stem}]"
        self.last_reason = ""
        self.last_detail = {}

    def configure(self, tournament):
        pass

    def choose(self, situation, actions, *, budget_ms: int = 0):
        obs = situation_to_obs(situation, situation.seat)
        obs_t = {k: torch.from_numpy(np.expand_dims(v, 0)).to(self.device) for k, v in obs.items()}

        with torch.no_grad():
            policy_logits, _ = self.model(obs_t)
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        # 选概率最高的合法出牌
        best_action = None
        best_prob = -1
        for action in actions:
            if action.kind == "discard" and action.tile is not None:
                tile = action.tile
                if 0 <= tile < 34 and probs[tile] > best_prob:
                    best_prob = probs[tile]
                    best_action = action

        return best_action if best_action else actions[0]


def run_ab(*, model_path, matches: int, rounds: int, seed: int, device: str = "cuda"):
    """跑 A/B 对拍。"""
    from majiang.sim.batch import run_batch
    from majiang.strategy.policy import HeuristicDecider

    def make_bc():
        return BCDecider(model_path, device)

    def make_heuristic():
        return HeuristicDecider()

    factories = [make_bc, make_heuristic, make_heuristic, make_heuristic]
    labels = ["bc", "heuristic", "heuristic", "heuristic"]

    result = run_batch(
        factories,
        matches=matches,
        rounds=rounds,
        base_score=1,
        seed=seed,
        labels=labels,
        dealer_rotation=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A/B 对拍")
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    print(f"模型: {args.model}")
    print(f"对拍: {args.matches} 场 × {args.rounds} 局，seed={args.seed}")
    print()

    t0 = time.perf_counter()
    result = run_ab(
        model_path=args.model,
        matches=args.matches,
        rounds=args.rounds,
        seed=args.seed,
        device=args.device,
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

    # 保存结果
    out_dir = Path("records")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ab_{Path(args.model).stem}_{args.seed}.json"
    import json
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "matches": args.matches,
            "rounds": args.rounds,
            "seed": args.seed,
            "elapsed_s": dt,
            "seats": [
                {
                    "label": s.label,
                    "total_score": s.total_score,
                    "place_points": s.place_points,
                    "god_count": s.god_count,
                    "wins": s.wins,
                    "rounds": s.rounds,
                }
                for s in result.seats
            ],
        }, f, indent=2)
    print(f"记录已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
