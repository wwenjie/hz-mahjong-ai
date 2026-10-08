#!/usr/bin/env python
"""A/B 对拍：BC 模型 vs heuristic（完整版）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model import MahjongTransformer
from nnrl2.obs import situation_to_obs


class BCDecider:
    def __init__(self, model_path, device="cuda"):
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
        obs_t = {}
        for k, v in obs.items():
            val = np.expand_dims(v, 0)
            obs_t[k] = torch.from_numpy(val).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(obs_t)
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        best_action = None
        best_prob = -1
        for action in actions:
            if action.kind == "discard" and action.tile is not None:
                tile = action.tile
                if 0 <= tile < 34 and probs[tile] > best_prob:
                    best_prob = probs[tile]
                    best_action = action

        return best_action if best_action else actions[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from majiang.sim.batch import run_batch
    from majiang.strategy.policy import HeuristicDecider

    def make_bc():
        return BCDecider(args.model, args.device)

    def make_heuristic():
        return HeuristicDecider()

    factories = [make_bc, make_heuristic, make_heuristic, make_heuristic]
    labels = ["bc", "heuristic", "heuristic", "heuristic"]

    print(f"对拍: {args.matches} 场 × {args.rounds} 局, seed={args.seed}")
    t0 = time.perf_counter()
    result = run_batch(factories, matches=args.matches, rounds=args.rounds, base_score=1, seed=args.seed, labels=labels, dealer_rotation=True)
    dt = time.perf_counter() - t0

    print(f"用时: {dt:.1f}s")
    print(f"场数: {result.matches}, 局数: {result.rounds}, 流局率: {result.flow_rate:.1%}")
    print()
    for stat in result.seats:
        print(f"  {stat.label:>12s} 总得分 {stat.total_score:>6d}  名次分 {stat.place_points:>5d}  胡率 {stat.win_rate:>5.1%}  均分 {stat.average_score:>7.1f}")

    # 保存结果
    out_dir = Path("records")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ab_{Path(args.model).stem}_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "matches": args.matches, "rounds": args.rounds,
            "seed": args.seed, "elapsed_s": dt,
            "seats": [{"label": s.label, "total_score": s.total_score, "place_points": s.place_points, "wins": s.wins, "rounds": s.rounds} for s in result.seats],
        }, f, indent=2)
    print(f"记录: {out_path}")


if __name__ == "__main__":
    main()
