#!/usr/bin/env python
"""A/B 对拍：v7 Suphx 式二分类头 BC 模型 vs v5 教师。

推理：predict_full 返回链式概率合成的 6 类动作概率 + tile 概率。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model_v7 import MahjongTransformerV7
from nnrl2.obs import situation_to_obs

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]


class BCV7Decider:
    def __init__(self, model_path: str | Path, device: str = "cuda", rule_fallback: bool = False):
        self.device = device
        self.rule_fallback = rule_fallback
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        args = ckpt.get("args", {})
        self.model = MahjongTransformerV7(
            d_model=args.get("d_model", 128),
            nhead=args.get("nhead", 8),
            num_layers=args.get("num_layers", 4),
            dim_feedforward=args.get("dim_feedforward", 512),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.name = f"bcv7[{Path(model_path).stem}]"
        self.last_reason = ""
        self.last_detail = {}

    def configure(self, tournament):
        pass

    def choose(self, situation, actions, *, budget_ms: int = 0):
        if self.rule_fallback:
            for kind in ("hu", "gang", "peng", "chi"):
                for action in actions:
                    if action.kind == kind:
                        self.last_reason = f"rule:{kind}"
                        return action
            return self._model_discard(situation, actions)
        return self._model_full(situation, actions)

    def _model_full(self, situation, actions):
        obs = situation_to_obs(situation, situation.seat)
        obs_t = {k: torch.from_numpy(np.expand_dims(v, 0)).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            a_probs, t_probs = self.model.predict_full(obs_t)
            a_probs = a_probs.squeeze(0).cpu().numpy()
            t_probs = t_probs.squeeze(0).cpu().numpy()
        kind_order = np.argsort(a_probs)[::-1]
        for kind_id in kind_order:
            kind = ACTION_NAMES[kind_id]
            if kind == "discard":
                act = self._pick_discard(actions, t_probs)
                if act is not None:
                    self.last_reason = f"model:discard(p={a_probs[kind_id]:.2f})"
                    return act
            elif kind == "pass":
                for action in actions:
                    if action.kind == "pass":
                        self.last_reason = f"model:pass(p={a_probs[kind_id]:.2f})"
                        return action
            else:
                for action in actions:
                    if action.kind == kind:
                        self.last_reason = f"model:{kind}(p={a_probs[kind_id]:.2f})"
                        return action
        self.last_reason = "fallback:first"
        return actions[0]

    def _model_discard(self, situation, actions):
        obs = situation_to_obs(situation, situation.seat)
        obs_t = {k: torch.from_numpy(np.expand_dims(v, 0)).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            probs = self.model.predict(obs_t).squeeze(0).cpu().numpy()
        return self._pick_discard(actions, probs)

    @staticmethod
    def _pick_discard(actions, probs):
        best_action = None
        best_prob = -1
        for action in actions:
            if action.kind == "discard" and action.tile is not None:
                tile = action.tile
                if 0 <= tile < 34 and probs[tile] > best_prob:
                    best_prob = probs[tile]
                    best_action = action
        return best_action


def run_ab_v7(*, model_path, matches: int, rounds: int, seed: int, device: str = "cuda", rule_fallback: bool = False):
    from majiang.sim.batch import run_batch
    from majiang.strategy.versions import build
    from majiang.strategy.policy import Mode

    def make_bc():
        return BCV7Decider(model_path, device, rule_fallback)

    def make_v5():
        return build("v5", Mode.QUALIFIER)

    factories = [make_bc, make_v5, make_v5, make_v5]
    labels = ["bcv7", "v5", "v5", "v5"]

    return run_batch(
        factories, matches=matches, rounds=rounds, base_score=1, seed=seed,
        labels=labels, dealer_rotation=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A/B 对拍：v7 Suphx 式 BC vs v5 教师")
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rule-fallback", action="store_true")
    args = ap.parse_args(argv)

    print(f"模型: {args.model}")
    print(f"对拍: {args.matches} 场 × {args.rounds} 局 vs v5 教师，seed={args.seed}, rule_fallback={args.rule_fallback}")
    print()

    t0 = time.perf_counter()
    result = run_ab_v7(
        model_path=args.model, matches=args.matches, rounds=args.rounds,
        seed=args.seed, device=args.device, rule_fallback=args.rule_fallback,
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
    suffix = "_rule" if args.rule_fallback else ""
    out_path = out_dir / f"ab_v5_{Path(args.model).stem}{suffix}_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "rule_fallback": args.rule_fallback,
            "matches": args.matches, "rounds": args.rounds, "seed": args.seed,
            "elapsed_s": dt,
            "seats": [{"label": s.label, "total_score": s.total_score, "place_points": s.place_points,
                        "god_count": s.god_count, "wins": s.wins, "rounds": s.rounds} for s in result.seats],
        }, f, indent=2)
    print(f"记录已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
