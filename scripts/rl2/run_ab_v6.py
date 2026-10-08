#!/usr/bin/env python
"""A/B 对拍：v6 双头 BC 模型 vs v5 教师。

与 run_ab_v5.py 的区别：加载 MahjongTransformerV6（双头），
推理时用动作头决定是否叫牌（无规则兜底，测模型真实能力），
可选 --rule-fallback 开启与 v5 相同的规则兜底做对齐对比。

用法::

    PYTHONPATH=src python scripts/run_ab_v6.py --model runs/bc_v6.pt --matches 40 --rounds 8
    PYTHONPATH=src python scripts/run_ab_v6.py --model runs/bc_v6.pt --matches 40 --rounds 8 --rule-fallback
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

from nnrl2.model_v6 import MahjongTransformerV6
from nnrl2.obs import situation_to_obs

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]


class BCV6Decider:
    """v6 双头 BC 模型决策器。"""

    def __init__(self, model_path: str | Path, device: str = "cuda", rule_fallback: bool = False):
        self.device = device
        self.rule_fallback = rule_fallback
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        args = ckpt.get("args", {})
        self.model = MahjongTransformerV6(
            d_model=args.get("d_model", 128),
            nhead=args.get("nhead", 8),
            num_layers=args.get("num_layers", 4),
            dim_feedforward=args.get("dim_feedforward", 512),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.name = f"bcv6[{Path(model_path).stem}]"
        self.last_reason = ""
        self.last_detail = {}

    def configure(self, tournament):
        pass

    def choose(self, situation, actions, *, budget_ms: int = 0):
        if self.rule_fallback:
            # 与 run_ab_v5.py 相同的规则兜底（对齐对比用）
            for kind in ("hu", "gang", "peng", "chi"):
                for action in actions:
                    if action.kind == kind:
                        self.last_reason = f"rule:{kind}"
                        return action
            return self._model_discard(situation, actions)
        return self._model_full(situation, actions)

    def _model_full(self, situation, actions):
        """纯模型决策：动作头选动作类型，tile 头选出牌。"""
        obs = situation_to_obs(situation, situation.seat)
        obs_t = {k: torch.from_numpy(np.expand_dims(v, 0)).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            a_probs, t_probs = self.model.predict_full(obs_t)
            a_probs = a_probs.squeeze(0).cpu().numpy()
            t_probs = t_probs.squeeze(0).cpu().numpy()

        # 按动作类型置信度从高到低尝试
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
        # 模型没找到匹配动作（理论上不该发生），兜底第一个
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


def run_ab_v6(*, model_path, matches: int, rounds: int, seed: int, device: str = "cuda", rule_fallback: bool = False):
    from majiang.sim.batch import run_batch
    from majiang.strategy.versions import build
    from majiang.strategy.policy import Mode

    def make_bc():
        return BCV6Decider(model_path, device, rule_fallback)

    def make_v5():
        return build("v5", Mode.QUALIFIER)

    factories = [make_bc, make_v5, make_v5, make_v5]
    labels = ["bcv6", "v5", "v5", "v5"]

    return run_batch(
        factories,
        matches=matches,
        rounds=rounds,
        base_score=1,
        seed=seed,
        labels=labels,
        dealer_rotation=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A/B 对拍：v6 双头 BC vs v5 教师")
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rule-fallback", action="store_true", help="开启规则兜底（与 v5 评测对齐）")
    args = ap.parse_args(argv)

    print(f"模型: {args.model}")
    print(f"对拍: {args.matches} 场 × {args.rounds} 局 vs v5 教师，seed={args.seed}, rule_fallback={args.rule_fallback}")
    print()

    t0 = time.perf_counter()
    result = run_ab_v6(
        model_path=args.model,
        matches=args.matches,
        rounds=args.rounds,
        seed=args.seed,
        device=args.device,
        rule_fallback=args.rule_fallback,
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
            "model": args.model,
            "rule_fallback": args.rule_fallback,
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
