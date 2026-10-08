#!/usr/bin/env python
"""A/B 对拍：v7 多头模型（hu/gang/peng/chi/discard 独立 sigmoid + tile 头）vs v5 教师。

设计：
- 决策合成用模型自带的 predict_full（Suphx 决策流链式展开，6 类概率）
- 在合法动作类型里取合成概率最高者；discard 再用 tile 头选具体牌
- 观测：nnrl2.obs.situation_to_obs 产出 12 键，与模型 forward 完全对齐

用法（本地）:
    PYTHONPATH=/home/wuwenjie01/majiang_rl2/src:/home/wuwenjie01/majiang_ai/src \
        /home/wuwenjie01/majiang_ai/.venv/bin/python -u scripts/run_ab_v7.py \
        --model /home/wuwenjie01/majiang_rl2/runs/bc_v7_base.pt \
        --matches 3 --rounds 4 --seeds 42 --device cpu

用法（AutoDL 53838，bundle 布局 <root>/ab/{majiang_rl,majiang_rl2,majiang_ai}）:
    cd /root/autodl-tmp/ab/majiang_rl && \
        /root/miniconda3/bin/python -u scripts/run_ab_v7.py \
        --model runs/ppo_v7_v3b.pt --matches 40 --rounds 8 \
        --seeds 20261003,771014 --device cpu --out runs/ab_v3b.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# --- 路径自举：本地开发布局 + AutoDL bundle 布局 ---
_HERE = Path(__file__).resolve().parent
for _p in ("/home/wuwenjie01/majiang_rl2/src", "/home/wuwenjie01/majiang_ai/src"):
    if Path(_p).is_dir():
        sys.path.insert(0, _p)
_BUNDLE = _HERE.parent.parent  # <bundle>/majiang_rl/scripts -> <bundle>
for _p in (_BUNDLE / "majiang_rl2" / "src", _BUNDLE / "majiang_ai" / "src"):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from nnrl2.model_v7 import MahjongTransformerV7
from nnrl2.obs import situation_to_obs

# 与 model_v7 的动作常量一致
KIND_TO_IDX = {"discard": 0, "chi": 1, "peng": 2, "gang": 3, "hu": 4, "pass": 5}
SCALAR_KEYS = {"god", "wall_remaining", "turn", "phase", "target", "seat", "dealer"}


class V7Decider:
    """v7 多头模型决策器：predict_full 合成 6 类概率，合法动作里取最大。"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        margs = ckpt.get("args", {})
        self.model = MahjongTransformerV7(
            d_model=margs.get("d_model", 128), nhead=margs.get("nhead", 8),
            num_layers=margs.get("num_layers", 4),
            dim_feedforward=margs.get("dim_feedforward", 512),
            dropout=margs.get("dropout", 0.1),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.name = f"v7[{Path(model_path).stem}]"
        self.last_reason = ""
        self.last_detail: dict = {}

    def configure(self, tournament):
        pass

    def _obs_to_tensor(self, situation) -> dict[str, torch.Tensor]:
        obs = situation_to_obs(situation, situation.seat)
        obs_t = {}
        for k, v in obs.items():
            t = torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
            if k in SCALAR_KEYS:
                t = t.reshape(1)  # 标量必须是 (batch,)
            else:
                t = t.unsqueeze(0)
            obs_t[k] = t
        return obs_t

    def choose(self, situation, actions, *, budget_ms: int = 0):
        obs_t = self._obs_to_tensor(situation)
        with torch.no_grad():
            action_probs, tile_probs = self.model.predict_full(obs_t)
        action_probs = action_probs.squeeze(0).cpu().numpy()  # (6,)
        tile_probs = tile_probs.squeeze(0).cpu().numpy()      # (34,)

        # 合法动作类型里取合成概率最高者
        kinds = {a.kind for a in actions}
        avail = [(k, KIND_TO_IDX[k]) for k in kinds if k in KIND_TO_IDX]
        best_kind = max(avail, key=lambda ki: action_probs[ki[1]])[0]
        best_kind_p = float(action_probs[KIND_TO_IDX[best_kind]])

        if best_kind == "discard":
            best, best_p = None, -1.0
            for a in actions:
                if a.kind == "discard" and a.tile is not None and 0 <= a.tile < 34:
                    if float(tile_probs[a.tile]) > best_p:
                        best_p = float(tile_probs[a.tile])
                        best = a
            if best is not None:
                self.last_reason = f"model:discard(act_p={best_kind_p:.2f},tile_p={best_p:.2f})"
                return best
            # 理论上不会到这（discard 是 best_kind 说明有合法 discard）
            for a in actions:
                if a.kind == "discard":
                    self.last_reason = "fallback:discard"
                    return a
            return actions[0]

        for a in actions:
            if a.kind == best_kind:
                self.last_reason = f"model:{best_kind}(p={best_kind_p:.2f})"
                return a
        # best_kind 在 kinds 里，不会到这；兜底
        for a in actions:
            if a.kind == "pass":
                self.last_reason = "fallback:pass"
                return a
        return actions[0]


def _result_to_dict(r) -> dict:
    return {
        "matches": r.matches, "rounds": r.rounds, "flows": r.flows,
        "details": dict(r.details),
        "seats": [
            {"seat": s.seat, "label": s.label, "total_score": s.total_score,
             "place_points": s.place_points, "game_place_points": s.game_place_points,
             "god_count": s.god_count, "rounds": s.rounds, "wins": s.wins,
             "fan_total": s.fan_total, "win_rate": s.win_rate,
             "average_score": s.average_score}
            for s in r.seats
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seeds", default="20261003,771014")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from majiang.sim.batch import run_batch
    from majiang.strategy.versions import build
    from majiang.strategy.policy import Mode

    def make_v7():
        return V7Decider(args.model, args.device)

    def make_v5():
        return build("v5", Mode.QUALIFIER)

    factories = [make_v7, make_v5, make_v5, make_v5]
    labels = ["v7", "v5a", "v5b", "v5c"]

    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"=== seed {seed} ===", flush=True)
        result = run_batch(
            factories,
            matches=args.matches,
            rounds=args.rounds,
            seed=seed,
            labels=labels,
        )
        print(result, flush=True)
        results.append(result)

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump([_result_to_dict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"saved to {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
