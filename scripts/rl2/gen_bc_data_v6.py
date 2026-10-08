#!/usr/bin/env python
"""生成 BC 数据集 v6：记录教师的全量动作（discard/chi/peng/gang/hu/pass）。

与 gen_bc_data_v5.py 的区别：v5 只记录 discard 决策（y=tile 0-33），
v6 记录每一次 choose 调用的完整决策：
  - y_action: 6 类动作（0=discard, 1=chi, 2=peng, 3=gang, 4=hu, 5=pass）
  - y_tile:   出牌/杠/吃的目标牌 0-33，非 tile 相关动作记 34（padding 类）

用法::

    PYTHONPATH=src python scripts/gen_bc_data_v6.py --matches 100 --out data/bc_v6_train.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.obs import situation_to_obs, obs_to_flat

# 动作类型 → 类别 id
ACTION_KIND_TO_ID = {
    "discard": 0,
    "chi": 1,
    "peng": 2,
    "gang": 3,
    "hu": 4,
    "pass": 5,
}
NUM_ACTION_KINDS = 6
TILE_PAD = 34  # 非 tile 动作的占位


class BCDataV6:
    def __init__(self):
        self.obs_list: list[dict[str, np.ndarray]] = []
        self.y_action: list[int] = []
        self.y_tile: list[int] = []

    def __len__(self):
        return len(self.obs_list)

    def add(self, obs, action_kind: int, tile: int):
        self.obs_list.append(obs)
        self.y_action.append(action_kind)
        self.y_tile.append(tile)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        x_flat = np.stack([obs_to_flat(obs) for obs in self.obs_list])
        y_action = np.asarray(self.y_action, dtype=np.int64)
        y_tile = np.asarray(self.y_tile, dtype=np.int64)
        fields = {}
        for key in self.obs_list[0].keys():
            fields[key] = np.stack([obs[key] for obs in self.obs_list])
        np.savez_compressed(path, x_flat=x_flat, y_action=y_action, y_tile=y_tile, **fields)


class RecordingV6:
    """包住主仓 v5 教师，记录每一次决策（全量动作）。"""

    def __init__(self, acc: BCDataV6, *, label: str = "v5[rec6]"):
        from majiang.strategy.policy import Mode
        from majiang.strategy.versions import build
        self.inner = build("v5", Mode.QUALIFIER)
        self.acc = acc
        self.name = label
        self.last_reason = ""
        self.last_detail: dict[str, object] = {}

    def configure(self, tournament) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = getattr(self.inner, "last_reason", "")
        self.last_detail = dict(getattr(self.inner, "last_detail", {}) or {})
        if choice is not None:
            kind = getattr(choice, "kind", None)
            if kind in ACTION_KIND_TO_ID:
                obs = situation_to_obs(situation, situation.seat)
                tile = choice.tile if (choice.tile is not None and 0 <= choice.tile < 34) else TILE_PAD
                self.acc.add(obs, ACTION_KIND_TO_ID[kind], tile)
        return choice


def generate_bc_data(*, matches: int, rounds: int = 8, seed: int = 20261002) -> BCDataV6:
    from majiang.sim.batch import run_match
    acc = BCDataV6()
    for index in range(matches):
        deciders = [RecordingV6(acc) for _ in range(4)]
        run_match(
            deciders,
            rounds=rounds,
            base_score=1,
            seed=seed * 100003 + index,
            start_dealer=index % 4,
        )
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 BC 数据集 v6（全量动作，教师=v5）")
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    data = generate_bc_data(matches=args.matches, rounds=args.rounds, seed=args.seed)
    dt = time.perf_counter() - t0
    data.save(args.out)

    # 动作分布统计
    counts = np.bincount(np.asarray(data.y_action), minlength=NUM_ACTION_KINDS)
    names = ["discard", "chi", "peng", "gang", "hu", "pass"]
    dist = "  ".join(f"{n}={c}" for n, c in zip(names, counts))
    print(
        f"[v6] 样本 {len(data):6d}  局 {args.matches * args.rounds:4d}  "
        f"用时 {dt:6.1f}s  -> {args.out}"
    )
    print(f"[v6] 动作分布: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
