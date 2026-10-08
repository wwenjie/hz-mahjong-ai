#!/usr/bin/env python
"""生成 BC 数据集 v7：全量动作 + 可行动作掩码（Suphx 式二分类头训练用）。

与 v6 的区别：
  - 新增 ``avail`` 字段：shape (N, 6)，bool/0-1
      avail[i, k] = 1 表示第 i 个决策点动作 k 是可行的（出现在 actions 列表里）
  - 训练时：对每个二分类头，仅对 avail==1 的样本计算损失；
      其中教师选了该动作的为正样本，其余可行但未选的为负样本。
  - 这样「不可行」不再被误当负样本（v6 六分类 softmax 的隐含缺陷）。

用法::

    PYTHONPATH=src python scripts/gen_bc_data_v7.py --matches 100 --out data/bc_v7_train.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.obs import situation_to_obs, obs_to_flat

ACTION_KIND_TO_ID = {
    "discard": 0,
    "chi": 1,
    "peng": 2,
    "gang": 3,
    "hu": 4,
    "pass": 5,
}
NUM_ACTION_KINDS = 6
TILE_PAD = 34


class BCDataV7:
    def __init__(self):
        self.obs_list: list[dict[str, np.ndarray]] = []
        self.y_action: list[int] = []
        self.y_tile: list[int] = []
        self.avail: list[list[int]] = []

    def __len__(self):
        return len(self.obs_list)

    def add(self, obs, action_kind: int, tile: int, avail_mask: list[int]):
        self.obs_list.append(obs)
        self.y_action.append(action_kind)
        self.y_tile.append(tile)
        self.avail.append(avail_mask)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        x_flat = np.stack([obs_to_flat(obs) for obs in self.obs_list])
        y_action = np.asarray(self.y_action, dtype=np.int64)
        y_tile = np.asarray(self.y_tile, dtype=np.int64)
        avail = np.asarray(self.avail, dtype=np.uint8)
        fields = {}
        for key in self.obs_list[0].keys():
            fields[key] = np.stack([obs[key] for obs in self.obs_list])
        np.savez_compressed(
            path, x_flat=x_flat, y_action=y_action, y_tile=y_tile, avail=avail, **fields
        )


class RecordingV7:
    """包住主仓 v5 教师，记录决策 + 可行动作掩码。"""

    def __init__(self, acc: BCDataV7, *, label: str = "v5[rec7]"):
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
        # 先构造可行性掩码（无论教师选了什么都记录）
        avail_mask = [0] * NUM_ACTION_KINDS
        for a in actions:
            k = ACTION_KIND_TO_ID.get(getattr(a, "kind", None))
            if k is not None:
                avail_mask[k] = 1

        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = getattr(self.inner, "last_reason", "")
        self.last_detail = dict(getattr(self.inner, "last_detail", {}) or {})
        if choice is not None:
            kind = getattr(choice, "kind", None)
            if kind in ACTION_KIND_TO_ID:
                obs = situation_to_obs(situation, situation.seat)
                tile = choice.tile if (choice.tile is not None and 0 <= choice.tile < 34) else TILE_PAD
                self.acc.add(obs, ACTION_KIND_TO_ID[kind], tile, avail_mask)
        return choice


def generate_bc_data(*, matches: int, rounds: int = 8, seed: int = 20261002) -> BCDataV7:
    from majiang.sim.batch import run_match
    acc = BCDataV7()
    for index in range(matches):
        deciders = [RecordingV7(acc) for _ in range(4)]
        run_match(
            deciders,
            rounds=rounds,
            base_score=1,
            seed=seed * 100003 + index,
            start_dealer=index % 4,
        )
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 BC 数据集 v7（全量动作 + 可行性掩码，教师=v5）")
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    data = generate_bc_data(matches=args.matches, rounds=args.rounds, seed=args.seed)
    dt = time.perf_counter() - t0
    data.save(args.out)

    counts = np.bincount(np.asarray(data.y_action), minlength=NUM_ACTION_KINDS)
    names = ["discard", "chi", "peng", "gang", "hu", "pass"]
    dist = "  ".join(f"{n}={c}" for n, c in zip(names, counts))
    avail = np.asarray(data.avail, dtype=np.int64).sum(axis=0)
    avail_dist = "  ".join(f"{n}={c}" for n, c in zip(names, avail))
    print(
        f"[v7] 样本 {len(data):6d}  局 {args.matches * args.rounds:4d}  "
        f"用时 {dt:6.1f}s  -> {args.out}"
    )
    print(f"[v7] 动作分布: {dist}")
    print(f"[v7] 可行分布: {avail_dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
