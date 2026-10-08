#!/usr/bin/env python
"""生成 BC Oracle 数据集（D1+D2 合并）：全量动作 + 可行动作掩码 + Oracle 辅助标签。

与 v7 的区别（基于论文方法）：
  - 新增 ``final_scores`` 字段：shape (N, 4)，int32
      Suphx 全局奖励预测——本局最终四家净分（同一局内所有决策点共享同一标签）
  - 新增 ``other_hands`` 字段：shape (N, 3, 34)，int8
      PerfectDou PTIE——其他三家真实手牌计数（相对座位序：下家/对家/上家）
      训练时作为辅助预测目标（不是输入），推理时砍掉辅助头
  - 新增 ``wall_counts`` 字段：shape (N, 34)，int8
      牌墙剩余牌计数分布（完美信息，辅助目标）
  - 新增 ``match_id`` 字段：shape (N,)，int32
      对局唯一标识（用于按局分组/课程学习）

合规红线：other_hands / wall_counts 只作训练辅助目标，绝不进推理管线。

用法::

    PYTHONPATH=src python scripts/gen_bc_data_oracle.py --matches 100 --out data/bc_oracle_train.npz
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
TILE_KINDS = 34


class BCDataOracle:
    def __init__(self):
        self.obs_list: list[dict[str, np.ndarray]] = []
        self.y_action: list[int] = []
        self.y_tile: list[int] = []
        self.avail: list[list[int]] = []
        self.final_scores: list[list[int]] = []      # (4,) 本局最终净分
        self.other_hands: list[np.ndarray] = []       # (3, 34) 其他三家手牌
        self.wall_counts: list[np.ndarray] = []       # (34,) 牌墙剩余分布
        self.match_id: list[int] = []                 # 对局唯一标识

    def __len__(self):
        return len(self.obs_list)

    def add(self, obs, action_kind: int, tile: int, avail_mask: list[int],
            final_scores, other_hands, wall_counts, match_id: int):
        self.obs_list.append(obs)
        self.y_action.append(action_kind)
        self.y_tile.append(tile)
        self.avail.append(avail_mask)
        self.final_scores.append(final_scores)
        self.other_hands.append(other_hands)
        self.wall_counts.append(wall_counts)
        self.match_id.append(match_id)

    def set_final_scores(self, start: int, final_scores) -> None:
        """回填 [start, len) 范围样本的 final_scores（match 结束后调用）。"""
        for i in range(start, len(self.final_scores)):
            self.final_scores[i] = list(final_scores)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        x_flat = np.stack([obs_to_flat(obs) for obs in self.obs_list])
        y_action = np.asarray(self.y_action, dtype=np.int64)
        y_tile = np.asarray(self.y_tile, dtype=np.int64)
        avail = np.asarray(self.avail, dtype=np.uint8)
        final_scores = np.asarray(self.final_scores, dtype=np.int32)
        other_hands = np.stack(self.other_hands).astype(np.int8)
        wall_counts = np.stack(self.wall_counts).astype(np.int8)
        match_id = np.asarray(self.match_id, dtype=np.int32)
        fields = {}
        for key in self.obs_list[0].keys():
            fields[key] = np.stack([obs[key] for obs in self.obs_list])
        np.savez_compressed(
            path, x_flat=x_flat, y_action=y_action, y_tile=y_tile, avail=avail,
            final_scores=final_scores, other_hands=other_hands,
            wall_counts=wall_counts, match_id=match_id, **fields
        )


class RecordingOracle:
    """包住主仓 v5 教师，记录决策 + 可行动作掩码 + Oracle 辅助标签。"""

    def __init__(self, acc: BCDataOracle, *, label: str = "v5[oracle]", seat: int = -1):
        from majiang.strategy.policy import Mode
        from majiang.strategy.versions import build
        self.inner = build("v5", Mode.QUALIFIER)
        self.acc = acc
        self.name = label
        self.last_reason = ""
        self.last_detail: dict[str, object] = {}
        self._state = None       # RoundState（observe_state 注入）
        self._seat = seat        # 本决策器座位（调试用）
        self._match_id = 0       # run_match 之前由 generate_oracle_data 设置

    def configure(self, tournament) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(tournament)

    def observe_state(self, state) -> None:
        """仿真器钩子：接收 RoundState（完美信息来源）。"""
        self._state = state

    def set_match_id(self, mid: int) -> None:
        """run_match 之前设置（样本在对局中实时记录，需提前绑定 match_id）。"""
        self._match_id = mid

    def _extract_oracle(self, situation):
        """从 RoundState 提取 oracle 标签（以 situation.seat 为基准）。"""
        if self._state is None:
            # 无注入（兼容旧路径）：返回全零占位
            return (np.zeros((3, TILE_KINDS), dtype=np.int8),
                    np.zeros(TILE_KINDS, dtype=np.int8))
        seat = situation.seat
        # 其他三家手牌（相对座位序：下家、对家、上家）
        other_hands = np.zeros((3, TILE_KINDS), dtype=np.int8)
        for i in range(1, 4):
            abs_seat = (seat + i) % 4
            other_hands[i - 1] = np.asarray(self._state.seats[abs_seat].hand, dtype=np.int8)
        # 牌墙剩余分布
        wall_counts = np.zeros(TILE_KINDS, dtype=np.int8)
        for tile in self._state.wall:
            wall_counts[tile] += 1
        return other_hands, wall_counts

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
                other_hands, wall_counts = self._extract_oracle(situation)
                self.acc.add(
                    obs, ACTION_KIND_TO_ID[kind], tile, avail_mask,
                    [0, 0, 0, 0],  # final_scores 占位，match 结束后按索引回填
                    other_hands, wall_counts, self._match_id,
                )
        return choice


def generate_oracle_data(*, matches: int, rounds: int = 8, seed: int = 20261006) -> BCDataOracle:
    from majiang.sim.batch import run_match
    acc = BCDataOracle()
    for index in range(matches):
        deciders = [RecordingOracle(acc, seat=s) for s in range(4)]
        for d in deciders:
            d.set_match_id(index)  # 必须在 run_match 之前（样本实时记录）
        start = len(acc)
        result = run_match(
            deciders,
            rounds=rounds,
            base_score=1,
            seed=seed * 100003 + index,
            start_dealer=index % 4,
        )
        # match 结束后回填：本场最终累计净分 → 该 match 所有样本共享
        final = [s.total_score for s in result.seats]
        acc.set_final_scores(start, final)
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 BC Oracle 数据集（Suphx 全局奖励 + PerfectDou PTIE）")
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    data = generate_oracle_data(matches=args.matches, rounds=args.rounds, seed=args.seed)
    dt = time.perf_counter() - t0
    data.save(args.out)

    counts = np.bincount(np.asarray(data.y_action), minlength=NUM_ACTION_KINDS)
    names = ["discard", "chi", "peng", "gang", "hu", "pass"]
    dist = "  ".join(f"{n}={c}" for n, c in zip(names, counts))
    print(
        f"[oracle] 样本 {len(data):6d}  局 {args.matches * args.rounds:4d}  "
        f"用时 {dt:6.1f}s  -> {args.out}"
    )
    print(f"[oracle] 动作分布: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
