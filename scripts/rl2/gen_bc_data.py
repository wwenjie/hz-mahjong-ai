#!/usr/bin/env python
"""生成 BC 数据集（Transformer 版）。

用法::

    PYTHONPATH=src uv run python scripts/gen_bc_data.py --train-matches 8 --valid-matches 3

与 majiang_rl 的 BC 不同：
- 观测 = Mahjax 风格 dict observation（token 化，无人工特征）
- 动作 = 出牌 tile id（0-33）
- 教师 = 主仓 heuristic（默认档）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 把 majiang_rl2/src 加入 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.obs import situation_to_obs, FLAT_FEATURE_DIM


class BCData:
    """BC 数据集容器。"""

    def __init__(self):
        self.obs_list: list[dict[str, np.ndarray]] = []
        self.y_list: list[int] = []
        self.flat_list: list[np.ndarray] = []

    def __len__(self):
        return len(self.obs_list)

    def add(self, obs: dict[str, np.ndarray], action_tile: int):
        self.obs_list.append(obs)
        self.y_list.append(action_tile)

    def build_flat(self) -> tuple[np.ndarray, np.ndarray]:
        """构建展平特征版本（用于简单基线）。"""
        from nnrl2.obs import obs_to_flat
        x = np.stack([obs_to_flat(obs) for obs in self.obs_list])
        y = np.asarray(self.y_list, dtype=np.int64)
        return x, y

    def save(self, path: str | Path):
        """保存为 npz。"""
        from nnrl2.obs import obs_to_flat
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 flat 版本（兼容）
        x_flat = np.stack([obs_to_flat(obs) for obs in self.obs_list])
        y = np.asarray(self.y_list, dtype=np.int64)

        # 保存 dict 版本（各字段分开存）
        fields = {}
        for key in self.obs_list[0].keys():
            fields[key] = np.stack([obs[key] for obs in self.obs_list])

        np.savez_compressed(
            path,
            x_flat=x_flat,
            y=y,
            **fields,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BCData":
        """加载 npz。"""
        data = np.load(path)
        bc = cls()
        # 重建 obs_list
        keys = [k for k in data.keys() if k not in ("x_flat", "y")]
        n = len(data["y"])
        for i in range(n):
            obs = {k: data[k][i] for k in keys}
            bc.obs_list.append(obs)
            bc.y_list.append(int(data["y"][i]))
        return bc


class RecordingHeuristic:
    """包住主仓启发式，边对局边记录出牌决策。"""

    def __init__(self, acc: BCData, *, label: str = "heuristic[rec]"):
        # 延迟导入主仓（避免启动时硬依赖）
        from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

        self.inner = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
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

        # 只记录出牌决策
        if (
            choice is not None
            and getattr(choice, "kind", None) == "discard"
            and choice.tile is not None
            and situation.drawn_tile is not None
        ):
            obs = situation_to_obs(situation, situation.seat)
            self.acc.add(obs, choice.tile)

        return choice


def generate_bc_data(*, matches: int, rounds: int = 8, seed: int = 20260928) -> BCData:
    """跑启发式自对弈，返回 BC 数据集。"""
    from majiang.sim.batch import run_match

    acc = BCData()
    for index in range(matches):
        deciders = [RecordingHeuristic(acc) for _ in range(4)]
        run_match(
            deciders,
            rounds=rounds,
            base_score=1,
            seed=seed * 100003 + index,
            start_dealer=index % 4,
        )
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 BC 数据集（Transformer 版）")
    ap.add_argument("--train-matches", type=int, default=8)
    ap.add_argument("--valid-matches", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=20260928)
    ap.add_argument("--valid-seed", type=int, default=771014)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, matches, seed in (
        ("train", args.train_matches, args.train_seed),
        ("valid", args.valid_matches, args.valid_seed),
    ):
        t0 = time.perf_counter()
        data = generate_bc_data(matches=matches, rounds=args.rounds, seed=seed)
        dt = time.perf_counter() - t0
        target = out / f"bc_{name}.npz"
        data.save(target)
        per_round = matches * args.rounds
        print(
            f"[{name}] 样本 {len(data):6d}  局 {per_round:4d}  用时 {dt:6.1f}s "
            f"（{dt / max(per_round,1) * 1000:.0f} ms/局）  -> {target}"
        )
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
