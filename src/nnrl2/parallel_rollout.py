"""多进程并行 rollout 收集（可插拔模块）。

架构：
- 主进程持有 GPU 模型 + PPO 更新
- worker 进程（spawn）只用 CPU 跑模拟与推理，单线程 torch 防超订阅
- 每次收集前主进程把 state_dict 广播给 worker（同步策略）
- collect_fn 可插拔：默认 v7 收集器；Oracle Guiding 等可传自己的
  ``"module:function"`` 路径，签名须为
  ``fn(model, device, seed=, rounds=, opponent_factories=) -> (traj, result)``

用法::

    from nnrl2.parallel_rollout import ParallelRolloutPool
    from nnrl2.model_v7 import MahjongTransformerV7

    pool = ParallelRolloutPool(
        MahjongTransformerV7, {"d_model": 128, ...},
        num_workers=12,
        opponent_kind="v5",           # None = 自对弈
        state_dict=model.state_dict(),
    )
    for traj, result in pool.collect(seed_start=0, num_episodes=24, rounds=8):
        ...
    pool.sync_model(model.state_dict())   # 下一轮收集前同步新权重
    pool.close()
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

# worker 候选路径（本机 + AutoDL 布局），存在才插入
_CANDIDATE_PATHS = [
    str(Path(__file__).resolve().parent.parent.parent / "src"),  # nnrl2 自身（冗余保险）
    "/home/wuwenjie01/majiang_ai/src",
    "/root/autodl-tmp/ab/majiang_ai/src",
    "/root/autodl-tmp/majiang_ai/src",
    "/root/majiang_ai/src",
]

DEFAULT_COLLECT_FN = "nnrl2.policy_v7:collect_episode_v7"


class ParallelRolloutPool:
    """多进程 rollout 收集池。

    Parameters
    ----------
    model_cls : 模型类（须可 pickle，即模块级类，如 MahjongTransformerV7）
    model_kwargs : 构造参数
    num_workers : worker 进程数
    opponent_kind : None=自对弈；"v5"=3 个 v5 教师（Mode.QUALIFIER）
    state_dict : 初始权重（CPU 或 GPU tensor 均可，内部转 CPU 广播）
    collect_fn : 收集函数 "module:function" 路径
    extra_paths : 额外要插入 worker sys.path 的路径
    """

    def __init__(
        self,
        model_cls: type,
        model_kwargs: dict,
        num_workers: int = 12,
        opponent_kind: Optional[str] = None,
        state_dict: Optional[dict] = None,
        collect_fn: str = DEFAULT_COLLECT_FN,
        extra_paths: Optional[list[str]] = None,
    ):
        self.num_workers = num_workers
        paths = [p for p in _CANDIDATE_PATHS if os.path.isdir(p)]
        if extra_paths:
            paths.extend(extra_paths)

        ctx = mp.get_context("spawn")  # spawn：避免 fork 继承 CUDA 上下文
        self.pool = ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(model_cls, model_kwargs, opponent_kind, collect_fn, paths),
        )
        if state_dict is not None:
            self.sync_model(state_dict)

    def sync_model(self, state_dict: dict):
        """把主进程最新权重广播到所有 worker。"""
        cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        self.pool.map(_worker_sync_model, [cpu_state] * self.num_workers)

    def collect(self, seed_start: int, num_episodes: int, rounds: int = 8) -> list:
        """并行收集 num_episodes 个 episode，返回 [(traj, result), ...]。

        第 i 个 episode 的种子为 ``seed_start + i``（可复现）。
        """
        tasks = [(seed_start + i, rounds) for i in range(num_episodes)]
        return self.pool.starmap(_worker_collect_episode, tasks)

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# --- worker 进程函数（模块级，pickle 需要） ---

_worker_model: Any = None
_worker_opponent_kind: Optional[str] = None
_worker_collect_fn: Any = None


def _resolve(path: str):
    import importlib

    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)


def _worker_init(model_cls, model_kwargs, opponent_kind, collect_fn, paths):
    """worker 进程初始化：设路径、限线程、建 CPU 模型。"""
    global _worker_model, _worker_opponent_kind, _worker_collect_fn

    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    # 关键：worker 单线程，否则 N worker × N 线程超订阅拖垮吞吐
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 不调用 eval()：与串行训练路径行为一致（dropout 状态保持与 v3a/v3b 血统相同）
    _worker_model = model_cls(**model_kwargs)
    _worker_opponent_kind = opponent_kind
    _worker_collect_fn = _resolve(collect_fn)


def _worker_sync_model(state_dict):
    global _worker_model
    _worker_model.load_state_dict(state_dict)


def _worker_collect_episode(seed, rounds):
    global _worker_model, _worker_opponent_kind, _worker_collect_fn

    opponent_factories = None
    if _worker_opponent_kind == "v5":
        from majiang.strategy.versions import build
        from majiang.strategy.policy import Mode

        opponent_factories = [lambda: build("v5", Mode.QUALIFIER) for _ in range(3)]

    return _worker_collect_fn(
        _worker_model,
        torch.device("cpu"),
        seed=seed,
        rounds=rounds,
        opponent_factories=opponent_factories,
    )


__all__ = ["ParallelRolloutPool", "DEFAULT_COLLECT_FN"]
