"""对主仓库的**只读**访问桥（继承 majiang_rl 的模式）。

本仓库（majiang_rl2，Transformer BC + PPO 线）与参赛主仓库 `hz-mahjong-ai` 是两套
所有权。为了在不违反主仓库「同一文件只有一个所有人」铁律的前提下复用其**公开离线
数据**，这里把所有跨仓访问收敛到一个地方，并把「只读」做成代码约束而不是口头约定。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

DEFAULT_REPO = "/home/wuwenjie01/majiang_ai"
ENV_REPO = "MAJIANG_REPO"


class ReadOnlyViolation(RuntimeError):
    """试图向主仓库写入时抛出。"""


def repo_root() -> Path:
    """主仓库根目录（只读）。"""
    return Path(os.environ.get(ENV_REPO, DEFAULT_REPO)).resolve()


def data_dir() -> Path:
    """主仓库的 ``data/`` 目录（只读）。"""
    return repo_root() / "data"


def _resolve_readable(path: str | Path) -> Path:
    root = repo_root()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = data_dir() / candidate
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ReadOnlyViolation(f"路径不在主仓库内：{candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"主仓库中不存在：{candidate}")
    return candidate


def forbid_write(path: str | Path) -> None:
    """任何写入前调用：路径若落在主仓库内，直接拒绝。"""
    root = repo_root()
    candidate = Path(path)
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = Path(os.path.abspath(str(path)))
    if candidate == root or root in candidate.parents:
        raise ReadOnlyViolation(
            f"拒绝写入主仓库（只读边界）：{candidate}\n"
            f"主仓库根本应只读；majiang_rl2 产物请写入本仓库自身。"
        )


def iter_event_files() -> Iterator[Path]:
    """遍历 ``data/auto_sessions/*/events/*.json``（真实对局事件流，只读）。"""
    events_root = data_dir() / "auto_sessions"
    if not events_root.is_dir():
        return
    for room in sorted(p for p in events_root.iterdir() if p.is_dir()):
        for f in sorted(room.glob("events/*.json")):
            yield f


def load_npz(name: str):
    """加载主仓库 ``data/`` 下的 npz 文件（只读）。"""
    import numpy as np
    path = _resolve_readable(name)
    return np.load(path)
