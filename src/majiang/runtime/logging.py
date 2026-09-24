"""结构化日志（tasks.md 1.5 / 4.10）。

每行一条 JSON，便于 ``jq`` 过滤复盘；同时可选输出人读格式到控制台。设计要点：

- **凭据绝不落盘**：令牌是 32 位以上的十六进制串，写出前统一做脱敏；同时约定调用方
  不把令牌放进字段。
- **每条记录自带时刻**：``ts`` 为本地时间 ISO，``mono_ms`` 为单调时钟毫秒，用于算
  决策耗时（不受系统时间调整影响）。
- 按房间分文件：``<目录>/<tournament_id>.jsonl``，避免多房间混在一起。

日志事件命名约定：``runtime.*`` / ``tournament.*`` / ``game.*`` / ``decision.*`` /
``action.*`` / ``error.*``。
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

DEFAULT_LOG_DIR = "logs"
TOKEN_PATTERN = re.compile(r"\b[0-9a-fA-F]{32,}\b")
REDACTED = "<redacted>"

_SENSITIVE_KEYS = frozenset({"token", "authorization", "auth", "secret"})


def redact(value: Any) -> Any:
    """递归脱敏：长十六进制串与敏感键。"""
    if isinstance(value, str):
        return TOKEN_PATTERN.sub(REDACTED, value)
    if isinstance(value, Mapping):
        return {
            key: (REDACTED if str(key).lower() in _SENSITIVE_KEYS else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


@runtime_checkable
class Logger(Protocol):
    def log(self, event: str, **fields: Any) -> None: ...

    def child(self, **context: Any) -> Logger: ...


class NullLogger:
    """测试与禁用日志时使用。"""

    def log(self, event: str, **fields: Any) -> None:
        return None

    def child(self, **context: Any) -> NullLogger:
        return self


@dataclass
class JsonlLogger:
    """JSONL 日志写入器，线程安全。"""

    path: Path
    console: bool = True
    stream: TextIO = field(default=sys.stderr)
    context: Mapping[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _started: float = field(default_factory=time.monotonic, repr=False)
    _handle: TextIO | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def child(self, **context: Any) -> JsonlLogger:
        """派生一个共享文件句柄与锁的子记录器（不重复打开文件）。"""
        clone = object.__new__(JsonlLogger)
        clone.path = self.path
        clone.console = self.console
        clone.stream = self.stream
        clone.context = {**self.context, **context}
        clone._lock = self._lock
        clone._started = self._started
        clone._handle = self._handle
        return clone

    def log(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "mono_ms": int((time.monotonic() - self._started) * 1000),
            "event": event,
            **self.context,
            **fields,
        }
        record = redact(record)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if self._handle is not None:
                self._handle.write(line + "\n")
                self._handle.flush()
        if self.console:
            self._echo(record)

    def _echo(self, record: Mapping[str, Any]) -> None:
        head = f"[{record['mono_ms']:>8}ms] {record.get('event', '')}"
        details = " ".join(
            f"{key}={value}"
            for key, value in record.items()
            if key not in {"ts", "mono_ms", "event", "game_id", "tournament_id", "round_no"}
        )
        tag = "/".join(
            str(record[key])
            for key in ("tournament_id", "game_id", "round_no")
            if record.get(key) not in (None, "")
        )
        line = f"{head} {tag} {details}".rstrip()
        with self._lock:
            print(line, file=self.stream, flush=True)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def open_logger(
    tournament_id: str,
    directory: str | os.PathLike[str] = DEFAULT_LOG_DIR,
    *,
    console: bool = True,
) -> JsonlLogger:
    """按房间打开日志文件。房间 id 未知时用 ``unbound`` 占位。"""
    name = tournament_id or "unbound"
    return JsonlLogger(path=Path(directory) / f"{name}.jsonl", console=console)


def iter_records(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """读取 JSONL 日志（复盘工具用）。"""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


__all__ = [
    "DEFAULT_LOG_DIR",
    "JsonlLogger",
    "Logger",
    "NullLogger",
    "iter_records",
    "open_logger",
    "redact",
]
