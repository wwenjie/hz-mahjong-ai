"""请求限速与并发闸（tasks.md 3.13）。

平台的 ``/state`` 限速是**每用户 16 次/秒**、并发挂起轮询 ≤32——限速与并发都不随场次
数量放宽，因此必须由所有对局共用一个令牌桶与一个并发闸，而不是每场各自限速。

实现为传输层装饰器，这样「限速」这件事对所有调用路径（心跳、快照、动作提交）自动
生效，运行时无需在每个调用点重复记得限速。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from majiang.client.transport import JSON, Transport


class RateLimitTimeout(RuntimeError):
    """在给定超时内没能取得令牌。"""


DEFAULT_BURST = 2.0


@dataclass
class TokenBucket:
    """令牌桶。``rate_per_sec`` 为稳态速率，``capacity`` 为突发上限。

    突发上限刻意压得比速率低：平台的 16/s 是每用户窗口限速，若允许一次性打出 16 个
    请求，叠加上一个窗口的尾部就会越界（实测撞出过 ``RATE_LIMITED``）。压到 4 后
    稳态吞吐不变，但不会打满窗口。
    """

    rate_per_sec: float
    capacity: float | None = None
    _tokens: float = field(init=False, repr=False)
    _updated: float = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.rate_per_sec <= 0:
            raise ValueError(f"速率必须为正: {self.rate_per_sec}")
        if self.capacity is None:
            self.capacity = min(DEFAULT_BURST, self.rate_per_sec)
        if self.capacity <= 0:
            raise ValueError(f"容量必须为正: {self.capacity}")
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(float(self.capacity), self._tokens + elapsed * self.rate_per_sec)
            self._updated = now

    def acquire(self, amount: float = 1.0, timeout: float | None = None) -> bool:
        """取走 ``amount`` 个令牌，必要时阻塞等待。超时返回 False。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return True
                wait = (amount - self._tokens) / self.rate_per_sec
            if deadline is not None and time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 0.25))

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class ConcurrencyGate:
    """并发闸：限制同时在途的请求数（用于「并发挂起轮询 ≤32」）。"""

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError(f"并发上限必须为正: {limit}")
        self.limit = limit
        self._semaphore = threading.BoundedSemaphore(limit)

    def acquire(self, timeout: float | None = None) -> bool:
        return self._semaphore.acquire(timeout=timeout)

    def release(self) -> None:
        self._semaphore.release()

    @property
    def in_flight(self) -> int:
        return self.limit - self._semaphore._value  # type: ignore[attr-defined]


@dataclass
class RateLimitedTransport:
    """给任意传输加上全局令牌桶与并发闸。

    ``priority_bucket`` 若给定，则匹配 ``priority_marker`` 的请求（即动作提交）从**预留桶**
    取令牌，其余请求走主桶。两个桶的速率之和即为总速率，因此总量不变。

    为什么要预留：动作提交有硬窗口（实测自动房出牌 3 秒、碰/吃 1 秒），而轮询没有。
    共用一个桶时，提交会排在大量在途轮询后面；一旦因此触发平台的 429，传输层的退避重试
    （0.5 秒起、最多 4 次）会直接吃掉整个窗口，最后拿到 409 ``INVALID_ACTION`` ——
    实测拒绝时刻距决策记录中位 2217ms，而代码里两者是相邻两行，这就是根因。
    """

    inner: Transport
    bucket: TokenBucket
    gate: ConcurrencyGate | None = None
    acquire_timeout: float | None = 30.0
    priority_bucket: TokenBucket | None = None
    priority_marker: str = "/action"

    def _is_priority(self, method: str, path: str) -> bool:
        return (
            self.priority_bucket is not None
            and method == "POST"
            and path.endswith(self.priority_marker)
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> JSON:
        if self._is_priority(method, path):
            # 只从预留桶取，绝不回落到主桶——否则又会被轮询挤住，等于白留
            assert self.priority_bucket is not None
            if not self.priority_bucket.acquire(timeout=self.acquire_timeout):
                raise RateLimitTimeout(f"取得动作预留额度超时: {method} {path}")
        elif not self.bucket.acquire(timeout=self.acquire_timeout):
            raise RateLimitTimeout(f"取得限速令牌超时: {method} {path}")
        held = False
        if self.gate is not None:
            if not self.gate.acquire(timeout=self.acquire_timeout):
                raise RateLimitTimeout(f"取得并发额度超时: {method} {path}")
            held = True
        try:
            return self.inner.request(method, path, token=token, body=body, timeout=timeout)
        finally:
            if held:
                self.gate.release()  # type: ignore[union-attr]


__all__ = [
    "ConcurrencyGate",
    "DEFAULT_BURST",
    "RateLimitedTransport",
    "RateLimitTimeout",
    "TokenBucket",
]
