"""SSE 通知流客户端（tasks.md 3.14）。

平台端点 ``GET /api/games/{id}/notify`` 的语义（v12 起）：

- 连接即收初始帧 ``{"seq": N}``——当前事件水位
- 此后每次状态变更推 ``{"seq": 新水位}``——**帧里只有水位，没有变化内容**
- 每 30s 一行 ``: keepalive`` 维持连接
- 场终/死场推 ``{"seq": N, "closed": true}`` 后关流
- 每用户 32 并发连接（超出 429），**不占用 /state 的 16/s 频率额度**

因此它替代不了 ``/state``，只是把「定时轮询」换成「等推送再拉」：收到帧后仍要
``GET /state?seq=本地`` 拉增量。价值在于省下 /state 的频率额度——对 M=10 场并发
对局的调度是实质改善，因为轮询频率正是并发数的瓶颈。

两个实现要点：

1. **读取超时必须大于 keepalive 间隔**。否则服务端正常的保活会被我们当成断线，
   变成「连上就断」的空转。这里默认 45s（keepalive 30s）。
2. **4xx 是永久条件，不做空转重连**；只有网络层失败与限速（429）才退避重连。
   这与 ``errors.py`` 的可重试分类保持一致。
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .errors import ApiError, TransportError, is_retryable, make_api_error
from .transport import _build_ssl_context

NOTIFY_PATH = "/api/games/{game_id}/notify"
KEEPALIVE_SECONDS = 30.0
# keepalive 每 30s 一行，读取超时必须留出余量，否则会把保活误判成断线
DEFAULT_READ_TIMEOUT = 45.0
DEFAULT_RECONNECT_BACKOFF = 1.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


@dataclass(frozen=True, slots=True)
class NotifyFrame:
    """一帧水位信号。``closed`` 为真表示场终/死场，流即将结束。"""

    seq: int
    closed: bool = False


def parse_frame(line: str) -> NotifyFrame | None:
    """解析一行 SSE。空行、``: keepalive`` 注释与非法 JSON 一律返回 ``None``。"""
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:") :].strip()
    if not payload:
        return None
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        seq = int(raw.get("seq", 0) or 0)
    except (TypeError, ValueError):
        return None
    return NotifyFrame(seq=seq, closed=bool(raw.get("closed")))


class NotifyStream:
    """``GET /api/games/{id}/notify`` 的读取端。"""

    def __init__(
        self,
        base_url: str,
        game_id: str,
        token: str,
        *,
        verify_tls: bool = False,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        reconnect_backoff: float = DEFAULT_RECONNECT_BACKOFF,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        on_keepalive: Callable[[], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.game_id = game_id
        self.token = token
        self.read_timeout = read_timeout
        self.reconnect_backoff = reconnect_backoff
        self.max_consecutive_failures = max_consecutive_failures
        self.on_keepalive = on_keepalive
        self.reconnects = 0
        self.keepalives = 0
        self._context = _build_ssl_context(verify_tls)

    @property
    def url(self) -> str:
        return self.base_url + NOTIFY_PATH.format(game_id=self.game_id)

    def _open(self):
        request = urllib.request.Request(self.url, method="GET")
        request.add_header("Accept", "text/event-stream")
        request.add_header("Authorization", f"Bearer {self.token}")
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=self._context))
        return opener.open(request, timeout=self.read_timeout)

    def _http_error(self, error: urllib.error.HTTPError) -> ApiError:
        try:
            payload = json.loads(error.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            payload = None
        return make_api_error(error.code, payload)

    def frames(self, *, stop: Callable[[], bool] | None = None) -> Iterator[NotifyFrame]:
        """持续产出水位帧，直到 ``closed`` 帧、``stop()`` 为真，或连续失败超限。

        网络中断与限速（429）会退避后重连；其余 4xx 立即抛出 :class:`ApiError`——
        那是令牌或场次的问题，重连只是空转。
        """
        consecutive = 0
        last_error: Exception | None = None
        while stop is None or not stop():
            try:
                with self._open() as response:
                    consecutive = 0
                    for raw in response:
                        if stop is not None and stop():
                            return
                        line = raw.decode("utf-8", "replace")
                        if line.strip().startswith(":"):
                            self.keepalives += 1
                            if self.on_keepalive is not None:
                                self.on_keepalive()
                            continue
                        frame = parse_frame(line)
                        if frame is None:
                            continue
                        yield frame
                        if frame.closed:
                            return
            except urllib.error.HTTPError as error:
                api_error = self._http_error(error)
                if not is_retryable(api_error):
                    raise api_error from error
                consecutive += 1
                last_error = api_error
            except (urllib.error.URLError, OSError) as error:
                consecutive += 1
                last_error = error
            if consecutive >= self.max_consecutive_failures:
                raise TransportError(
                    f"notify 流连续 {consecutive} 次失败后放弃: {last_error}"
                ) from last_error
            self.reconnects += 1
            time.sleep(self.reconnect_backoff)


__all__ = [
    "DEFAULT_READ_TIMEOUT",
    "DEFAULT_RECONNECT_BACKOFF",
    "KEEPALIVE_SECONDS",
    "NOTIFY_PATH",
    "NotifyFrame",
    "NotifyStream",
    "parse_frame",
]
