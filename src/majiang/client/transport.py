"""HTTP 传输层。

用可替换的传输抽象隔离网络细节：默认实现基于标准库 ``urllib``（与官方最小 Bot
同源），测试使用内存假实现，因此协议逻辑可在无网络环境下验证。

约定：所有请求携带 ``Authorization: Bearer <令牌>``；部署实例使用自签证书，因此
默认不校验 TLS。``429`` 与 ``5xx`` 按退避重试，其余错误按 ``code`` 分类抛出
:class:`ApiError`。
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .errors import ApiError, TransportError, make_api_error

JSON = dict[str, Any]

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT = 35.0
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF = 0.5


@runtime_checkable
class Transport(Protocol):
    """传输抽象：把一次请求映射为解析后的 JSON 或异常。"""

    def request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> JSON: ...


def _build_ssl_context(verify: bool) -> ssl.SSLContext:
    if not verify:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    return ssl.create_default_context()


@dataclass
class HttpTransport:
    """基于标准库的 JSON over HTTP 传输。"""

    base_url: str
    verify_tls: bool = False
    timeout: float = DEFAULT_TIMEOUT
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff: float = DEFAULT_BACKOFF
    _context: ssl.SSLContext = field(init=False, repr=False)
    _sleep: Any = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._context = _build_ssl_context(self.verify_tls)

    def request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> JSON:
        url = self.base_url + path if path.startswith("/") else f"{self.base_url}/{path}"
        payload = None if body is None else json.dumps(body).encode()
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(url, data=payload, method=method)
            request.add_header("Content-Type", "application/json")
            if token:
                request.add_header("Authorization", "Bearer " + token)
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout if timeout is None else timeout,
                    context=self._context,
                ) as response:
                    return self._decode(response.read())
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode(errors="replace")
                parsed = self._try_decode(raw)
                if exc.code in RETRY_STATUS and attempt < self.max_attempts:
                    last_error = make_api_error(exc.code, parsed, raw)
                    self._sleep(self.backoff * attempt)
                    continue
                raise make_api_error(exc.code, parsed, raw) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_attempts:
                    last_error = exc
                    self._sleep(self.backoff * attempt)
                    continue
                raise TransportError(f"请求 {method} {path} 失败: {exc}") from exc
        raise TransportError(f"请求 {method} {path} 重试耗尽: {last_error}")

    @staticmethod
    def _decode(raw: bytes) -> JSON:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(f"响应不是合法 JSON: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise TransportError(f"响应不是 JSON 对象: {parsed!r}")
        return parsed

    @staticmethod
    def _try_decode(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


__all__ = ["ApiError", "HttpTransport", "JSON", "Transport", "TransportError"]
