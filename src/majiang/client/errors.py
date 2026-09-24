"""平台 API 错误与可重试性分类。

判型一律用 ``code`` 而非 HTTP 状态码或 message（平台文档明确要求）。状态码与 code
并非一一对应——例如 ``FEATURE_DISABLED`` 在 v29 起同时以 403 出现在三个入口，
而跨轮重开测试房此前是 409。
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """平台返回了错误响应。"""

    def __init__(self, status: int, code: str, message: str, body: str = "") -> None:
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.body = body

    def __str__(self) -> str:
        return f"HTTP {self.status} {self.code}: {self.message}"


class TransportError(Exception):
    """网络层失败（连接中断、超时、TLS 错误等），未取得 HTTP 响应。"""


# 可退避重试：限速与临时忙
RETRYABLE_CODES = frozenset({"RATE_LIMITED", "MATCH_BUSY"})

# 永久条件：重试不会改变结果，应改变策略或放弃
PERMANENT_CODES = frozenset(
    {
        "UNAUTHORIZED",
        "TOKEN_NOT_SCOPED",
        "FORBIDDEN",
        "FEATURE_DISABLED",
        "PORTAL_BINDING_REQUIRED",
        "TOURNAMENT_NOT_FOUND",
        "GAME_NOT_FOUND",
        "TOURNAMENT_CLOSED",
        "TOURNAMENT_STARTED",
        "NOT_REGISTERED",
        "NOT_QUALIFIED",
        "AUTO_MATCH_ONLY",
        "MATCH_LIMIT_REACHED",
        "GAME_NOT_FINISHED",
        "NO_ROOM_AVAILABLE",
        "INVALID_INPUT",
    }
)

# 竞态类：动作已失效，需重建快照后重新决策（既不是退避重试，也不是永久条件）
RACE_CODES = frozenset({"INVALID_ACTION"})


def code_of(payload: Any) -> tuple[str, str]:
    """从错误响应体里取出 ``(code, message)``。"""
    if isinstance(payload, dict):
        return str(payload.get("code", "")), str(payload.get("message", ""))
    return "", ""


def make_api_error(status: int, payload: Any, body: str = "") -> ApiError:
    code, message = code_of(payload)
    if not code:
        code = f"HTTP_{status}"
    return ApiError(status=status, code=code, message=message, body=body)


def is_retryable(error: ApiError) -> bool:
    return error.code in RETRYABLE_CODES


def is_permanent(error: ApiError) -> bool:
    return error.code in PERMANENT_CODES


def is_race(error: ApiError) -> bool:
    return error.code in RACE_CODES
