"""SSE 通知流测试（tasks.md 3.14）。

**为什么只能靠假流验证开心路径**：通知流属于玩家 API，而测试房对玩家 API 整体 404
（实测 ``/api/games/<已结束场次>/notify`` 返回 ``404 GAME_NOT_FOUND``）。因此真机只能
验证「4xx 被正确判为永久条件、不空转重连」这一条，帧解析、保活与 close 语义
必须用假流覆盖。这里两者都测。
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from majiang.client.errors import ApiError, TransportError
from majiang.client.notify import NotifyFrame, NotifyStream, parse_frame


class FakeResponse:
    """模拟 ``urlopen`` 的返回：可迭代行 + 支持 ``with``。"""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._lines)


def frame_bytes(**payload: object) -> bytes:
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def flushed_stream(lines: list[bytes], **kwargs: object) -> NotifyStream:
    """返回一个只产出一轮 ``lines`` 的流，重连不等待（避免测试变慢）。"""
    stream = NotifyStream("https://example.invalid", "g1", "tok", reconnect_backoff=0.0, **kwargs)
    stream._open = lambda: FakeResponse(list(lines))  # type: ignore[method-assign]
    return stream


# --- parse_frame -------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    ["", "   ", "\n", ": keepalive", "event: ping", "data:", "data:   ", "data: not-json",
     "data: 123", "data: [1,2]", 'data: {"seq": "abc"}'],
)
def test_parse_frame_ignores_non_frames(line: str) -> None:
    assert parse_frame(line) is None


def test_parse_frame_reads_seq_and_closed() -> None:
    assert parse_frame('data: {"seq": 42}') == NotifyFrame(seq=42, closed=False)
    assert parse_frame('data: {"seq": 7, "closed": true}') == NotifyFrame(seq=7, closed=True)
    # closed 缺省为假
    assert parse_frame('data: {"seq": 7, "closed": false}') == NotifyFrame(seq=7, closed=False)


# --- frames ------------------------------------------------------------------------


def test_frames_yields_initial_then_closed_and_stops() -> None:
    stream = flushed_stream([frame_bytes(seq=100), frame_bytes(seq=104, closed=True)])
    assert list(stream.frames()) == [NotifyFrame(100, False), NotifyFrame(104, True)]
    # closed 之后不再重连
    assert stream.reconnects == 0


def test_frames_counts_keepalive_and_invokes_hook() -> None:
    seen: list[int] = []
    stream = flushed_stream(
        [b": keepalive\n", frame_bytes(seq=1), b": keepalive\n", frame_bytes(seq=2, closed=True)],
        on_keepalive=lambda: seen.append(1),
    )
    assert list(stream.frames()) == [NotifyFrame(1), NotifyFrame(2, True)]
    assert stream.keepalives == 2 and len(seen) == 2


def test_frames_skips_malformed_lines() -> None:
    stream = flushed_stream(
        [b"\n", b"data: {broken\n", b"event: x\n", frame_bytes(seq=9, closed=True)]
    )
    assert list(stream.frames()) == [NotifyFrame(9, True)]


def test_frames_reconnects_after_transient_failure_then_succeeds() -> None:
    stream = NotifyStream("https://example.invalid", "g1", "tok", reconnect_backoff=0.0)
    attempts: list[int] = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError("boom")
        return FakeResponse([frame_bytes(seq=5, closed=True)])

    stream._open = flaky  # type: ignore[method-assign]
    assert list(stream.frames()) == [NotifyFrame(5, True)]
    assert stream.reconnects == 1 and len(attempts) == 2


def test_frames_gives_up_after_max_consecutive_failures() -> None:
    stream = NotifyStream(
        "https://example.invalid", "g1", "tok", reconnect_backoff=0.0, max_consecutive_failures=3
    )
    stream._open = lambda: (_ for _ in ()).throw(urllib.error.URLError("boom"))  # type: ignore[method-assign]
    with pytest.raises(TransportError, match="放弃"):
        list(stream.frames())


def test_frames_raises_immediately_on_permanent_http_error() -> None:
    """4xx 是令牌或场次问题，重连只是空转——必须立刻抛，且不退避重试。"""
    stream = NotifyStream(
        "https://example.invalid", "g1", "tok", reconnect_backoff=0.0, max_consecutive_failures=5
    )
    body = io.BytesIO(json.dumps({"code": "GAME_NOT_FOUND", "message": "no such game"}).encode())

    def denied():
        raise urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {}, body)

    stream._open = denied  # type: ignore[method-assign]
    with pytest.raises(ApiError) as caught:
        list(stream.frames())
    assert caught.value.code == "GAME_NOT_FOUND"
    # 一次都不重连
    assert stream.reconnects == 0


def test_frames_retries_on_rate_limited() -> None:
    """429 属于可重试：退避后重连，而不是当作永久错误。"""
    stream = NotifyStream(
        "https://example.invalid", "g1", "tok", reconnect_backoff=0.0, max_consecutive_failures=5
    )
    attempts: list[int] = []

    def limited():
        attempts.append(1)
        if len(attempts) == 1:
            body = io.BytesIO(json.dumps({"code": "RATE_LIMITED"}).encode())
            raise urllib.error.HTTPError("https://example.invalid", 429, "Too Many", {}, body)
        return FakeResponse([frame_bytes(seq=3, closed=True)])

    stream._open = limited  # type: ignore[method-assign]
    assert list(stream.frames()) == [NotifyFrame(3, True)]
    assert stream.reconnects == 1


def test_frames_honours_stop() -> None:
    stream = flushed_stream([frame_bytes(seq=1), frame_bytes(seq=2), frame_bytes(seq=3)])
    seen = []

    def stop() -> bool:
        return len(seen) >= 1

    for frame in stream.frames(stop=stop):
        seen.append(frame)
    assert seen == [NotifyFrame(1)]
    assert stream.reconnects == 0


def test_url_is_built_from_game_id() -> None:
    stream = NotifyStream("https://host:1/", "abc", "tok")
    assert stream.url == "https://host:1/api/games/abc/notify"
