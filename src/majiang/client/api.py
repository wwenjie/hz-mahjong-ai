"""平台玩家 API 的方法封装。

只负责「把一次调用映射为一个模型对象或异常」，不含任何调度、重试策略或状态机逻辑，
那些属于运行时层。

令牌约定：参赛令牌绑定具体锦标赛，可直接调用 ``/me/*`` 系列端点；全局令牌的
``/api/me`` 会返回空 ``tournament_id``，此时调用 ``/me/rules`` 或 ``/me/ready``
会得到 ``400 TOKEN_NOT_SCOPED``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import ApiError
from .models import MyInfo, TournamentConfig, TournamentState
from .transport import Transport

ME_PATH = "/api/me"
MATCH_PATH = "/api/match"
RULES_PATH = "/api/tournaments/me/rules"
READY_ME_PATH = "/api/tournaments/me/ready"
GUIDE_VERSION_PATH = "/portal/api/guide/version"

# 代码所依据的接入指南版本。平台发布新版本时，若含破坏性变更（端点或快照字段改动），
# 启动自检会告警。跟进指南改动后同步更新此常量。
#
# v35（更新于 2026-09-23）唯一的破坏性变更已跟进：新增具名 ``404 TOURNAMENT_GONE``
# ——「房暂时不可达」应与 ``404 TOURNAMENT_NOT_FOUND``（「房不存在」）区分并**重试**，
# 已加入 ``errors.RETRYABLE_CODES``。
KNOWN_GUIDE_VERSION = 35


@dataclass(frozen=True, slots=True)
class GameEnvelope:
    """``GET /api/games/{id}/state`` 的响应。

    ``pending`` 表示长轮询在超时窗口内没有新事件；``snapshot`` 非空时为权威全量局面
    （``seq=0`` 起步或跨局落后时由服务端下发）；``events`` 为增量事件。
    """

    seq: int
    pending: bool = False
    finished: bool = False
    snapshot: Mapping[str, Any] | None = None
    events: tuple[Mapping[str, Any], ...] = ()
    gap: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> GameEnvelope:
        snapshot = raw.get("snapshot")
        gap = bool(raw.get("gap"))
        if isinstance(snapshot, Mapping):
            gap = gap or bool(snapshot.get("gap"))
        return cls(
            seq=int(raw.get("seq", 0) or 0),
            pending=bool(raw.get("pending")),
            finished=bool(raw.get("finished")),
            snapshot=snapshot if isinstance(snapshot, Mapping) else None,
            events=tuple(e for e in raw.get("events") or () if isinstance(e, Mapping)),
            gap=gap,
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class GuideVersion:
    version: int
    updated_at: str = ""
    changes: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> GuideVersion:
        return cls(
            version=int(raw.get("version", 0) or 0),
            updated_at=str(raw.get("updated_at", "")),
            changes=tuple(c for c in raw.get("changes") or () if isinstance(c, Mapping)),
        )

    def breaking_since(self, known_version: int) -> tuple[str, ...]:
        """列出比 ``known_version`` 更新的破坏性变更摘要。"""
        summaries: list[str] = []
        for change in self.changes:
            try:
                version = int(change.get("version", 0) or 0)
            except (TypeError, ValueError):
                continue
            if version > known_version and change.get("type") == "breaking":
                summaries.append(str(change.get("summary", "")))
        return tuple(summaries)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """``POST /api/match`` 的返回：``{room_id, config, round_no}``。

    ``round_no == 0`` 且 ``config["Kind"] == "auto"`` 表示自动房已建好但**尚未开赛**
    （等满 4 人）。等待期重复调用幂等返回原房，绝不双房双席。
    """

    room_id: str
    round_no: int = 0
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> MatchResult:
        return cls(
            room_id=str(raw.get("room_id", "")),
            round_no=int(raw.get("round_no", 0) or 0),
            config=raw.get("config") or {},
        )


class PlatformApi:
    """玩家 API 客户端。"""

    def __init__(self, transport: Transport, token: str | None = None) -> None:
        self._transport = transport
        self._token = token

    @property
    def token(self) -> str | None:
        return self._token

    def with_token(self, token: str) -> PlatformApi:
        return PlatformApi(self._transport, token)

    def _call(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        token: str | None = None,
        authenticated: bool = True,
    ) -> Mapping[str, Any]:
        return self._transport.request(
            method,
            path,
            token=(self._token if token is None else token) if authenticated else None,
            body=body,
        )

    def me(self, token: str | None = None) -> MyInfo:
        return MyInfo.parse(self._call("GET", ME_PATH, token=token))

    def rules(self, token: str | None = None) -> TournamentConfig:
        """当前令牌绑定锦标赛的规则配置（需参赛令牌）。"""
        return TournamentConfig.parse(self._call("GET", RULES_PATH, token=token).get("config") or {})

    def ready_me(self, token: str | None = None) -> Mapping[str, Any]:
        """到位 / 阶段出席确认（参赛令牌直达，幂等）。"""
        return self._call("POST", READY_ME_PATH, token=token)

    def register(self, tournament_id: str, token: str | None = None) -> Mapping[str, Any]:
        return self._call("POST", f"/api/tournaments/{tournament_id}/register", token=token)

    def ready(self, tournament_id: str, token: str | None = None) -> Mapping[str, Any]:
        return self._call("POST", f"/api/tournaments/{tournament_id}/ready", token=token)

    def tournament(self, tournament_id: str, token: str | None = None) -> TournamentState:
        return TournamentState.parse(
            self._call("GET", f"/api/tournaments/{tournament_id}", token=token)
        )

    def game_state(
        self,
        game_id: str,
        seq: int = 0,
        *,
        token: str | None = None,
        timeout: float | None = None,
    ) -> GameEnvelope:
        """长轮询状态与事件流。``seq=0`` 取全量快照，``seq=N`` 取 N 之后的新事件。"""
        raw = self._transport.request(
            "GET",
            f"/api/games/{game_id}/state?seq={seq}",
            token=self._token if token is None else token,
            timeout=timeout,
        )
        return GameEnvelope.parse(raw)

    def submit_action(
        self,
        game_id: str,
        payload: Mapping[str, Any],
        token: str | None = None,
    ) -> Mapping[str, Any]:
        return self._call("POST", f"/api/games/{game_id}/action", body=payload, token=token)

    def match(self, body: Mapping[str, Any] | None = None) -> MatchResult:
        """自动匹配建房与入席。

        **需全局令牌**——报名令牌会 400 ``TOKEN_NOT_SCOPED``，存量匿名全局令牌会
        403 ``PORTAL_BINDING_REQUIRED``。``body`` 可声明可承受上限 ``{"M":10,"Rounds":8}``，
        但**不要声明低于服务默认（M=10/Rounds=8）的上限**（会 404 ``NO_ROOM_AVAILABLE``，
        永久条件），省略最安全。限速 10 次/分/用户，等待期重复调用幂等返回原房。
        """
        return MatchResult.parse(self._call("POST", MATCH_PATH, body=body))

    def guide_version(self) -> GuideVersion:
        """免认证的接入指南版本，用于启动自检。"""
        return GuideVersion.parse(
            self._call("GET", GUIDE_VERSION_PATH, authenticated=False)
        )


__all__ = [
    "GUIDE_VERSION_PATH",
    "KNOWN_GUIDE_VERSION",
    "MATCH_PATH",
    "ApiError",
    "GameEnvelope",
    "GuideVersion",
    "MatchResult",
    "PlatformApi",
]


def merge_active_games(active: Sequence[str], tournament: Sequence[str]) -> tuple[str, ...]:
    """把「进行中场次」与「本赛事场次」求交集（全局令牌时用于过滤）。

    参赛令牌下服务端已按锦标赛限定，直接返回 ``active`` 即可。
    """
    tournament_set = set(tournament)
    return tuple(game_id for game_id in active if game_id in tournament_set)
