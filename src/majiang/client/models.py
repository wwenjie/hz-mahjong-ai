"""平台响应模型。

字段名以 2026-09-23 实测（指南 v34）为准，比文档多出的键（``Kind`` / ``TimeoutMin`` /
``OnlineConfirm`` / ``my_games_by_batch`` / ``active_count`` 等）同样纳入模型。

解析策略：必需键缺失则报错（避免静默用错配置），未知键一律收进 ``extra``，因此平台
新增字段不会导致解析失败——官方文档承诺接口变更为「契约加法」。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

STATUS_REGISTERING = "registering"
STATUS_RUNNING = "running"
STATUS_STAGE_DONE = "stage_done"
STATUS_STAGE_OPEN = "stage_open"
STATUS_FINISHED = "finished"
STATUS_CLOSED = "closed"
STATUS_VOID = "void"

STATUSES = (
    STATUS_REGISTERING,
    STATUS_RUNNING,
    STATUS_STAGE_DONE,
    STATUS_STAGE_OPEN,
    STATUS_FINISHED,
    STATUS_CLOSED,
    STATUS_VOID,
)

# 终态：可以停止主循环
TERMINAL_STATUSES = frozenset({STATUS_FINISHED, STATUS_CLOSED, STATUS_VOID})

# 接受到位/出席确认的状态
READY_STATUSES = frozenset({STATUS_REGISTERING, STATUS_STAGE_OPEN})

ROLE_FINALIST = "finalist"
ROLE_BACKUP = "backup"

DEFAULT_DISCARD_TIMEOUT_SEC = 3.0
DEFAULT_PENG_TIMEOUT_SEC = 1.0
DEFAULT_CHI_TIMEOUT_SEC = 1.0

_CONFIG_MAP: dict[str, str] = {
    "M": "max_concurrent_games",
    "Rounds": "rounds_per_game",
    "BaseScore": "base_score",
    "YouCaiBiKao": "you_cai_bi_kao",
    "DiscardTimeoutSec": "discard_timeout_sec",
    "PengTimeoutSec": "peng_timeout_sec",
    "ChiTimeoutSec": "chi_timeout_sec",
    "Kind": "kind",
    "Name": "name",
    "Description": "description",
    "StartAt": "start_at",
    "RegisterDeadlineAt": "register_deadline_at",
    "TimeoutMin": "timeout_min",
    "OnlineConfirm": "online_confirm",
}


_REVERSE_CONFIG_MAP: dict[str, str] = {target: wire for wire, target in _CONFIG_MAP.items()}


class ModelError(ValueError):
    """响应结构与预期不符。"""


def _require(source: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in source:
        raise ModelError(f"{where} 缺少必需字段 {key!r}")
    return source[key]


def _require_config(known: Mapping[str, Any], target: str) -> Any:
    """取必需配置项；报错时同时给出平台原始字段名与内部字段名。"""
    if target not in known:
        wire = _REVERSE_CONFIG_MAP.get(target, target)
        raise ModelError(f"config 缺少必需字段 {wire}（内部名 {target}）")
    return known[target]


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_known(raw: Mapping[str, Any], mapping: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """按 ``mapping`` 拆分已知键与未知键。"""
    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        target = mapping.get(key)
        if target is None:
            extra[key] = value
        else:
            known[target] = value
    return known, extra


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    """锦标赛配置。底分、局数、同时场数与超时都以服务端返回为准，禁止写死。"""

    max_concurrent_games: int
    rounds_per_game: int
    base_score: int
    you_cai_bi_kao: bool = False
    discard_timeout_sec: float = DEFAULT_DISCARD_TIMEOUT_SEC
    peng_timeout_sec: float = DEFAULT_PENG_TIMEOUT_SEC
    chi_timeout_sec: float = DEFAULT_CHI_TIMEOUT_SEC
    kind: str = ""
    name: str = ""
    description: str = ""
    start_at: int = 0
    register_deadline_at: int = 0
    timeout_min: int = 0
    online_confirm: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_test_room(self) -> bool:
        return self.kind == "test"

    @property
    def is_auto_room(self) -> bool:
        return self.kind == "auto"

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> TournamentConfig:
        known, extra = split_known(raw, _CONFIG_MAP)
        return cls(
            max_concurrent_games=int(_require_config(known, "max_concurrent_games")),
            rounds_per_game=int(_require_config(known, "rounds_per_game")),
            base_score=int(_require_config(known, "base_score")),
            you_cai_bi_kao=bool(known.get("you_cai_bi_kao", False)),
            discard_timeout_sec=_number(
                known.get("discard_timeout_sec"), DEFAULT_DISCARD_TIMEOUT_SEC
            ),
            peng_timeout_sec=_number(known.get("peng_timeout_sec"), DEFAULT_PENG_TIMEOUT_SEC),
            chi_timeout_sec=_number(known.get("chi_timeout_sec"), DEFAULT_CHI_TIMEOUT_SEC),
            kind=str(known.get("kind", "")),
            name=str(known.get("name", "")),
            description=str(known.get("description", "")),
            start_at=int(known.get("start_at", 0) or 0),
            register_deadline_at=int(known.get("register_deadline_at", 0) or 0),
            timeout_min=int(known.get("timeout_min", 0) or 0),
            online_confirm=bool(known.get("online_confirm", False)),
            extra=extra,
        )


def game_id_of(entry: Any) -> str:
    """``active_games`` 的元素是对象（含 ``game_id`` 键），也兼容裸字符串。"""
    if isinstance(entry, Mapping):
        return str(entry.get("game_id", ""))
    return str(entry)


@dataclass(frozen=True, slots=True)
class MyInfo:
    """``GET /api/me`` 的响应。

    实测确认：``active_games`` 是**对象列表**（每个含 ``game_id``），不是字符串列表；
    原始条目保留在 ``active_game_details`` 里以便取用其他字段。
    """

    user_id: str
    tournament_id: str
    active_games: tuple[str, ...] = ()
    active_count: int = 0
    active_game_details: tuple[Mapping[str, Any], ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_scoped(self) -> bool:
        """参赛令牌绑定具体锦标赛；全局令牌的 tournament_id 为空串。"""
        return bool(self.tournament_id)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> MyInfo:
        entries = tuple(raw.get("active_games") or ())
        games = tuple(gid for gid in (game_id_of(entry) for entry in entries) if gid)
        return cls(
            user_id=str(raw.get("user_id", "")),
            tournament_id=str(raw.get("tournament_id", "")),
            active_games=games,
            active_count=int(raw.get("active_count", len(games)) or 0),
            active_game_details=tuple(e for e in entries if isinstance(e, Mapping)),
            extra={
                key: value
                for key, value in raw.items()
                if key not in {"user_id", "tournament_id", "active_games", "active_count"}
            },
        )


@dataclass(frozen=True, slots=True)
class RankingEntry:
    user_id: str
    rank: int | None
    total_score: int
    place_points: int
    god_count: int
    games_played: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> RankingEntry:
        rank = raw.get("rank")
        return cls(
            user_id=str(raw.get("user_id", "")),
            rank=None if rank is None else int(rank),
            total_score=int(raw.get("total_score", 0) or 0),
            place_points=int(raw.get("place_points", 0) or 0),
            god_count=int(raw.get("god_count", 0) or 0),
            games_played=int(raw.get("games_played", 0) or 0),
            extra={
                key: value
                for key, value in raw.items()
                if key
                not in {"user_id", "rank", "total_score", "place_points", "god_count", "games_played"}
            },
        )


@dataclass(frozen=True, slots=True)
class StageInfo:
    number: int
    role: str
    total: int
    name: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> StageInfo:
        return cls(
            number=int(raw.get("no", 0) or 0),
            role=str(raw.get("role", "")),
            total=int(raw.get("total", 0) or 0),
            name=str(raw.get("name", "")),
            extra={
                key: value
                for key, value in raw.items()
                if key not in {"no", "role", "total", "name"}
            },
        )


@dataclass(frozen=True, slots=True)
class TournamentState:
    """``GET /api/tournaments/{id}`` 的响应（开赛后另含多阶段字段组）。"""

    tournament_id: str
    status: str
    config: TournamentConfig
    my_games: tuple[str, ...] = ()
    my_games_by_batch: Mapping[str, Sequence[str]] = field(default_factory=dict)
    ranking: tuple[RankingEntry, ...] = ()
    ready_users: int = 0
    registered_users: int = 0
    voided_reason: str = ""
    stage: StageInfo | None = None
    stage_status: str | None = None
    stage_crashed: bool = False
    qualified: bool | None = None
    qualify_role: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def accepts_ready(self) -> bool:
        return self.status in READY_STATUSES

    @property
    def is_eliminated(self) -> bool:
        """阶段确认期内已无本阶段资格。"""
        return self.status == STATUS_STAGE_OPEN and self.qualified is False

    @property
    def has_stage_qualification(self) -> bool:
        return bool(self.qualified)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> TournamentState:
        stage_raw = raw.get("stage")
        games = tuple(str(g) for g in raw.get("my_games") or ())
        by_batch = raw.get("my_games_by_batch") or {}
        qualified = raw.get("qualified")
        role = raw.get("qualify_role")
        return cls(
            tournament_id=str(raw.get("tournament_id", "")),
            status=str(raw.get("status", "")),
            config=TournamentConfig.parse(raw.get("config") or {}),
            my_games=games,
            my_games_by_batch={
                str(batch): tuple(str(g) for g in ids) for batch, ids in by_batch.items()
            },
            ranking=tuple(RankingEntry.parse(entry) for entry in raw.get("ranking") or ()),
            ready_users=int(raw.get("ready_users", 0) or 0),
            registered_users=int(raw.get("registered_users", 0) or 0),
            voided_reason=str(raw.get("voided_reason", "")),
            stage=None if not isinstance(stage_raw, Mapping) else StageInfo.parse(stage_raw),
            stage_status=None if raw.get("stage_status") is None else str(raw["stage_status"]),
            stage_crashed=bool(raw.get("stage_crashed", False)),
            qualified=None if qualified is None else bool(qualified),
            qualify_role=None if role in (None, "") else str(role),
            extra={
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "tournament_id",
                    "status",
                    "config",
                    "my_games",
                    "my_games_by_batch",
                    "ranking",
                    "ready_users",
                    "registered_users",
                    "voided_reason",
                    "stage",
                    "stage_status",
                    "stage_crashed",
                    "qualified",
                    "qualify_role",
                }
            },
        )


def known_game_ids(*sources: Iterable[str]) -> tuple[str, ...]:
    """合并多个场次来源并去重，保持首次出现的顺序。"""
    seen: dict[str, None] = {}
    for source in sources:
        for game_id in source:
            seen.setdefault(game_id, None)
    return tuple(seen)
