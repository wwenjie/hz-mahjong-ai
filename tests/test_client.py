import pytest

from majiang.client.api import GameEnvelope, GuideVersion, PlatformApi
from majiang.client.errors import ApiError, is_permanent, is_race, is_retryable, make_api_error
from majiang.client.models import (
    STATUS_FINISHED,
    STATUS_REGISTERING,
    STATUS_STAGE_DONE,
    STATUS_STAGE_OPEN,
    ModelError,
    MyInfo,
    TournamentConfig,
    TournamentState,
)

# 以下两份响应是 2026-09-23 实测（指南 v34）的原样返回
REAL_ME = {
    "active_count": 0,
    "active_games": [],
    "tournament_id": "t_0cfde5a00075",
    "user_id": "u_a2dd7a6524f8",
}

REAL_CONFIG = {
    "M": 10,
    "Rounds": 1,
    "BaseScore": 1,
    "Name": "",
    "Description": "",
    "Kind": "test",
    "YouCaiBiKao": False,
    "StartAt": 1790138853,
    "RegisterDeadlineAt": 0,
    "DiscardTimeoutSec": 3,
    "PengTimeoutSec": 1,
    "ChiTimeoutSec": 1,
    "TimeoutMin": 30,
    "OnlineConfirm": False,
}

REAL_TOURNAMENT = {
    "config": REAL_CONFIG,
    "my_games": [],
    "my_games_by_batch": {},
    "ranking": [],
    "ready_users": 0,
    "registered_users": 4,
    "status": "registering",
    "tournament_id": "t_0cfde5a00075",
    "voided_reason": "",
}


class FakeTransport:
    def __init__(self, routes: dict[tuple[str, str], object]) -> None:
        self.routes = routes
        self.calls: list[dict[str, object]] = []

    def request(self, method, path, *, token=None, body=None, timeout=None):
        self.calls.append(
            {"method": method, "path": path, "token": token, "body": body, "timeout": timeout}
        )
        key = (method, path)
        if key not in self.routes:
            raise AssertionError(f"未预期的请求: {method} {path}")
        result = self.routes[key]
        if isinstance(result, Exception):
            raise result
        return result


def test_me_parses_real_payload() -> None:
    info = MyInfo.parse(REAL_ME)
    assert info.user_id == "u_a2dd7a6524f8"
    assert info.tournament_id == "t_0cfde5a00075"
    assert info.is_scoped is True
    assert info.active_games == () and info.active_count == 0


def test_global_token_is_not_scoped() -> None:
    info = MyInfo.parse({"user_id": "u_1", "tournament_id": "", "active_games": []})
    assert info.is_scoped is False


def test_active_games_are_objects_not_strings() -> None:
    info = MyInfo.parse(
        {
            "user_id": "u_1",
            "tournament_id": "t_1",
            "active_count": 2,
            "active_games": [
                {"game_id": "t_1_r1_b0_t0", "round_no": 1},
                {"game_id": "t_1_r1_b1_t0", "round_no": 1},
            ],
        }
    )
    assert info.active_games == ("t_1_r1_b0_t0", "t_1_r1_b1_t0")
    assert info.active_count == 2
    assert info.active_game_details[0] == {"game_id": "t_1_r1_b0_t0", "round_no": 1}


def test_active_games_also_accept_bare_strings() -> None:
    info = MyInfo.parse({"active_games": ["g_1", {"game_id": "g_2"}, {"no_id": 1}]})
    assert info.active_games == ("g_1", "g_2")


def test_config_parses_real_payload() -> None:
    config = TournamentConfig.parse(REAL_CONFIG)
    assert config.max_concurrent_games == 10
    assert config.rounds_per_game == 1
    assert config.base_score == 1
    assert config.you_cai_bi_kao is False
    assert config.discard_timeout_sec == 3.0
    assert config.peng_timeout_sec == 1.0
    assert config.chi_timeout_sec == 1.0
    assert config.kind == "test" and config.is_test_room
    assert config.timeout_min == 30
    assert config.online_confirm is False
    assert config.start_at == 1790138853


def test_config_tolerates_unknown_keys() -> None:
    config = TournamentConfig.parse({**REAL_CONFIG, "FutureFlag": True, "Extra": {"a": 1}})
    assert config.extra == {"FutureFlag": True, "Extra": {"a": 1}}
    assert config.max_concurrent_games == 10


@pytest.mark.parametrize("missing", ["M", "Rounds", "BaseScore"])
def test_config_requires_core_keys(missing: str) -> None:
    raw = {k: v for k, v in REAL_CONFIG.items() if k != missing}
    with pytest.raises(ModelError, match=missing):
        TournamentConfig.parse(raw)


def test_config_applies_documented_timeout_defaults() -> None:
    raw = {k: v for k, v in REAL_CONFIG.items()}
    raw.pop("DiscardTimeoutSec")
    raw.pop("PengTimeoutSec")
    raw.pop("ChiTimeoutSec")
    config = TournamentConfig.parse(raw)
    assert (config.discard_timeout_sec, config.peng_timeout_sec, config.chi_timeout_sec) == (
        3.0,
        1.0,
        1.0,
    )


def test_tournament_parses_real_payload() -> None:
    state = TournamentState.parse(REAL_TOURNAMENT)
    assert state.tournament_id == "t_0cfde5a00075"
    assert state.status == STATUS_REGISTERING
    assert state.registered_users == 4 and state.ready_users == 0
    assert state.my_games == () and state.ranking == ()
    assert state.accepts_ready is True
    assert state.is_terminal is False
    assert state.stage is None and state.stage_crashed is False
    assert state.qualified is None


def test_tournament_stage_fields() -> None:
    raw = {
        **REAL_TOURNAMENT,
        "status": STATUS_STAGE_OPEN,
        "stage": {"no": 2, "role": "qualify", "total": 4, "name": "测试房间-第2轮"},
        "stage_status": "open",
        "stage_crashed": True,
        "qualified": True,
        "qualify_role": "backup",
    }
    state = TournamentState.parse(raw)
    assert state.stage is not None and state.stage.number == 2
    assert state.stage.name == "测试房间-第2轮"
    assert state.stage.role == "qualify" and state.stage.total == 4
    assert state.stage_crashed is True
    assert state.qualified is True and state.qualify_role == "backup"
    assert state.has_stage_qualification is True
    assert state.is_eliminated is False


def test_eliminated_when_stage_open_without_qualification() -> None:
    state = TournamentState.parse(
        {**REAL_TOURNAMENT, "status": STATUS_STAGE_OPEN, "qualified": False}
    )
    assert state.is_eliminated is True


@pytest.mark.parametrize("status", [STATUS_STAGE_DONE, STATUS_FINISHED])
def test_ready_rejected_outside_ready_phases(status: str) -> None:
    state = TournamentState.parse({**REAL_TOURNAMENT, "status": status})
    assert state.accepts_ready is False
    assert state.is_terminal is (status == STATUS_FINISHED)


def test_ranking_entry_parsing() -> None:
    state = TournamentState.parse(
        {
            **REAL_TOURNAMENT,
            "status": "running",
            "ranking": [
                {
                    "user_id": "u_1",
                    "total_score": 42,
                    "place_points": 7,
                    "god_count": 3,
                    "games_played": 8,
                    "rank": 1,
                }
            ],
        }
    )
    entry = state.ranking[0]
    assert (entry.rank, entry.total_score, entry.place_points) == (1, 42, 7)
    assert (entry.god_count, entry.games_played) == (3, 8)


def test_game_envelope_pending_and_finished() -> None:
    pending = GameEnvelope.parse({"pending": True})
    assert pending.pending is True and pending.snapshot is None
    finished = GameEnvelope.parse({"finished": True, "snapshot": {"scores": [0, 0, 0, 0]}})
    assert finished.finished is True
    assert finished.snapshot == {"scores": [0, 0, 0, 0]}


def test_game_envelope_gap_from_snapshot_or_envelope() -> None:
    assert GameEnvelope.parse({"gap": True, "snapshot": {}}).gap is True
    assert GameEnvelope.parse({"snapshot": {"gap": True}}).gap is True
    assert GameEnvelope.parse({"snapshot": {}}).gap is False


def test_game_envelope_events_and_seq() -> None:
    envelope = GameEnvelope.parse(
        {"seq": 12, "events": [{"seq": 11, "type": "tile_drawn"}, {"type": "discard"}]}
    )
    assert envelope.seq == 12
    assert len(envelope.events) == 2
    assert envelope.events[0]["type"] == "tile_drawn"


def test_api_calls_carry_token_and_path() -> None:
    transport = FakeTransport(
        {
            ("GET", "/api/me"): REAL_ME,
            ("GET", "/api/tournaments/me/rules"): {"config": REAL_CONFIG},
            ("POST", "/api/tournaments/t_0cfde5a00075/ready"): {},
            ("GET", "/api/tournaments/t_0cfde5a00075"): REAL_TOURNAMENT,
        }
    )
    api = PlatformApi(transport, "tok")
    api.me()
    api.rules()
    api.ready("t_0cfde5a00075")
    api.tournament("t_0cfde5a00075")
    assert [call["token"] for call in transport.calls] == ["tok"] * 4
    assert transport.calls[2]["method"] == "POST"


def test_submit_action_sends_body_and_uses_explicit_game_path() -> None:
    transport = FakeTransport({("POST", "/api/games/g_1/action"): {"ok": True}})
    PlatformApi(transport, "tok").submit_action("g_1", {"action": "discard", "tile": "1w"})
    assert transport.calls[0]["body"] == {"action": "discard", "tile": "1w"}


def test_game_state_passes_seq_in_query_and_timeout() -> None:
    transport = FakeTransport({("GET", "/api/games/g_1/state?seq=7"): {"seq": 9, "events": []}})
    envelope = PlatformApi(transport, "tok").game_state("g_1", 7, timeout=33.0)
    assert envelope.seq == 9
    assert transport.calls[0]["path"] == "/api/games/g_1/state?seq=7"
    assert transport.calls[0]["timeout"] == 33.0


def test_guide_version_is_unauthenticated() -> None:
    transport = FakeTransport(
        {
            ("GET", "/portal/api/guide/version"): {
                "version": 34,
                "updated_at": "2026-09-14",
                "changes": [
                    {"version": 34, "type": "added", "summary": "新增垫底行"},
                    {"version": 33, "type": "breaking", "summary": "旧行为不再兼容"},
                    {"version": 20, "type": "breaking", "summary": "更早的破坏性变更"},
                ],
            }
        }
    )
    version = PlatformApi(transport, "tok").guide_version()
    assert transport.calls[0]["token"] is None
    assert version.version == 34
    assert version.breaking_since(21) == ("旧行为不再兼容",)
    assert version.breaking_since(34) == ()


def test_error_classification() -> None:
    assert is_retryable(make_api_error(429, {"code": "RATE_LIMITED"})) is True
    assert is_retryable(make_api_error(409, {"code": "MATCH_BUSY"})) is True
    assert is_permanent(make_api_error(403, {"code": "FEATURE_DISABLED"})) is True
    assert is_permanent(make_api_error(409, {"code": "TOKEN_NOT_SCOPED"})) is True
    assert is_race(make_api_error(409, {"code": "INVALID_ACTION"})) is True
    assert is_retryable(make_api_error(409, {"code": "INVALID_ACTION"})) is False
    unknown = make_api_error(500, None, "boom")
    assert unknown.code == "HTTP_500" and not is_permanent(unknown)


def test_error_carries_body_for_diagnosis() -> None:
    error = make_api_error(400, {"code": "INVALID_INPUT", "message": "手牌必须恰好 13 张"}, "raw")
    assert isinstance(error, ApiError)
    assert error.code == "INVALID_INPUT" and error.status == 400
    assert "13 张" in error.message and error.body == "raw"


# --- 指南版本自检（tasks.md 1.4） ---------------------------------------------------


def test_tournament_gone_is_retryable_not_permanent() -> None:
    """指南 v35 的破坏性变更：TOURNAMENT_GONE 与 TOURNAMENT_NOT_FOUND 共用 404。

    两者含义相反——前者「房暂时不可达」应重试，后者「房不存在」应放弃。判型必须用
    响应体的 code。这里把结论钉住：若哪天有人只看状态码，或把它误并入永久集合，
    比赛时会把可恢复的场面当成淘汰。
    """
    gone = make_api_error(404, {"code": "TOURNAMENT_GONE", "message": "房暂时不可达"})
    missing = make_api_error(404, {"code": "TOURNAMENT_NOT_FOUND"})
    assert gone.status == missing.status == 404
    assert is_retryable(gone) is True
    assert is_permanent(gone) is False
    assert is_retryable(missing) is False
    assert is_permanent(missing) is True


def test_error_without_code_falls_back_to_retryable_side() -> None:
    """响应体没有 code 时退化为 HTTP_<status>，落在「未知」一侧（重试），而非放弃。"""
    bare = make_api_error(404, None, "raw")
    assert bare.code == "HTTP_404"
    assert is_retryable(bare) is False and is_permanent(bare) is False


def test_guide_warning_is_silent_when_platform_matches_or_is_older() -> None:
    from majiang.cli import KNOWN_GUIDE_VERSION, guide_version_warning

    assert guide_version_warning(GuideVersion(version=KNOWN_GUIDE_VERSION)) is None
    assert guide_version_warning(GuideVersion(version=KNOWN_GUIDE_VERSION - 1)) is None


def test_guide_warning_lists_breaking_changes_newer_than_known() -> None:
    from majiang.cli import KNOWN_GUIDE_VERSION, guide_version_warning

    info = GuideVersion(
        version=KNOWN_GUIDE_VERSION + 2,
        updated_at="2026-10-01",
        changes=(
            {"version": KNOWN_GUIDE_VERSION + 1, "type": "feature", "summary": "新增回放接口"},
            {"version": KNOWN_GUIDE_VERSION + 2, "type": "breaking", "summary": "快照改名 my_hand"},
            # 与已知版本相同或更旧的破坏性变更不应出现
            {"version": KNOWN_GUIDE_VERSION, "type": "breaking", "summary": "历史变更"},
        ),
    )
    warning = guide_version_warning(info)
    assert warning is not None
    assert "快照改名 my_hand" in warning
    assert "新增回放接口" not in warning
    assert "历史变更" not in warning
    assert f"v{KNOWN_GUIDE_VERSION + 2}" in warning


def test_guide_warning_notes_updates_without_breaking_flag() -> None:
    from majiang.cli import KNOWN_GUIDE_VERSION, guide_version_warning

    info = GuideVersion(version=KNOWN_GUIDE_VERSION + 1, changes=())
    warning = guide_version_warning(info)
    assert warning is not None
    assert "未标记破坏性变更" in warning


def test_guide_warning_flags_unrecognisable_payload() -> None:
    """version 缺失或非正数说明响应形状变了——这本身就是破坏性变更的信号。"""
    from majiang.cli import guide_version_warning

    warning = guide_version_warning(GuideVersion(version=0))
    assert warning is not None
    assert "无法识别" in warning
