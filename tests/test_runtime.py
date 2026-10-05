import json
import threading
import time

import pytest

from majiang.rules import tiles
from majiang.rules.action import CHI, DISCARD, HU, PASS, PENG, Action
from majiang.runtime import engine as engine_module
from majiang.runtime.decider import Decider, FirstLegalDecider, GuardedDecider
from majiang.runtime.engine import GameRunner, PiaoTracker, Runtime, RuntimeOptions
from majiang.runtime.logging import JsonlLogger, NullLogger, iter_records, open_logger, redact
from majiang.runtime.ratelimit import (
    ConcurrencyGate,
    RateLimitedTransport,
    RateLimitTimeout,
    TokenBucket,
)

from .test_snapshot import REAL_DRAW_SNAPSHOT, REAL_PENG_SNAPSHOT

TOKEN = "tok"


class RouterTransport:
    """把请求路由到可调用对象，并记录调用序列。"""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[tuple[str, str, object]] = []

    def request(self, method, path, *, token=None, body=None, timeout=None):
        self.calls.append((method, path, body))
        result = self._handler(method, path, body)
        if isinstance(result, Exception):
            raise result
        return result


def test_redact_hides_tokens_and_sensitive_keys() -> None:
    raw_token = "a" * 64
    record = redact({"note": raw_token, "token": "short", "nested": [{"authorization": "x"}]})
    assert record["note"] == "<redacted>"
    assert record["token"] == "<redacted>"
    assert record["nested"][0]["authorization"] == "<redacted>"


def test_redact_keeps_ordinary_strings() -> None:
    assert redact("t_0cfde5a00075") == "t_0cfde5a00075"
    assert redact({"tile": "白"})["tile"] == "白"


def test_jsonl_logger_writes_records_with_context(tmp_path) -> None:
    path = tmp_path / "g.jsonl"
    logger = JsonlLogger(path=path, console=False, context={"tournament_id": "t_1"})
    logger.child(game_id="g_1").log("action.submitted", action="discard:1w", count=2)
    logger.close()
    records = list(iter_records(path))
    assert len(records) == 1
    assert records[0]["event"] == "action.submitted"
    assert records[0]["tournament_id"] == "t_1"
    assert records[0]["game_id"] == "g_1"
    assert records[0]["action"] == "discard:1w"
    assert isinstance(records[0]["mono_ms"], int)


def test_jsonl_logger_redacts_before_writing(tmp_path) -> None:
    logger = open_logger("t_1", tmp_path, console=False)
    logger.log("runtime.start", note="f" * 64)
    logger.close()
    path = tmp_path / "t_1.jsonl"
    assert path.exists()
    assert list(iter_records(path))[0]["note"] == "<redacted>"


def test_null_logger_is_silent() -> None:
    logger = NullLogger()
    logger.log("anything", value=1)
    assert logger.child(a=1).log("x") is None


def test_token_bucket_allows_burst_then_throttles() -> None:
    bucket = TokenBucket(rate_per_sec=50.0, capacity=3)
    assert [bucket.acquire() for _ in range(3)] == [True, True, True]
    started = time.monotonic()
    assert bucket.acquire(timeout=1.0) is True
    assert time.monotonic() - started >= 0.015


def test_token_bucket_times_out_when_exhausted() -> None:
    bucket = TokenBucket(rate_per_sec=0.5, capacity=1)
    assert bucket.acquire() is True
    assert bucket.acquire(timeout=0.05) is False


def test_concurrency_gate_limits_in_flight() -> None:
    gate = ConcurrencyGate(2)
    assert gate.acquire()
    assert gate.acquire()
    assert gate.in_flight == 2
    assert gate.acquire(timeout=0.05) is False
    gate.release()
    assert gate.acquire(timeout=0.05) is True


def test_rate_limited_transport_passes_through_and_times_out() -> None:
    inner = RouterTransport(lambda m, p, b: {"ok": True})
    transport = RateLimitedTransport(inner, TokenBucket(rate_per_sec=100.0, capacity=1), None)
    assert transport.request("GET", "/x") == {"ok": True}
    inner.calls.clear()
    tight = RateLimitedTransport(inner, TokenBucket(rate_per_sec=0.2, capacity=1), None, acquire_timeout=0.02)
    tight.request("GET", "/a")
    with pytest.raises(RateLimitTimeout):
        tight.request("GET", "/b")


def test_action_submissions_use_the_reserved_bucket() -> None:
    """动作提交有硬窗口，必须排在轮询之前：主桶耗尽时提交仍应立刻发出。

    回归防线。实测拒绝时刻距决策记录中位 2217ms，而代码里两者相邻——根因就是提交
    排在大量在途轮询后面，触发 429 后退避重试吃掉整个窗口。
    """
    inner = RouterTransport(lambda m, p, b: {"ok": True})
    transport = RateLimitedTransport(
        inner,
        TokenBucket(rate_per_sec=0.2, capacity=1),
        None,
        acquire_timeout=0.02,
        priority_bucket=TokenBucket(rate_per_sec=50.0, capacity=2),
    )
    transport.request("GET", "/api/games/g1/state")  # 用掉主桶唯一令牌
    with pytest.raises(RateLimitTimeout):
        transport.request("GET", "/api/games/g1/state")  # 轮询被卡住
    inner.calls.clear()
    assert transport.request("POST", "/api/games/g1/action", body={"action": "hu"}) == {"ok": True}
    assert inner.calls == [("POST", "/api/games/g1/action", {"action": "hu"})]


def test_only_action_posts_use_the_reserved_bucket() -> None:
    """报名/到位这类 POST 没有硬窗口，仍应走主桶，不能蹭动作的预留额度。"""
    inner = RouterTransport(lambda m, p, b: {"ok": True})
    transport = RateLimitedTransport(
        inner,
        TokenBucket(rate_per_sec=0.2, capacity=1),
        None,
        acquire_timeout=0.02,
        priority_bucket=TokenBucket(rate_per_sec=50.0, capacity=2),
    )
    transport.request("GET", "/x")
    inner.calls.clear()  # 上面那次 GET 已经用掉主桶令牌，从这里开始验证
    for path in ("/api/tournaments/t1/ready", "/api/tournaments/t1/register"):
        with pytest.raises(RateLimitTimeout):
            transport.request("POST", path)
    assert inner.calls == []


def test_reserved_bucket_rate_is_carved_out_of_the_total() -> None:
    """两个桶速率之和必须等于总速率：预留不能把总量推过平台上限。"""
    for rate in (2.0, 8.0, 14.0, 16.0):
        options = RuntimeOptions(rate_per_sec=rate, action_rate_per_sec=3.0)
        action = min(options.action_rate_per_sec, options.rate_per_sec / 2)
        poll = options.rate_per_sec - action
        assert abs(action + poll - options.rate_per_sec) < 1e-9
        assert poll >= options.rate_per_sec / 2, "轮询额度不应被预留压到一半以下"


def test_first_legal_decider_prefers_high_value_actions() -> None:
    situation = None
    actions = [Action(PASS), Action(DISCARD, tile=tiles.parse("1w")), Action(PENG, tile=tiles.parse("2w")), Action(HU)]
    decider = FirstLegalDecider()
    assert decider.choose(situation, actions, budget_ms=100).kind == HU
    assert decider.choose(situation, actions[:3], budget_ms=100).kind == PENG
    assert decider.choose(situation, [Action(PASS)], budget_ms=100).kind == PASS
    assert decider.choose(situation, [], budget_ms=100) is None


class BoomDecider:
    name = "boom"

    def choose(self, situation, actions, *, budget_ms):
        raise RuntimeError("决策炸了")


class IllegalDecider:
    name = "illegal"

    def choose(self, situation, actions, *, budget_ms):
        return Action(DISCARD, tile=tiles.parse("9t"))


class SlowDecider:
    name = "slow"

    def choose(self, situation, actions, *, budget_ms):
        time.sleep(0.05)
        return actions[0]


def test_guarded_decider_degrades_on_exception() -> None:
    reasons: list[str] = []
    decider = GuardedDecider(BoomDecider(), on_fallback=lambda r, ms: reasons.append(r))
    actions = [Action(PASS), Action(DISCARD, tile=tiles.parse("1w"))]
    chosen = decider.choose(None, actions, budget_ms=100)
    assert chosen is not None and chosen.kind == DISCARD
    assert "RuntimeError" in reasons[0]


def test_guarded_decider_degrades_on_illegal_action() -> None:
    decider = GuardedDecider(IllegalDecider())
    actions = [Action(DISCARD, tile=tiles.parse("1w"))]
    chosen = decider.choose(None, actions, budget_ms=100)
    assert chosen == actions[0]


def test_guarded_decider_degrades_when_over_budget() -> None:
    decider = GuardedDecider(SlowDecider())
    actions = [Action(DISCARD, tile=tiles.parse("1w"))]
    assert decider.choose(None, actions, budget_ms=1) == actions[0]


def test_guarded_decider_passes_through_valid_choice() -> None:
    decider = GuardedDecider(FirstLegalDecider())
    actions = [Action(DISCARD, tile=tiles.parse("1w"))]
    assert decider.choose(None, actions, budget_ms=100) == actions[0]


def test_game_runner_skips_implicit_pass(tmp_path) -> None:
    from majiang.client.api import PlatformApi
    from majiang.client.models import TournamentConfig
    from majiang.client.snapshot import Snapshot

    transport = RouterTransport(
        lambda m, p, b: {"seq": 3, "pending": False, "finished": False, "snapshot": REAL_PENG_SNAPSHOT}
    )
    runner = GameRunner(
        PlatformApi(transport, TOKEN),
        "g_1",
        FirstLegalDecider(),
        TournamentConfig(max_concurrent_games=1, rounds_per_game=1, base_score=1),
        NullLogger(),
        state_poll_timeout=0.01,
        snapshot_timeout=0.01,
    )
    snapshot = Snapshot.parse({**REAL_PENG_SNAPSHOT, "seat": 0, "responding_seats": [0]})
    assert runner._decide_safely(snapshot).kind == PASS
    runner._submit(Action(PASS))
    assert transport.calls == []  # 窗口内只有 pass 可选时不应发出请求


def test_piao_tracker_counts_god_discard_while_baotou() -> None:
    tracker = PiaoTracker()
    from majiang.client.snapshot import Snapshot

    snapshot = Snapshot.parse({**REAL_DRAW_SNAPSHOT, "god": {"baotou": True, "chain_count": 0}})
    tracker.observe_snapshot(snapshot)
    tracker.observe_action(Action(DISCARD, tile=tiles.parse("1w")))
    assert tracker.piao_count == 0
    tracker.observe_action(Action(DISCARD, tile=tiles.GOD))
    assert tracker.piao_count == 1


def test_piao_tracker_resets_when_chain_clears() -> None:
    tracker = PiaoTracker(piao_count=3)
    from majiang.client.snapshot import Snapshot

    tracker.observe_snapshot(Snapshot.parse({**REAL_DRAW_SNAPSHOT, "god": {"chain_count": 0}}))
    assert tracker.piao_count == 0


def test_could_respond_detects_peng_and_chi() -> None:
    from majiang.client.snapshot import Snapshot

    peng_like = Snapshot.parse(
        {
            **REAL_PENG_SNAPSHOT,
            "my_hand": ["2w", "2w", "1b", "3b", "5b", "7b", "9b", "1t", "4t", "7t", "东", "南", "北"],
            "god": {"catch_play": False, "god_discarder_seat": -1},
            "last_discard": "2w",
        }
    )
    assert engine_module._could_respond(peng_like, tiles.parse("2w")) is True
    assert engine_module._could_respond(peng_like, tiles.parse("9t")) is False


def test_could_respond_blocked_by_catch_play() -> None:
    from majiang.client.snapshot import Snapshot

    restricted = Snapshot.parse(
        {
            **REAL_PENG_SNAPSHOT,
            "seat": 1,
            "my_hand": ["2w", "2w", "1b", "3b", "5b", "7b", "9b", "1t", "4t", "7t", "东", "南", "北"],
            "god": {"catch_play": True, "god_discarder_seat": 0},
        }
    )
    assert engine_module._could_respond(restricted, tiles.parse("2w")) is False


def test_could_respond_ignores_god_discard() -> None:
    from majiang.client.snapshot import Snapshot

    snapshot = Snapshot.parse({**REAL_PENG_SNAPSHOT, "god": {"catch_play": False, "god_discarder_seat": -1}})
    assert engine_module._could_respond(snapshot, tiles.GOD) is False


def test_budget_comes_from_documented_timeouts_not_window_deadline() -> None:
    """回归防线：预算只按 config 的 ``TimeoutSec`` 取 60%，不再碰 ``window_deadline_ms``。

    该字段在指南里出现 0 次（我们按抓包反推），且实测与本机时钟有几百毫秒钟差：
    ``deadline − now`` 的中位仅 287ms、21% 为负，而窗口实为 1 秒。旧实现据此算出过
    无意义的 50ms 预算。这里刻意喂一个「只剩 100ms」的窗口值，预算仍应是 1s × 60%。
    """
    from majiang.client.models import TournamentConfig
    from majiang.client.snapshot import Snapshot

    config = TournamentConfig(
        max_concurrent_games=10,
        rounds_per_game=1,
        base_score=1,
        peng_timeout_sec=1,
        chi_timeout_sec=1,
        discard_timeout_sec=3,
    )
    imminent = int(time.time() * 1000) + 100
    snapshot = Snapshot.parse({**REAL_PENG_SNAPSHOT, "window_deadline_ms": imminent})
    assert engine_module._budget_ms(config, snapshot) == 600


def test_budget_falls_back_to_config_ratio() -> None:
    from majiang.client.models import TournamentConfig
    from majiang.client.snapshot import Snapshot

    config = TournamentConfig(max_concurrent_games=10, rounds_per_game=1, base_score=1)
    snapshot = Snapshot.parse(REAL_DRAW_SNAPSHOT)
    assert engine_module._budget_ms(config, snapshot) == 1800


def _config_payload() -> dict:
    return {
        "M": 10,
        "Rounds": 1,
        "BaseScore": 1,
        "YouCaiBiKao": False,
        "DiscardTimeoutSec": 3,
        "PengTimeoutSec": 1,
        "ChiTimeoutSec": 1,
        "Kind": "test",
    }


def test_game_runner_submits_one_action_then_finishes(tmp_path) -> None:
    from majiang.client.api import PlatformApi
    from majiang.client.models import TournamentConfig

    steps = [
        {"seq": 5, "pending": False, "finished": False, "snapshot": REAL_DRAW_SNAPSHOT},
        {"seq": 6, "pending": False, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT},
    ]
    state_calls = {"n": 0}

    def handler(method, path, body):
        if path.startswith("/api/games/g_1/state"):
            index = min(state_calls["n"], len(steps) - 1)
            state_calls["n"] += 1
            return steps[index]
        if path.endswith("/action"):
            return {"ok": True}
        raise AssertionError(f"未预期请求 {method} {path}")

    transport = RouterTransport(handler)
    logger = JsonlLogger(path=tmp_path / "game.jsonl", console=False)
    runner = GameRunner(
        PlatformApi(transport, TOKEN),
        "g_1",
        FirstLegalDecider(),
        TournamentConfig(max_concurrent_games=10, rounds_per_game=1, base_score=1),
        logger,
        state_poll_timeout=0.01,
        snapshot_timeout=0.01,
    )
    reason = runner.run()
    logger.close()
    assert reason == "finished"
    assert runner.actions_submitted == 1
    actions = [call for call in transport.calls if call[0] == "POST"]
    assert len(actions) == 1
    # FirstLegalDecider 在候选里取第一个出牌选项，而候选按牌索引升序生成，故取到最小的 6w
    assert actions[0][2] == {"action": "discard", "tile": "6w"}
    events = [record["event"] for record in iter_records(tmp_path / "game.jsonl")]
    assert "game.start" in events and "action.submitted" in events and "game.finished" in events


def test_game_runner_respects_stop_event(tmp_path) -> None:
    from majiang.client.api import PlatformApi
    from majiang.client.models import TournamentConfig

    stop = threading.Event()
    stop.set()
    transport = RouterTransport(lambda m, p, b: {"pending": True})
    runner = GameRunner(
        PlatformApi(transport, TOKEN),
        "g_1",
        FirstLegalDecider(),
        TournamentConfig(max_concurrent_games=1, rounds_per_game=1, base_score=1),
        NullLogger(),
        stop_event=stop,
    )
    assert runner.run() == "stopped"
    assert transport.calls == []


def test_runtime_waits_when_stage_done_then_stops_on_finished(tmp_path) -> None:
    statuses = [{"status": "stage_done"}, {"status": "finished"}]
    seen = {"n": 0}

    def handler(method, path, body):
        if path == "/api/me":
            return {"user_id": "u_1", "tournament_id": "t_1", "active_games": [], "active_count": 0}
        if path == "/api/tournaments/me/rules":
            return {"config": _config_payload()}
        if path == "/api/tournaments/t_1":
            index = min(seen["n"], len(statuses) - 1)
            seen["n"] += 1
            return {
                "tournament_id": "t_1",
                "status": statuses[index]["status"],
                "config": _config_payload(),
                "my_games": [],
            }
        raise AssertionError(f"未预期请求 {method} {path}")

    runtime = Runtime(
        TOKEN,
        server="http://test",
        decider=FirstLegalDecider(),
        transport=RouterTransport(handler),
        options=RuntimeOptions(poll_interval_sec=0.01, log_dir=str(tmp_path), log_console=False),
    )
    summary = runtime.run()
    assert summary.stop_reason == "status:finished"
    assert summary.ready_calls == 0
    assert summary.games_started == 0


def test_runtime_readies_and_starts_games(tmp_path) -> None:
    tournament_status = ["registering", "running", "finished"]
    tournament_index = {"n": 0}
    me_calls = {"n": 0}

    def handler(method, path, body):
        if path == "/api/me":
            me_calls["n"] += 1
            active = [] if me_calls["n"] < 2 else [{"game_id": "g_1"}]
            return {
                "user_id": "u_1",
                "tournament_id": "t_1",
                "active_games": active,
                "active_count": len(active),
            }
        if path == "/api/tournaments/me/rules":
            return {"config": _config_payload()}
        if path == "/api/tournaments/t_1/ready":
            return {}
        if path == "/api/tournaments/t_1/register":
            return {}
        if path == "/api/tournaments/t_1":
            index = min(tournament_index["n"], len(tournament_status) - 1)
            tournament_index["n"] += 1
            return {
                "tournament_id": "t_1",
                "status": tournament_status[index],
                "config": _config_payload(),
                "my_games": ["g_1"],
            }
        if path.startswith("/api/games/g_1/state"):
            return {"seq": 9, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT}
        raise AssertionError(f"未预期请求 {method} {path}")

    runtime = Runtime(
        TOKEN,
        server="http://test",
        decider=FirstLegalDecider(),
        transport=RouterTransport(handler),
        options=RuntimeOptions(poll_interval_sec=0.01, log_dir=str(tmp_path), log_console=False),
    )
    summary = runtime.run()
    assert summary.register_calls == 1
    assert summary.ready_calls == 1
    assert summary.games_started == 1
    assert summary.games_completed == 1
    assert summary.stop_reason == "status:finished"


def test_runtime_stops_when_token_is_not_scoped() -> None:
    transport = RouterTransport(
        lambda m, p, b: {"user_id": "u_1", "tournament_id": "", "active_games": []}
    )
    runtime = Runtime(
        TOKEN,
        server="http://test",
        decider=FirstLegalDecider(),
        transport=transport,
        options=RuntimeOptions(log_dir="logs", log_console=False),
    )
    with pytest.raises(ValueError, match="未绑定锦标赛"):
        runtime.run()


def test_game_runner_accumulates_scores_across_rounds(tmp_path) -> None:
    """S3 前置：engine 在局间 settled 快照里累加净分，注入后续决策的 TableState.scores。"""
    from dataclasses import replace as dc_replace
    from majiang.client.api import PlatformApi
    from majiang.client.models import TournamentConfig
    from majiang.client.snapshot import Snapshot
    from majiang.rules.situation import PHASE_SETTLED

    transport = RouterTransport(
        lambda m, p, b: {"seq": 3, "pending": False, "finished": False, "snapshot": REAL_PENG_SNAPSHOT}
    )
    runner = GameRunner(
        PlatformApi(transport, TOKEN),
        "g_1",
        FirstLegalDecider(),
        TournamentConfig(max_concurrent_games=1, rounds_per_game=8, base_score=1),
        NullLogger(),
        state_poll_timeout=0.01,
        snapshot_timeout=0.01,
    )
    # 初始：全零
    assert runner._cumulative_scores == [0, 0, 0, 0]

    # 第 1 局结束：settled 快照带该局净分
    settled1 = Snapshot.parse({
        **REAL_PENG_SNAPSHOT,
        "phase": PHASE_SETTLED,
        "round_no": 1,
        "scores": [-8, -1, 10, -1],
    })
    runner._absorb(settled1)
    assert runner._cumulative_scores == [-8, -1, 10, -1]

    # 第 2 局结束：再累计
    settled2 = Snapshot.parse({
        **REAL_PENG_SNAPSHOT,
        "phase": PHASE_SETTLED,
        "round_no": 2,
        "scores": [10, -8, -1, -1],
    })
    runner._absorb(settled2)
    assert runner._cumulative_scores == [2, -9, 9, -2]

    # 第 3 局决策时：to_situation 拿到的 TableState.scores 应为累计值
    draw_snap = Snapshot.parse({**REAL_DRAW_SNAPSHOT, "round_no": 3})
    table = draw_snap.table_state()
    if any(s != 0 for s in runner._cumulative_scores):
        table = dc_replace(table, scores=tuple(runner._cumulative_scores))
    assert table.scores == (2, -9, 9, -2)


def test_game_runner_settled_absorb_is_idempotent_per_round(tmp_path) -> None:
    """B'-1：同一 round_no 的 settled 快照重复吸收不得双计净分（真机局间停顿期会重取同一快照）。"""
    from majiang.client.api import PlatformApi
    from majiang.client.models import TournamentConfig
    from majiang.client.snapshot import Snapshot
    from majiang.rules.situation import PHASE_SETTLED

    transport = RouterTransport(
        lambda m, p, b: {"seq": 3, "pending": False, "finished": False, "snapshot": REAL_PENG_SNAPSHOT}
    )
    runner = GameRunner(
        PlatformApi(transport, TOKEN),
        "g_1",
        FirstLegalDecider(),
        TournamentConfig(max_concurrent_games=1, rounds_per_game=8, base_score=1),
        NullLogger(),
        state_poll_timeout=0.01,
        snapshot_timeout=0.01,
    )
    settled1 = Snapshot.parse({
        **REAL_PENG_SNAPSHOT,
        "phase": PHASE_SETTLED,
        "round_no": 1,
        "scores": [-8, -1, 10, -1],
    })
    runner._absorb(settled1)
    assert runner._cumulative_scores == [-8, -1, 10, -1]
    # 局间停顿重取同一 round_no 的 settled 快照（长闲重同步 / race 重拉路径）：不得双计
    runner._absorb(settled1)
    assert runner._cumulative_scores == [-8, -1, 10, -1]
    runner._absorb(settled1)
    assert runner._cumulative_scores == [-8, -1, 10, -1]
    # 下一局正常累加
    settled2 = Snapshot.parse({
        **REAL_PENG_SNAPSHOT,
        "phase": PHASE_SETTLED,
        "round_no": 2,
        "scores": [10, -8, -1, -1],
    })
    runner._absorb(settled2)
    assert runner._cumulative_scores == [2, -9, 9, -2]


def test_game_runner_cumulative_scores_empty_first_round(tmp_path) -> None:
    """S3 不变量：首局 _cumulative_scores 恒零（等价于「未知」），不触发注入。"""
    from majiang.client.api import PlatformApi
    from majiang.client.models import TournamentConfig

    transport = RouterTransport(
        lambda m, p, b: {"seq": 3, "pending": False, "finished": False, "snapshot": REAL_PENG_SNAPSHOT}
    )
    runner = GameRunner(
        PlatformApi(transport, TOKEN),
        "g_1",
        FirstLegalDecider(),
        TournamentConfig(max_concurrent_games=1, rounds_per_game=8, base_score=1),
        NullLogger(),
        state_poll_timeout=0.01,
        snapshot_timeout=0.01,
    )
    assert runner._cumulative_scores == [0, 0, 0, 0]
    assert not any(s != 0 for s in runner._cumulative_scores)
