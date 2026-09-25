"""7.2/7.3 稳定性与边界演练（全部离线，假 transport，不打平台）。

每项演练记录「失败时的表现」：错误怎么被吞掉/上报、恢复后能否续打、
边界窗口里动作会不会被误提交。运行：uv run pytest tests/test_stability.py -v
"""
import threading
import time

import pytest

from majiang.client.api import PlatformApi
from majiang.client.errors import ApiError, TransportError
from majiang.client.models import TournamentConfig
from majiang.runtime.decider import FirstLegalDecider
from majiang.runtime.engine import (
    DEFAULT_LONG_IDLE_RESYNC_ROUNDS,
    GameRunner,
    Runtime,
    RuntimeOptions,
)
from majiang.runtime.logging import NullLogger

from .test_snapshot import REAL_DRAW_SNAPSHOT

TOKEN = "tok"


class RouterTransport:
    """把请求路由到可调用对象，并记录调用序列（含查询参数）。"""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[tuple[str, str, object]] = []

    def request(self, method, path, *, token=None, body=None, timeout=None):
        self.calls.append((method, path, body))
        result = self._handler(method, path, body)
        if isinstance(result, Exception):
            raise result
        return result


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


def _config() -> TournamentConfig:
    return TournamentConfig.parse(_config_payload())


def _runner(handler, *, submit_pass=False) -> GameRunner:
    api = PlatformApi(RouterTransport(handler), TOKEN)
    return GameRunner(api, "g_1", FirstLegalDecider(), _config(), NullLogger(),
                      submit_pass=submit_pass)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)


# ---------- 7.2 网络与平台故障 ----------

def test_network_outage_then_recovers() -> None:
    """网络中断（连续 TransportError）后恢复：对局不死、动作照提、错误被计数。"""
    outages = {"left": 5}
    steps = iter([
        {"seq": 5, "snapshot": REAL_DRAW_SNAPSHOT},
        {"seq": 6, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT},
    ])

    def handler(method, path, body):
        if path.startswith("/api/games/g_1/state") and outages["left"] > 0:
            outages["left"] -= 1
            return TransportError("connection refused")
        if path.startswith("/api/games/g_1/state"):
            return next(steps)
        if path == "/api/games/g_1/action":
            return {}
        raise AssertionError(f"未预期请求 {method} {path}")

    runner = _runner(handler)
    assert runner.run() == "finished"
    assert runner.errors == 5          # 5 次中断全部被计数，没有静默
    assert runner.actions_submitted == 1  # 恢复后决策窗口仍被正确处理


def test_platform_5xx_storm_then_recovers() -> None:
    """平台连续 5xx（未知 code → 可重试分支）：对局不死、恢复后续打。"""
    storm = {"left": 8}
    steps = iter([
        {"seq": 5, "snapshot": REAL_DRAW_SNAPSHOT},
        {"seq": 6, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT},
    ])

    def handler(method, path, body):
        if path.startswith("/api/games/g_1/state") and storm["left"] > 0:
            storm["left"] -= 1
            return ApiError(500, "HTTP_500", "Internal Server Error")
        if path.startswith("/api/games/g_1/state"):
            return next(steps)
        if path == "/api/games/g_1/action":
            return {}
        raise AssertionError(f"未预期请求 {method} {path}")

    runner = _runner(handler)
    assert runner.run() == "finished"
    assert runner.errors == 8
    assert runner.actions_submitted == 1


def test_permanent_game_error_stops_immediately() -> None:
    """失败时的表现：对局级永久错误（GAME_NOT_FOUND）立刻停本场，不重试轰炸。"""
    calls = {"n": 0}

    def handler(method, path, body):
        calls["n"] += 1
        return ApiError(404, "GAME_NOT_FOUND", "gone")

    runner = _runner(handler)
    assert runner.run() == "api:GAME_NOT_FOUND"
    assert calls["n"] == 1  # 一次即停，无重试
    assert runner.errors == 0  # 永久错误不算「错误计数」，是终止信号


def test_stage_crashed_triggers_re_ready(tmp_path) -> None:
    """阶段中断重赛：stage_crashed 翻转后必须重新到位（ready 键含 crashed 位）。"""
    states = [
        {"status": "stage_open", "stage_crashed": False},
        {"status": "stage_open", "stage_crashed": True},   # 平台阶段崩溃
        {"status": "stage_open", "stage_crashed": False},  # 重赛确认：需再次 ready
        {"status": "finished", "stage_crashed": False},
    ]
    seen = {"n": 0}
    ready_calls = {"n": 0}

    def handler(method, path, body):
        if path == "/api/me":
            return {"user_id": "u_1", "tournament_id": "t_1", "active_games": [], "active_count": 0}
        if path == "/api/tournaments/me/rules":
            return {"config": _config_payload()}
        if path == "/api/tournaments/t_1/ready":
            ready_calls["n"] += 1
            return {}
        if path == "/api/tournaments/t_1":
            state = states[min(seen["n"], len(states) - 1)]
            seen["n"] += 1
            return {
                "tournament_id": "t_1",
                "status": state["status"],
                "stage_status": "crashed" if state["stage_crashed"] else "ok",
                "stage_crashed": state["stage_crashed"],
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
    # 4 个不同 key（crashed 位参与 key）各到位一次：漏掉 crashed 位会把重赛确认吞掉
    assert ready_calls["n"] == 3


def test_restart_takeover_via_active_games(tmp_path) -> None:
    """进程被杀后重启接管：本地无持久化，新进程靠 /api/me active_games 重新发现对局。

    确定性设计：平台在「已有动作提交、且收到 seq=0 全量请求」时判对局结束——
    只有新进程的 GameRunner 会从 seq=0 重新同步，旧进程只会增量轮询。
    失败时的表现：旧进程死后对局在平台侧继续；新进程全量重建，不重复提交动作。
    """
    server = {"actions": 0, "finishing": False, "finished": False}

    def handler(method, path, body):
        if path == "/api/me":
            active = [] if server["finished"] else [{"game_id": "g_1"}]
            return {"user_id": "u_1", "tournament_id": "t_1",
                    "active_games": active, "active_count": len(active)}
        if path == "/api/tournaments/me/rules":
            return {"config": _config_payload()}
        if path == "/api/tournaments/t_1":
            return {"tournament_id": "t_1", "status": "running",
                    "config": _config_payload(), "my_games": ["g_1"]}
        if path == "/api/games/g_1/action":
            server["actions"] += 1
            return {}
        if path.startswith("/api/games/g_1/state"):
            if server["finishing"] and "seq=0" in path:
                # 我们「死亡」期间平台侧对局已结束；新进程的首次全量同步看到终态
                server["finished"] = True
                return {"seq": 9, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT}
            if server["actions"] == 0:
                return {"seq": 1, "snapshot": REAL_DRAW_SNAPSHOT}  # 待决策
            return {"seq": 2, "pending": True}  # 增量轮询：无新事件
        raise AssertionError(f"未预期请求 {method} {path}")

    def make_runtime():
        return Runtime(
            TOKEN,
            server="http://test",
            decider=FirstLegalDecider(),
            transport=RouterTransport(handler),
            options=RuntimeOptions(poll_interval_sec=0.01, log_dir=str(tmp_path),
                                   log_console=False),
        )

    first = make_runtime()
    summary1 = first.run(duration_sec=1.0)  # 「被杀」：打了半场就停
    assert summary1.games_started == 1
    assert server["actions"] == 1
    assert not server["finished"]  # 平台视角：对局仍未完

    server["finishing"] = True  # 死亡期间平台把对局打完了

    second = make_runtime()  # 重启：全新内存状态
    summary2 = second.run(duration_sec=5.0)
    assert server["finished"]         # 新进程的 seq=0 全量同步看到了终态
    assert summary2.games_started == 1  # 从 active_games 重新发现
    assert server["actions"] == 1     # 关键：没有重复提交


# ---------- 7.3 时序边界 ----------

def _settled_snapshot() -> dict:
    snap = dict(REAL_DRAW_SNAPSHOT)
    snap["phase"] = "settled"
    return snap


def test_settled_phase_submits_nothing() -> None:
    """局间 settled：快照为残牌，必须零动作提交；若平台此时 409 也只是竞态日志。"""
    steps = iter([
        {"seq": 5, "snapshot": _settled_snapshot()},
        {"seq": 6, "snapshot": _settled_snapshot()},
        {"seq": 7, "finished": True, "snapshot": _settled_snapshot()},
    ])
    submits = {"n": 0}

    def handler(method, path, body):
        if path.startswith("/api/games/g_1/state"):
            return next(steps)
        if path == "/api/games/g_1/action":
            submits["n"] += 1
            return {}
        raise AssertionError(f"未预期请求 {method} {path}")

    runner = _runner(handler)
    assert runner.run() == "finished"
    assert submits["n"] == 0  # settled 窗口一次都没提交


def test_long_poll_pending_resyncs_with_full_snapshot() -> None:
    """跨局无事件：长轮询连续 pending → 每 DEFAULT_LONG_IDLE_RESYNC_ROUNDS 次拉回全量快照。"""
    pending_rounds = {"left": DEFAULT_LONG_IDLE_RESYNC_ROUNDS * 2}
    seqs: list[str] = []
    steps = iter([
        {"seq": 5, "snapshot": REAL_DRAW_SNAPSHOT},
        {"seq": 6, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT},
    ])

    def handler(method, path, body):
        if path.startswith("/api/games/g_1/state"):
            seqs.append(path)
            if pending_rounds["left"] > 0:
                pending_rounds["left"] -= 1
                return {"seq": 5, "pending": True}
            return next(steps)
        if path == "/api/games/g_1/action":
            return {}
        raise AssertionError(f"未预期请求 {method} {path}")

    runner = _runner(handler)
    assert runner.run() == "finished"
    # 首批 pending 用增量 seq=5；到阈值后必须有一次 seq=0 的全量重同步
    assert any("seq=0" in p for p in seqs), f"未见全量重同步: {seqs}"
    assert runner.actions_submitted == 1  # 恢复后仍能决策提交


def test_action_race_409_is_tolerated() -> None:
    """动作冲突竞态：提交被 409 INVALID_ACTION 拒绝 → 记日志、不崩溃、不重试同动作。"""
    submits = {"n": 0}
    steps = iter([
        {"seq": 5, "snapshot": REAL_DRAW_SNAPSHOT},
        {"seq": 6, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT},
    ])

    def handler(method, path, body):
        if path.startswith("/api/games/g_1/state"):
            return next(steps)
        if path == "/api/games/g_1/action":
            submits["n"] += 1
            return ApiError(409, "INVALID_ACTION", "window closed")
        raise AssertionError(f"未预期请求 {method} {path}")

    runner = _runner(handler)
    assert runner.run() == "finished"
    assert submits["n"] == 1          # 被拒后不原地重试
    assert runner.actions_submitted == 0  # 被拒不计入已提交
    assert runner.errors == 0         # 竞态是正常事件，不算错误


def test_concurrent_games_isolated_and_gated(tmp_path) -> None:
    """多场同时进决策窗口：两局各自完成、各自提交一次、并发闸门不超 max_in_flight。"""
    in_flight = {"cur": 0, "peak": 0}
    lock = threading.Lock()
    state = {gid: {"actions": 0, "finished": False} for gid in ("g_1", "g_2")}

    def handler(method, path, body):
        if path == "/api/me":
            active = [{"game_id": g} for g, s in state.items() if not s["finished"]]
            return {"user_id": "u_1", "tournament_id": "t_1",
                    "active_games": active, "active_count": len(active)}
        if path == "/api/tournaments/me/rules":
            return {"config": _config_payload()}
        if path == "/api/tournaments/t_1":
            return {"tournament_id": "t_1", "status": "running",
                    "config": _config_payload(),
                    "my_games": [g for g, s in state.items() if not s["finished"]]}
        for gid in ("g_1", "g_2"):
            if path == f"/api/games/{gid}/action":
                state[gid]["actions"] += 1
                return {}
            if path.startswith(f"/api/games/{gid}/state"):
                if state[gid]["actions"] == 0:
                    return {"seq": 1, "snapshot": REAL_DRAW_SNAPSHOT}
                state[gid]["finished"] = True
                return {"seq": 9, "finished": True, "snapshot": REAL_DRAW_SNAPSHOT}
        raise AssertionError(f"未预期请求 {method} {path}")

    def counting(method, path, body):
        track = "/state" in path or "/action" in path
        if track:
            with lock:
                in_flight["cur"] += 1
                in_flight["peak"] = max(in_flight["peak"], in_flight["cur"])
        try:
            return handler(method, path, body)
        finally:
            if track:
                with lock:
                    in_flight["cur"] -= 1

    runtime = Runtime(
        TOKEN,
        server="http://test",
        decider=FirstLegalDecider(),
        transport=RouterTransport(counting),
        options=RuntimeOptions(poll_interval_sec=0.01, log_dir=str(tmp_path),
                               log_console=False, max_in_flight=1, max_workers=4),
    )
    summary = runtime.run(duration_sec=10.0)
    assert summary.games_started == 2
    assert summary.games_completed == 2
    assert summary.actions_submitted == 2  # 两局各一次，互不串窗口
    assert in_flight["peak"] <= 1          # 闸门全程生效
