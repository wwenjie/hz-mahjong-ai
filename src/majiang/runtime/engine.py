"""并发对局运行时（tasks.md 分组 4）。

结构：一个 ``Runtime`` 作为监督者，每个进行中的场次交给一个 ``GameRunner`` 线程。
监督者每轮读取赛事状态并据此推进（报名/到位/确认/对局/等待），本身也充当心跳——它
的周期性请求天然满足「最近 90 秒内有已认证请求」的在线判据。

请求预算：``/state`` 长轮询本身很省，但不能「每个事件都拉一次全量快照」。本模块用
两条规则压住请求量：

1. 只有事件可能触发**本人动作**时才拉快照——即事件属于本人，或他人打出的牌在我们
   手上存在碰/明杠/吃的可能。
2. 连续两轮长轮询都没有任何事件时才做一次对账快照，避免长期失步。

并发原语用线程池而非 asyncio：传输层是阻塞式标准库实现，线程池更直接；若将来换成
异步传输再引入事件循环。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from majiang.client.api import PlatformApi, merge_active_games
from majiang.client.errors import ApiError, TransportError, is_permanent, is_race
from majiang.client.models import (
    READY_STATUSES,
    STATUS_FINISHED,
    STATUS_REGISTERING,
    TournamentConfig,
    TournamentState,
)
from majiang.client.snapshot import Snapshot
from majiang.client.transport import HttpTransport, Transport
from majiang.rules import tiles
from majiang.rules.action import DISCARD, PASS, Action, chi_combinations, legal_actions
from majiang.rules.god import NO_DISCARDER
from majiang.rules.situation import (
    PHASE_DRAW,
    PHASE_RESPONSE_CHI,
    PHASE_RESPONSE_PENG,
    PHASE_SETTLED,
)

from .decider import Decider, GuardedDecider
from .logging import Logger, NullLogger, open_logger
from .ratelimit import ConcurrencyGate, RateLimitedTransport, TokenBucket

DEFAULT_RATE_PER_SEC = 16.0
# 从总速率里划给「动作提交」的预留额度。动作有硬窗口（出牌 3 秒、碰/吃 1 秒），
# 轮询没有；预留在预留桶里，提交就不会排在大量在途轮询后面。
DEFAULT_ACTION_RATE_PER_SEC = 3.0
DEFAULT_MAX_IN_FLIGHT = 32
DEFAULT_MAX_WORKERS = 16
DEFAULT_POLL_INTERVAL_SEC = 10.0
DEFAULT_STATE_POLL_TIMEOUT_SEC = 35.0
DEFAULT_SNAPSHOT_TIMEOUT_SEC = 8.0
DEFAULT_LONG_IDLE_RESYNC_ROUNDS = 2
PERMANENT_GAME_CODES = frozenset({"GAME_NOT_FOUND", "FORBIDDEN", "GAME_NOT_FINISHED"})


@dataclass
class RunSummary:
    tournament_id: str = ""
    status: str = ""
    games_started: int = 0
    games_completed: int = 0
    actions_submitted: int = 0
    snapshots: int = 0
    falls_back: int = 0
    errors: int = 0
    ready_calls: int = 0
    register_calls: int = 0
    reopens: int = 0
    stop_reason: str = ""


@dataclass
class PiaoTracker:
    """本地累计「我方飘次数」。

    快照只给 ``chain_count``（飘与杠累计），不给飘次数，而番型计算需要它。这里用
    「爆头态打出白板」这一事件累计，并在链清零时复位。
    """

    piao_count: int = 0
    _baotou_before: bool = False

    def observe_snapshot(self, snapshot: Snapshot) -> None:
        chain = int(snapshot.god.get("chain_count") or 0)
        if chain == 0:
            self.piao_count = 0
        elif self.piao_count > chain:
            self.piao_count = chain
        self._baotou_before = bool(snapshot.god.get("baotou"))

    def observe_action(self, action: Action) -> None:
        if action.kind == DISCARD and action.tile == tiles.GOD and self._baotou_before:
            self.piao_count += 1


class GameRunner:
    """单场对局循环（同步阻塞，由线程池驱动）。"""

    def __init__(
        self,
        api: PlatformApi,
        game_id: str,
        decider: Decider,
        config: TournamentConfig,
        logger: Logger,
        *,
        stop_event: threading.Event | None = None,
        state_poll_timeout: float = DEFAULT_STATE_POLL_TIMEOUT_SEC,
        snapshot_timeout: float = DEFAULT_SNAPSHOT_TIMEOUT_SEC,
        submit_pass: bool = False,
    ) -> None:
        self._api = api
        self.game_id = game_id
        self._decider = decider
        self._config = config
        self._stop = stop_event or threading.Event()
        self._state_poll_timeout = state_poll_timeout
        self._snapshot_timeout = snapshot_timeout
        self._submit_pass = submit_pass
        self._piao = PiaoTracker()
        self._acted_windows: set[tuple[int, str, int | None]] = set()
        self.snapshot: Snapshot | None = None
        self.actions_submitted = 0
        self.snapshots = 0
        self.falls_back = 0
        self.errors = 0
        self.rounds_seen: set[int] = set()
        self.log = logger.child(game_id=game_id)

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def piao_count(self) -> int:
        return self._piao.piao_count

    def run(self) -> str:
        """跑完本场，返回停止原因。"""
        seq = 0
        need_snapshot = True
        idle_polls = 0
        self.log.log("game.start")
        while not self._stop.is_set():
            try:
                envelope = self._fetch(seq, need_snapshot)
            except ApiError as error:
                if self._handle_api_error(error):
                    return f"api:{error.code}"
                continue
            except TransportError as error:
                self.errors += 1
                self.log.log("error.transport", detail=str(error))
                time.sleep(1.0)
                continue

            if envelope.finished:
                self.log.log("game.finished", actions=self.actions_submitted)
                return "finished"
            if envelope.pending:
                idle_polls += 1
                need_snapshot = need_snapshot or idle_polls >= DEFAULT_LONG_IDLE_RESYNC_ROUNDS
                if idle_polls >= DEFAULT_LONG_IDLE_RESYNC_ROUNDS:
                    idle_polls = 0
                continue

            idle_polls = 0
            if envelope.snapshot is not None:
                try:
                    snapshot = Snapshot.parse(envelope.snapshot)
                except Exception as exc:  # noqa: BLE001 —— 单个坏快照不能打死整场对局
                    self.errors += 1
                    self.log.log("error.snapshot", detail=f"{type(exc).__name__}: {exc}")
                    need_snapshot = False
                    time.sleep(0.2)
                    continue
                seq = envelope.seq
                need_snapshot = False
                self.snapshots += 1
                self._absorb(snapshot)
            elif envelope.events:
                seq = max(seq, max(int(e.get("seq", 0) or 0) for e in envelope.events))
                if self._should_fetch_for_events(envelope.events):
                    need_snapshot = True
                continue
            else:
                continue

            action = self._decide_safely(self.snapshot)
            if action is None:
                continue
            self._submit(action)
            need_snapshot = True
        return "stopped"

    def _decide_safely(self, snapshot: Snapshot | None) -> Action | None:
        try:
            return self._decide(snapshot)
        except Exception as exc:  # noqa: BLE001 —— 决策异常不允许打断对局循环
            self.errors += 1
            self.log.log("error.decide", detail=f"{type(exc).__name__}: {exc}")
            return None

    def _fetch(self, seq: int, need_snapshot: bool):
        if need_snapshot:
            return self._api.game_state(self.game_id, 0, timeout=self._snapshot_timeout)
        return self._api.game_state(self.game_id, seq, timeout=self._state_poll_timeout)

    def _handle_api_error(self, error: ApiError) -> bool:
        """返回 True 表示本场应停止。"""
        if is_race(error):
            self.log.log("error.race", code=error.code)
            return False
        if error.code in PERMANENT_GAME_CODES or is_permanent(error):
            self.log.log("error.permanent", code=error.code, message=error.message)
            return True
        self.errors += 1
        self.log.log("error.api", code=error.code, message=error.message)
        time.sleep(0.5)
        return False

    def _absorb(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self._piao.observe_snapshot(snapshot)
        if snapshot.phase == PHASE_SETTLED:
            # 局间停顿：快照里的手牌是上一局残牌，不得用于牌型计算，也不提交动作
            self.log.log("game.settled", round_no=snapshot.round_no)
            return
        if snapshot.round_no not in self.rounds_seen:
            self.rounds_seen.add(snapshot.round_no)
            self.log.log(
                "game.round",
                round_no=snapshot.round_no,
                dealer=snapshot.dealer,
                wall_remaining=snapshot.wall_remaining,
                hand=len(snapshot.my_hand),
            )

    def _should_fetch_for_events(self, events: Sequence[Mapping[str, Any]]) -> bool:
        """事件是否可能触发本人动作。"""
        snapshot = self.snapshot
        if snapshot is None:
            return True
        for event in events:
            if int(event.get("seat", -1)) == snapshot.seat:
                return True
            kind = str(event.get("type", ""))
            if kind == "tile_discarded":
                tile = _optional_tile(event.get("tile"))
                if tile is not None and _could_respond(snapshot, tile):
                    return True
            elif kind not in KNOWN_EVENT_TYPES:
                return True
        return False

    def _decide(self, snapshot: Snapshot | None) -> Action | None:
        if snapshot is None or not snapshot.expects_action:
            return None
        window_key = _window_key(snapshot)
        if window_key is not None and window_key in self._acted_windows:
            return None  # 本响应窗口已提交过，重复提交只会换来 409
        situation = snapshot.to_situation(piao_count=self._piao.piao_count)
        actions = legal_actions(situation)
        if not actions:
            self.log.log("decision.no_action", phase=snapshot.phase)
            return None
        budget_ms = _budget_ms(self._config, snapshot)
        started = time.monotonic()
        choice = self._decider.choose(situation, actions, budget_ms=budget_ms)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.log.log(
            "decision.made",
            phase=snapshot.phase,
            round_no=snapshot.round_no,
            decider=self._decider.name,
            choice=None if choice is None else choice.describe(),
            reason=getattr(self._decider, "last_reason", ""),
            detail=getattr(self._decider, "last_detail", {}) or None,
            candidates=[a.describe() for a in actions],
            elapsed_ms=round(elapsed_ms, 2),
            budget_ms=budget_ms,
            # 仪器：`window_deadline_ms` 是指南里未出现的字段（我们按抓包反推），
            # 因此把原值与本地墙上时钟一并记下，用于事后核对它到底是绝对毫秒还是别的量纲。
            window_deadline_ms=snapshot.window_deadline_ms,
            local_ms=int(time.time() * 1000),
        )
        if choice is not None and window_key is not None:
            self._acted_windows.add(window_key)
        return choice

    def _submit(self, action: Action) -> None:
        if action.kind == PASS and not self._submit_pass:
            # 碰/吃窗口「固定走满」，不提交即为放弃响应，因此交 pass 纯粹是浪费请求
            # （实测：提交的 pass 有相当比例因窗口已关闭而被 409 拒绝）。
            self.log.log("action.skipped", action=action.describe(), reason="pass-is-implicit")
            return
        payload = action.to_payload()
        try:
            self._api.submit_action(self.game_id, payload)
        except ApiError as error:
            if is_race(error):
                # 记下当时的阶段：响应阶段被拒多半是窗口已关（提交太晚），
                # 出牌阶段被拒则更可能是局面已变（我们等了太久或与他人动作撞车）。
                snapshot = getattr(self, "snapshot", None)
                self.log.log(
                    "action.rejected",
                    action=action.describe(),
                    code=error.code,
                    phase=None if snapshot is None else snapshot.phase,
                    round_no=None if snapshot is None else snapshot.round_no,
                )
                return
            self.errors += 1
            self.log.log("error.action", action=action.describe(), code=error.code)
            return
        self.actions_submitted += 1
        self._piao.observe_action(action)
        self.log.log("action.submitted", action=action.describe(), payload=payload)


def _window_key(snapshot: Snapshot) -> tuple[int, str, int | None] | None:
    """响应窗口的唯一标识；非响应阶段返回 None。

    用「局号 + 阶段 + 窗口截止毫秒」作键：平台对窗口「固定走满」，同一窗口内重复提交
    会得到 409，因此在本地记住本窗口已响应即可，无需依赖服务端回执。
    """
    if snapshot.phase not in (PHASE_RESPONSE_PENG, PHASE_RESPONSE_CHI):
        return None
    return (snapshot.round_no, snapshot.phase, snapshot.window_deadline_ms)


def _optional_tile(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return value if isinstance(value, int) else tiles.parse(str(value))


def _could_respond(snapshot: Snapshot, tile: int) -> bool:
    """本地判断我们是否可能对这张弃牌有吃碰明杠反应（保守：宁可多拉一次快照）。"""
    god = snapshot.god
    if bool(god.get("catch_play")) and int(god.get("god_discarder_seat", NO_DISCARDER)) != snapshot.seat:
        return False  # 抓打圈内受限，不能吃碰明杠
    if tile == tiles.GOD:
        return False
    counts = tiles.counts_from(snapshot.my_hand)
    if counts[tile] >= 2:
        return True
    chi_melds = sum(1 for meld in snapshot.my_melds if meld.is_chi)
    if chi_melds < 2 and chi_combinations(counts, tile):
        return True
    return False


def _budget_ms(config: TournamentConfig, snapshot: Snapshot) -> int:
    """决策预算：按指南的权威口径取 ``config`` 里的 ``TimeoutSec`` 的 60%。

    出牌 3 秒、碰/吃 1 秒，且碰/吃窗口「固定走满」（服务端一定等满整段再结算，
    因此不会因为响应得快就提前关窗）。

    **刻意不使用 ``window_deadline_ms``**：该字段在指南里出现 0 次（是我们按抓包反推的）。
    实测把它与本机墙上时钟相减得到的「剩余时间」中位仅 287ms、21% 为负，而窗口实为 1 秒
    —— 两台机器之间有几百毫秒钟差，跨时钟相减不可靠，会算出无意义的 50ms 预算。
    决策实际耗时 1.78–12.17ms，远小于 60% 预算，因此不需要靠剩余时间精打细算。
    """
    base = {
        PHASE_DRAW: config.discard_timeout_sec,
        PHASE_RESPONSE_PENG: config.peng_timeout_sec,
        PHASE_RESPONSE_CHI: config.chi_timeout_sec,
    }.get(snapshot.phase, config.discard_timeout_sec)
    return max(50, int(base * 1000 * 0.6))


KNOWN_EVENT_TYPES = frozenset(
    {
        "tile_drawn",
        "tile_discarded",
        "timeout",
        "chi",
        "peng",
        "gang",
        "hu",
        "round_ended",
        "settled",
    }
)


@dataclass
class RuntimeOptions:
    rate_per_sec: float = DEFAULT_RATE_PER_SEC
    action_rate_per_sec: float = DEFAULT_ACTION_RATE_PER_SEC
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    max_workers: int = DEFAULT_MAX_WORKERS
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
    state_poll_timeout: float = DEFAULT_STATE_POLL_TIMEOUT_SEC
    snapshot_timeout: float = DEFAULT_SNAPSHOT_TIMEOUT_SEC
    log_dir: str = "logs"
    log_console: bool = True
    game_idle_timeout_sec: float = 0.0
    # 碰/吃窗口固定走满，不提交即为放弃响应，因此默认不交 pass——省掉大量必被拒的请求
    submit_pass: bool = False
    # 测试房支持跨轮复用：finished 后再次到位即开启下一轮。正式赛事的 finished 是终态，
    # 因此默认关闭，仅在联调时显式打开。
    reopen_finished_test_room: bool = False
    max_reopens: int = 5
    extra: Mapping[str, Any] = field(default_factory=dict)


class Runtime:
    """多场次并发运行时（单身份）。"""

    def __init__(
        self,
        token: str,
        *,
        server: str,
        decider: Decider,
        transport: Transport | None = None,
        options: RuntimeOptions | None = None,
        logger: Logger | None = None,
        tournament_id: str | None = None,
    ) -> None:
        self._options = options or RuntimeOptions()
        self._token = token
        # 显式指定时跳过「从 /api/me 推断」：自动房不会出现在 /api/me 的 active_games 里
        # （实测 active_games 为空），只能由 POST /api/match 的返回值带入。
        self._tournament_id = tournament_id
        # 总速率切成「轮询主桶 + 动作预留桶」，两桶之和 = rate_per_sec，总量不变。
        # 预留下限保证轮询至少还剩一半，避免把并发度掐死。
        action_rate = min(self._options.action_rate_per_sec, self._options.rate_per_sec / 2)
        bucket = TokenBucket(self._options.rate_per_sec - action_rate)
        priority_bucket = TokenBucket(action_rate)
        gate = ConcurrencyGate(self._options.max_in_flight)
        base = transport or HttpTransport(server)
        self._api = PlatformApi(
            RateLimitedTransport(base, bucket, gate, priority_bucket=priority_bucket), token
        )
        self._guarded = GuardedDecider(decider, on_fallback=self._on_fallback)
        self._stop = threading.Event()
        self._logger = logger or NullLogger()
        self._pool = ThreadPoolExecutor(max_workers=self._options.max_workers)
        self._runners: dict[str, GameRunner] = {}
        self._futures: dict[str, Future[str]] = {}
        self.summary = RunSummary()

    def _on_fallback(self, reason: str, elapsed_ms: float) -> None:
        self.summary.falls_back += 1
        self._logger.log("decision.fallback", reason=reason, elapsed_ms=round(elapsed_ms, 2))

    def request_stop(self) -> None:
        self._stop.set()
        for runner in list(self._runners.values()):
            runner.request_stop()

    def _config_for(self, tournament_id: str) -> TournamentConfig:
        """规则配置。

        自动房必须从赛事状态里取——``/api/tournaments/me/rules`` 是「我的绑定锦标赛」，
        需要报名令牌，而自动房用的是全局令牌，那条路径会 400 ``TOKEN_NOT_SCOPED``。
        """
        if self._tournament_id:
            return self._api.tournament(tournament_id).config
        return self._api.rules()

    def run(self, *, duration_sec: float | None = None) -> RunSummary:
        deadline = None if duration_sec is None else time.monotonic() + duration_sec
        me = self._api.me()
        tournament_id = self._tournament_id or me.tournament_id
        if not tournament_id:
            raise ValueError(
                "令牌未绑定锦标赛：请使用门户派发的参赛令牌，或用 --auto-match 走自动匹配"
            )
        self.summary.tournament_id = tournament_id
        config = self._config_for(tournament_id)
        configure = getattr(self._guarded, "configure", None)
        if callable(configure):
            configure(config)
        logger = open_logger(
            tournament_id, self._options.log_dir, console=self._options.log_console
        ).child(user_id=me.user_id)
        self._logger = logger
        logger.log(
            "runtime.start",
            decider=self._guarded.name,
            server_kind=config.kind,
            max_concurrent_games=config.max_concurrent_games,
            rounds_per_game=config.rounds_per_game,
            base_score=config.base_score,
            you_cai_bi_kao=config.you_cai_bi_kao,
        )
        scoped = me.is_scoped
        last_ready_key = ""
        try:
            while not self._stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    self.summary.stop_reason = "duration"
                    break
                try:
                    state = self._api.tournament(tournament_id)
                except ApiError as error:
                    self.summary.errors += 1
                    logger.log("error.tournament", code=error.code, message=error.message)
                    if is_permanent(error):
                        self.summary.stop_reason = f"api:{error.code}"
                        break
                    self._sleep()
                    continue
                except TransportError as error:
                    self.summary.errors += 1
                    logger.log("error.transport", detail=str(error))
                    self._sleep()
                    continue

                self.summary.status = state.status
                logger.log(
                    "tournament.status",
                    status=state.status,
                    stage=None if state.stage is None else state.stage.name,
                    stage_status=state.stage_status,
                    crashed=state.stage_crashed,
                    qualified=state.qualified,
                    ready_users=state.ready_users,
                    registered_users=state.registered_users,
                    # 战力榜必须**在会话进行中**记录：自动房结束后玩家 API 对该房一律 404，
                    # 事后就取不到了。形状：[[user_id, 总得分, 名次分, 白板数, 已打局数], ...]
                    ranking=[
                        [entry.user_id, entry.total_score, entry.place_points,
                         entry.god_count, entry.games_played]
                        for entry in state.ranking
                    ],
                )
                if state.is_terminal:
                    if (
                        self._options.reopen_finished_test_room
                        and config.is_test_room
                        and state.status == STATUS_FINISHED
                    ):
                        if self.summary.reopens >= self._options.max_reopens:
                            logger.log("tournament.reopen_limit", limit=self._options.max_reopens)
                            self.summary.stop_reason = "reopen-limit"
                            break
                        key = f"reopen:{len(state.my_games)}"
                        if key != last_ready_key:
                            logger.log("tournament.reopen", status=state.status)
                            self._ready(state, tournament_id, logger)
                            self.summary.reopens += 1
                            last_ready_key = key
                        self._sleep()
                        continue
                    self.summary.stop_reason = f"status:{state.status}"
                    break

                if state.status in READY_STATUSES:
                    # 崩溃重赛后确认要重做，因此把 stage_crashed 纳入键，避免误判为已确认
                    key = f"{state.status}:{state.stage_status}:{state.stage_crashed}"
                    # 自动房的入席与到位由 POST /api/match 完成：直连 register/ready 会
                    # 409 AUTO_MATCH_ONLY，重复到位也没有意义，因此整段跳过
                    if not config.is_auto_room and key != last_ready_key:
                        self._ready(state, tournament_id, logger)
                        last_ready_key = key
                    if state.is_eliminated:
                        self.summary.stop_reason = "eliminated"
                        logger.log("tournament.eliminated")
                        break
                    if self.summary.stop_reason == "not_qualified":
                        break
                elif state.status == "running":
                    self._sync_games(state, config, scoped, logger)
                self._reap()
                self._sleep()
        finally:
            self.request_stop()
            self._pool.shutdown(wait=True)
            self._reap()
            logger.log("runtime.stop", reason=self.summary.stop_reason, **self.summary.__dict__)
            logger.close()
        return self.summary

    def _ready(self, state: TournamentState, tournament_id: str, logger: Logger) -> None:
        """报名期先报名再到位；阶段确认期只做到位。

        冲突类错误（已开赛、已关闭、已报名）只记录不视为失败——主循环会按最新状态兜底。
        ``NOT_QUALIFIED`` 表示已淘汰，记入停止原因由主循环退出。
        """
        if state.status == STATUS_REGISTERING:
            try:
                self._api.register(tournament_id)
            except ApiError as error:
                if error.code == "NOT_QUALIFIED":
                    self.summary.stop_reason = "not_qualified"
                logger.log("tournament.register", code=error.code, message=error.message)
            else:
                self.summary.register_calls += 1
                logger.log("tournament.register", status=state.status)
        try:
            self._api.ready(tournament_id)
            self.summary.ready_calls += 1
            logger.log("tournament.ready", status=state.status)
        except ApiError as error:
            if error.code == "NOT_QUALIFIED":
                self.summary.stop_reason = "not_qualified"
            logger.log("error.ready", code=error.code, message=error.message)

    def _sync_games(
        self,
        state: TournamentState,
        config: TournamentConfig,
        scoped: bool,
        logger: Logger,
    ) -> None:
        try:
            me = self._api.me()
        except (ApiError, TransportError) as error:
            self.summary.errors += 1
            logger.log("error.me", detail=str(error))
            return
        active = me.active_games
        if not scoped:
            active = merge_active_games(active, state.my_games)
        for game_id in active:
            if game_id in self._runners:
                continue
            runner = GameRunner(
                self._api,
                game_id,
                self._guarded,
                config,
                logger,
                stop_event=self._stop,
                state_poll_timeout=self._options.state_poll_timeout,
                snapshot_timeout=self._options.snapshot_timeout,
                submit_pass=self._options.submit_pass,
            )
            self._runners[game_id] = runner
            self._futures[game_id] = self._pool.submit(runner.run)
            self.summary.games_started += 1
            logger.log("game.discovered", game_id=game_id, active=len(active))

    def _reap(self) -> None:
        for game_id, future in list(self._futures.items()):
            if not future.done():
                continue
            runner = self._runners.pop(game_id, None)
            self._futures.pop(game_id, None)
            self.summary.games_completed += 1
            if runner is not None:
                self.summary.actions_submitted += runner.actions_submitted
                self.summary.snapshots += runner.snapshots
                self.summary.errors += runner.errors
            reason = "error"
            try:
                reason = future.result()
            except Exception as exc:  # noqa: BLE001
                self.summary.errors += 1
                self._logger.log("error.game", game_id=game_id, detail=str(exc))
            self._logger.log("game.reaped", game_id=game_id, reason=reason)

    def _sleep(self) -> None:
        self._stop.wait(self._options.poll_interval_sec)


__all__ = [
    "GameRunner",
    "PiaoTracker",
    "RunSummary",
    "Runtime",
    "RuntimeOptions",
]
