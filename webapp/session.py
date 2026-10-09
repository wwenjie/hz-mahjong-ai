"""网页对局会话：3×决策器 + 1 名人类（网页输入）。

设计要点
--------
* **引擎复用**：直接调用 ``src/majiang/sim/round.py`` 的 ``run_round`` 与
  ``strategy/versions.py::build("v7")``（默认档，可用 ``--decider`` 换），
  不重写规则、不重写 AI。
* **人类即一个 Decider**：``HumanDecider.choose`` 阻塞游戏线程，直到网页
  ``POST /api/action`` 提交一个合法动作。这样引擎主循环（``play_round``）
  完全无需改动。
* **不泄底**：发给网页的状态只含「人类自己的手牌 + 公共信息」（各座弃牌、
  副露、手牌张数、牌墙余量），**绝不含对手手牌**——与比赛决策器的可见信息
  口径一致。
* **单线程游戏 + 事件总线**：游戏在一个后台线程跑；每次状态变化经 ``EventBus``
  广播给所有 SSE 订阅者。

仅依赖 Python 标准库与仓库内 ``src/``。
"""

from __future__ import annotations

import datetime
import json
import os
import queue
import random
import sys
import threading
import time
from typing import Any

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from majiang.rules import tiles as T  # noqa: E402
from majiang.rules.situation import (  # noqa: E402
    PHASE_DRAW,
    PHASE_RESPONSE_CHI,
    PHASE_RESPONSE_PENG,
    Situation,
)
from majiang.sim import round as R  # noqa: E402
from majiang.sim.batch import next_dealer  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402
from majiang.strategy.versions import build  # noqa: E402

SEATS = 4
DEFAULT_ROUNDS = 8
# 默认决策档：与线上采集器一致（v7 = v5 + 修复「多种吃法」的向听评估）。
DEFAULT_DECIDER = "v7"
# AI 主动出牌前的「思考」停顿（秒），默认 2 秒，让人看清过程。
AI_THINK_SECONDS = 2.0

import os as _os
_DEBUG = bool(_os.environ.get("WEBAPP_DEBUG"))


def _dbg(*a):
    if _DEBUG:
        import sys as _s
        print("[dbg]", *a, file=_s.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 序列化：内部结构 → JSON 视图（严守公开信息边界）
# --------------------------------------------------------------------------- #
def _meld_view(meld: Any) -> dict[str, Any]:
    return {"kind": meld.kind, "tiles": T.to_codes(meld.tiles)}


def _god_view(state: R.RoundState, seat: int) -> dict[str, Any]:
    g = state.god_state(seat)
    return {
        "hand_gods": g.hand_gods,
        "chain_count": g.chain_count,
        "piao_count": g.piao_count,
        "baotou": g.baotou,
        "catch_play": g.catch_play,
        "god_discarder_seat": g.god_discarder_seat,
    }


def others_hands_view(state: R.RoundState) -> list[list[str]]:
    """四家手牌全貌——**仅供「开天眼」娱乐模式**，默认不下发。"""
    return [T.to_codes(sorted(T.tiles_of(ss.hand))) for ss in state.seats]


def state_view(state: R.RoundState, human_seat: int, reveal: bool = False) -> dict[str, Any]:
    """从 ``RoundState`` 抽取**公开信息 + 人类手牌**。

    ``reveal`` 为真（开天眼）时额外附带 ``others_hands``（四家手牌全貌）。
    """
    me = state.seats[human_seat]
    view = {
        "human_seat": human_seat,
        "turn": state.turn,
        "round_no": state.round_no,
        "dealer_seat": state.dealer,
        "wall_remaining": len(state.wall),
        "catch_play": state.catch_play,
        "my_hand": T.to_codes(sorted(T.tiles_of(me.hand))),
        "my_melds": [_meld_view(m) for m in me.melds],
        "god": _god_view(state, human_seat),
        "discards": [T.to_codes(ss.discards) for ss in state.seats],
        "melds": [[_meld_view(m) for m in ss.melds] for ss in state.seats],
        "hand_counts": [T.total_tiles(ss.hand) for ss in state.seats],
    }
    if reveal:
        view["others_hands"] = others_hands_view(state)
    return view


def situation_view(situation: Situation, others_hands: list[list[str]] | None = None) -> dict[str, Any]:
    """从 ``Situation`` 抽取（该座位视角的）公开信息 + 手牌。

    ``others_hands`` 仅在开天眼时由调用方传入。
    """
    view = {
        "seat": situation.seat,
        "phase": situation.phase,
        "turn": situation.turn,
        "my_hand": T.to_codes(sorted(T.tiles_of(situation.hand.counts))),
        "my_melds": [_meld_view(m) for m in situation.hand.melds],
        "god": {
            "hand_gods": situation.god.hand_gods,
            "chain_count": situation.god.chain_count,
            "piao_count": situation.god.piao_count,
            "baotou": situation.god.baotou,
            "catch_play": situation.god.catch_play,
            "god_discarder_seat": situation.god.god_discarder_seat,
        },
        "wall_remaining": situation.table.wall_remaining,
        "dealer_seat": situation.table.dealer_seat,
        "round_no": situation.table.round_no,
        "scores": list(situation.table.scores) if situation.table.scores else None,
        "rounds_total": situation.table.rounds_total,
        "discards": [T.to_codes(d) for d in situation.discards],
        "melds": [[_meld_view(m) for m in group] for group in situation.melds],
        "hand_counts": list(situation.hand_counts),
        "offered_tile": None if situation.offered_tile is None else T.to_code(situation.offered_tile),
        "drawn_tile": None if situation.drawn_tile is None else T.to_code(situation.drawn_tile),
        "responding_seats": list(situation.responding_seats),
        "is_restricted": situation.is_restricted,
    }
    if others_hands is not None:
        view["others_hands"] = others_hands
    return view


def action_view(action: Any) -> dict[str, Any]:
    return {
        "kind": action.kind,
        "tile": None if action.tile is None else T.to_code(action.tile),
        "tiles": list(T.to_codes(action.tiles)),
        "gang_kind": action.gang_kind,
    }


def _action_key(action: Any) -> tuple:
    return (action.kind, action.tile, tuple(sorted(action.tiles)), action.gang_kind)


# --------------------------------------------------------------------------- #
# 事件总线
# --------------------------------------------------------------------------- #
class EventBus:
    """极简发布/订阅：每个 SSE 连接一个队列，新订阅者立即收到最近一条事件。"""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.last: dict[str, Any] | None = None

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subs.append(q)
        if self.last is not None:
            try:
                q.put_nowait(self.last)
            except queue.Full:
                pass
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict[str, Any]) -> None:
        self.last = event
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class HumanActionError(Exception):
    """网页提交了非法动作。"""


class _MatchCancelled(Exception):
    """本场已被新一代对局取代，旧线程立即退出。"""


def _sleep_stoppable(seconds: float, stop: threading.Event) -> bool:
    """可被 stop 立即打断的睡眠。返回 True 表示被要求停止。"""
    if seconds <= 0:
        return stop.is_set()
    return stop.wait(seconds)


class PacedDecider:
    """给 AI 决策器加一个「思考停顿」，让人能看清每一步。

    * **仅在主动出牌（``PHASE_DRAW``）时延迟**：响应吃/碰/杠不延迟，
      否则人类每打一张牌，多家各等 2 秒会拖得很慢。
    * ``choose`` 最终仍由内层决策器给出动作，**不改变 AI 策略本身**。
    * 停顿可被 ``stop_event`` 即时打断（新开一场时不再白等）。
    """

    def __init__(self, inner: Any, delay_s: float, stop_event: threading.Event) -> None:
        self._inner = inner
        self._delay_s = float(delay_s)
        self._stop = stop_event
        self.name = getattr(inner, "name", "ai")

    def choose(self, situation: Situation, actions, *, budget_ms: int = 0):
        if situation.phase == PHASE_DRAW and self._delay_s > 0:
            if _sleep_stoppable(self._delay_s, self._stop):
                raise _MatchCancelled()
        if self._stop.is_set():
            raise _MatchCancelled()
        return self._inner.choose(situation, actions, budget_ms=budget_ms)


# --------------------------------------------------------------------------- #
# 人类决策器：阻塞等待网页输入
# --------------------------------------------------------------------------- #
class HumanDecider:
    """把「人类」表示为引擎可调用的 Decider。

    ``choose`` 在游戏线程内阻塞，直到 HTTP 线程通过 :meth:`GameSession.submit`
    投入一个合法动作。为此必须先 ``publish`` 一条 ``action_required`` 事件，
    网页端据此渲染可点选项。
    """

    def __init__(self, session: GameSession, epoch: int) -> None:
        self.session = session
        self.epoch = epoch
        self.name = "human"

    def choose(self, situation: Situation, actions, budget_ms: int | None = None):
        return self.session.await_human(situation, actions, self.epoch)


# --------------------------------------------------------------------------- #
# 对局会话
# --------------------------------------------------------------------------- #
class GameSession:
    def __init__(
        self,
        *,
        human_seat: int = 0,
        rounds: int = DEFAULT_ROUNDS,
        base_score: int = 1,
        seed: int | None = None,
        mode: str = "qualifier",
        decider: str = DEFAULT_DECIDER,
        ai_delay: float = AI_THINK_SECONDS,
    ) -> None:
        self.human_seat = human_seat
        self.rounds = rounds
        self.base_score = base_score
        self.seed = seed
        self.mode_str = mode
        self.decider = str(decider)
        self.mode = Mode.QUALIFIER if mode == "qualifier" else Mode.FINAL
        # AI 主动出牌的「思考停顿」秒数（仅 PHASE_DRAW，响应阶段不延迟）。
        self.ai_delay = float(ai_delay)

        self.bus = EventBus()
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._pending_allowed: set[tuple] | None = None
        self._pending_result: tuple[int, Any] | None = None
        self._thread: threading.Thread | None = None
        self._epoch = 0
        self._stop = threading.Event()
        self._cumulative = [0] * SEATS
        self.rounds_played: list[dict[str, Any]] = []
        self.finished = False
        self.current_state: dict[str, Any] | None = None
        self.status = "idle"
        # 开天眼（娱乐模式）：为真时下发四家手牌全貌。
        self.reveal = False
        self._last_state: R.RoundState | None = None
        # 最近一次「等待人类决策」的事件：刷新/重连后借此恢复决策按钮。
        self._last_required: dict[str, Any] | None = None
        # 建议：人类决策点上，用决策器算「推荐动作」，仅供人参考/观察。
        # 关闭时（默认开）不产生任何额外计算。
        self.suggest_enabled = True
        self._suggest_decider: Any = None
        self._suggest_lock = threading.Lock()
        self._last_suggestion: dict[str, Any] | None = None
        self._decision_seq = 0
        # 「手动开下一局」闸门：每局结束后不自动续局，等网页点「开下一局」。
        self._round_gate = threading.Condition()
        self._advance_seq = 0        # 每请求一次下一局 +1；单调递增，供等待方比对
        self.awaiting_next = False   # 当前是否停在「上局已结束、等开下一局」

    def set_reveal(self, reveal: bool) -> None:
        """切换开天眼；切换后立即重播一次当前局面。"""
        self.reveal = bool(reveal)
        if self._last_state is not None:
            self._observe(self._last_state)

    def set_suggest(self, enabled: bool) -> None:
        """开关「建议」。开启时，人类决策点会用决策器算出推荐动作。"""
        self.suggest_enabled = bool(enabled)
        # 重播当前决策点，让刚开启的建议立刻生效（无需等到下一个决策点）。
        if self._last_required is not None:
            self.bus.publish(self._last_required)

    # -- 生命周期 --------------------------------------------------------- #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._begin()

    def _begin(self) -> None:
        """清空并启动一条新的对局线程；世代号 self._epoch 递增。"""
        self.finished = False
        self.status = "running"
        self.rounds_played = []
        self._cumulative = [0] * SEATS
        self.current_state = None
        self._last_state = None
        self._pending_allowed = None
        self._pending_result = None
        self._last_required = None
        self.awaiting_next = False
        self._stop = threading.Event()  # 新线程用新事件
        self._epoch += 1
        epoch = self._epoch
        self._thread = threading.Thread(
            target=self._run_match, args=(epoch,), daemon=True
        )
        self._thread.start()

    def reset(self, **kwargs) -> None:
        """原地重开一场：停掉旧线程，**保留同一事件总线**（SSE 连接不中断）。

        关键：不新建 Session 对象。旧实现里 ``/api/new`` 新建整个对象，
        浏览器 SSE 仍订阅旧总线，新对局的事件永远推不到页面（“新开一场没反应”）。
        """
        _dbg(f"reset 开始 epoch={self._epoch} kwargs={kwargs}")
        if kwargs.get("rounds"):
            self.rounds = int(kwargs["rounds"])
        if kwargs.get("seed") is not None:
            self.seed = kwargs["seed"]
        if kwargs.get("human_seat") is not None:
            self.human_seat = int(kwargs["human_seat"])
        if kwargs.get("mode"):
            self.mode_str = str(kwargs["mode"])
            self.mode = Mode.QUALIFIER if self.mode_str == "qualifier" else Mode.FINAL
        if kwargs.get("decider"):
            self.decider = str(kwargs["decider"])
        # 让旧线程立即失效：置停止事件（打断 AI 思考停顿）并唤醒可能阻塞在等人类决策的它
        self._stop.set()
        self._epoch += 1
        with self._cond:
            self._cond.notify_all()
        # 若旧线程正停在「等开下一局」，也要唤醒它，让它看到 epoch 变化后退出。
        with self._round_gate:
            self._round_gate.notify_all()
        old = self._thread
        if old is not None and old.is_alive():
            old.join(timeout=2.0)
        # 先广播 new_game，再启动新线程：否则新线程可能先发 action_required，
        # 紧接着 new_game 把它清掉，页面就丢掉新对局的第一个决策点。
        self.bus.publish(
            {"type": "new_game", "rounds": self.rounds,
             "human_seat": self.human_seat, "reveal": self.reveal}
        )
        self._begin()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def next_round(self) -> tuple[bool, str | None]:
        """网页点「开下一局」：放行主循环继续打下一局。

        幂等：若当前并不在等（没停每局之间），返回提示而不报错。
        """
        with self._round_gate:
            if not self.awaiting_next:
                return False, "当前不在「等开下一局」状态"
            self.awaiting_next = False
            self._advance_seq += 1   # 单调递增：仅放行当次等待，不会误放行未来某次
            self._round_gate.notify_all()
        _dbg(f"next_round 放行 seq={self._advance_seq} epoch={self._epoch}")
        return True, None

    # -- 报障快照 --------------------------------------------------------- #
    def _seat_full(self, seat: Any) -> dict[str, Any]:
        return {
            "hand": list(seat.hand),
            "melds": [_meld_view(m) for m in seat.melds],
            "discards": list(seat.discards),
            "chain_count": seat.chain_count,
            "piao_count": seat.piao_count,
            "god_count": seat.god_count,
        }

    def _state_full(self) -> dict[str, Any] | None:
        """全量内部局面（含**有序牌墙**与四家暗手）——用于离线精确复现。

        牌墙就是「摸牌顺序」本身：只要它有序，未来每一张摸牌都唯一确定 ⇒
        连同四家手牌/副露/弃牌/庄家/轮次，可把本局**逐位重放到底**，
        不需要用户在网页上保持场景。
        """
        st = self._last_state
        if st is None:
            return None
        return {
            "round_no": st.round_no,
            "dealer": st.dealer,
            "turn": st.turn,
            "catch_play": st.catch_play,
            "god_discarder": st.god_discarder,
            "prior_scores": list(st.prior_scores),
            "rounds_total": st.rounds_total,
            "wall_len": len(st.wall),
            "wall": list(st.wall),
            "seats": [self._seat_full(ss) for ss in st.seats],
        }

    def snapshot_report(self, comment: str = "") -> dict[str, Any]:
        """保存一份「报障快照」：当时局面 + 引擎建议 + 用户备注，落盘 JSON。

        落盘目录 ``webapp/reports/``；返回快照 dict（含 ``id`` 与 ``path``）。
        **绝不抛异常**：报障本身不能影响对局，出错也要返回一个可读结果。
        """
        comment = str(comment or "")[:2000]
        sug = self._last_suggestion
        req = self._last_required
        now = datetime.datetime.now()
        rid = f"report_{now.strftime('%Y%m%d_%H%M%S')}_seq{self._decision_seq}"
        report: dict[str, Any] = {
            "schema": "majiang.webapp.report.v1",
            "id": rid,
            "created_at": now.isoformat(timespec="seconds"),
            "comment": comment,
            "session": {
                "human_seat": self.human_seat,
                "rounds": self.rounds,
                "mode": self.mode_str,
                "decider": self.decider,
                "seed": self.seed,
                "status": self.status,
                "suggest_enabled": self.suggest_enabled,
                "reveal": self.reveal,
                "ai_delay": self.ai_delay,
            },
            "decision": {
                "decision_seq": self._decision_seq,
                "phase": (req or {}).get("phase") or (self.current_state or {}).get("phase"),
                "actions": (req or {}).get("actions"),
                "situation": (req or {}).get("view"),   # 决策点局面视图（重放用）
                "suggestion": dict(sug) if isinstance(sug, dict) else sug,
            },
            "public_state": self.current_state,   # 人类视角（公开信息 + 自己手牌）
            "full_state": self._state_full(),      # 全量（含牌墙顺序，供精确复现）
            "recent_rounds": list(self.rounds_played)[-4:],
        }
        path = ""
        try:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, rid + ".json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            _dbg(f"报障落盘失败: {type(exc).__name__}: {exc}")
        report["path"] = path
        _dbg(f"报障已保存 id={rid} path={path}")
        return report

    # -- 引擎回调 --------------------------------------------------------- #
    def _observe(self, state: R.RoundState, epoch: int | None = None) -> None:
        if epoch is not None and epoch != self._epoch:
            _dbg(f"observe 陈旧抑制 epoch={epoch} cur={self._epoch}")
            return
        self._last_state = state
        view = state_view(state, self.human_seat, reveal=self.reveal)
        self.current_state = view
        self.bus.publish({"type": "state", "view": view, "required": self._last_required})

    def await_human(
        self, situation: Situation, actions, epoch: int | None = None
    ):
        _dbg(f"await_human 进入 epoch={epoch} cur={self._epoch} phase={situation.phase} "
             f"turn={situation.turn} seat={situation.seat} 动作数={len(actions)}")
        if epoch is not None and epoch != self._epoch:
            raise _MatchCancelled()
        hands = (
            others_hands_view(self._last_state)
            if (self.reveal and self._last_state is not None)
            else None
        )
        view = situation_view(situation, others_hands=hands)
        act_views = [action_view(a) for a in actions]
        allowed = {_action_key(a) for a in actions}
        with self._lock:
            self._pending_allowed = allowed
        self.current_state = {**(self.current_state or {}), "phase": situation.phase}
        # 每个决策点一个递增序号：建议带同序号标签，前端只采纳「与当前决策点同序号」的建议，
        # 避免慢计算（决策器 p99 可达数百 ms）在新决策点到达后才完成、被误当当前建议。
        self._decision_seq += 1
        seq = self._decision_seq
        required_event = {
            "type": "action_required",
            "view": view,
            "actions": act_views,
            "phase": situation.phase,
            "seq": seq,
        }
        self._last_required = required_event
        with self._suggest_lock:
            self._last_suggestion = None
        if epoch is None or epoch == self._epoch:
            self.bus.publish(required_event)
        # 建议：在后台线程用决策器算推荐动作，**绝不阻塞对局主循环**。
        if self.suggest_enabled and actions and (epoch is None or epoch == self._epoch):
            threading.Thread(
                target=self._compute_suggestion,
                args=(situation, tuple(actions), epoch, seq),
                daemon=True,
            ).start()
        with self._cond:
            while True:
                if epoch is not None and epoch != self._epoch:
                    with self._lock:
                        self._pending_allowed = None
                    raise _MatchCancelled()
                pr = self._pending_result
                if pr is not None and (epoch is None or pr[0] == epoch):
                    self._pending_result = None
                    result = pr[1]
                    break
                self._cond.wait(timeout=0.5)
        with self._lock:
            self._pending_allowed = None
        self._last_required = None
        _dbg(f"await_human 返回 epoch={epoch} 动作={result.describe()}")
        return result

    def _get_suggest_decider(self):
        """惰性构造一个决策器专用于「建议」（与 AI 的实例分开，避免互相污染）。"""
        with self._suggest_lock:
            tag = (self.decider, self.mode)
            if self._suggest_decider is None or getattr(self, "_suggest_tag", None) != tag:
                self._suggest_decider = build(self.decider, self.mode)
                self._suggest_tag = tag
            return self._suggest_decider

    def _compute_suggestion(self, situation: Situation, actions, epoch: int | None, seq: int) -> None:
        """后台线程：用决策器给当前决策点算一个推荐动作，publish 出去供网页显示。

        带 ``seq``：若计算期间人类已进入下一个决策点，本次结果作废（不 publish），
        否则会把「上一个决策点的建议」误当成本决策点的建议。
        """
        started = time.monotonic()
        try:
            decider = self._get_suggest_decider()
            chosen = decider.choose(situation, actions, budget_ms=3000)
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
            if epoch is not None and epoch != self._epoch:
                return  # 本场已被新一场取代，丢弃
            if seq != self._decision_seq:
                return  # 人类已进入下一个决策点，本次建议过期，丢弃
            rec = action_view(chosen) if chosen is not None else None
            reason = str(getattr(decider, "last_reason", "") or "")
            suggestion = {
                "type": "suggestion",
                "phase": situation.phase,
                "seq": seq,
                "action": rec,
                "reason": reason,
                "elapsed_ms": elapsed_ms,
            }
            with self._suggest_lock:
                self._last_suggestion = suggestion
            self.bus.publish(suggestion)
            _dbg(f"建议计算完成 seq={seq} phase={situation.phase} 动作={rec} 耗时={elapsed_ms}ms")
        except Exception as exc:  # noqa: BLE001 —— 建议失败绝不影响对局
            _dbg(f"建议计算失败: {type(exc).__name__}: {exc}")

    def submit(self, payload: dict[str, Any]) -> tuple[bool, str | None]:
        """HTTP 线程调用：校验并投递人类动作。

        关键：校验通过后**立即**清空 ``_pending_allowed``，使同一决策点的
        重复提交（双击、网络重试、快速脚本）被拒，而不是覆盖 ``_pending_result``
        造成状态错乱。
        """
        from majiang.rules.action import Action

        try:
            action = _parse_action(payload)
        except Exception as exc:  # noqa: BLE001
            return False, f"动作解析失败: {exc}"
        with self._lock:
            allowed = self._pending_allowed
            if allowed is None:
                return False, "当前没有等待中的决策点"
            if _action_key(action) not in allowed:
                return False, f"非法动作: {action.describe()}"
            self._pending_allowed = None  # 立即消费：重复提交将被拒
        with self._cond:
            self._pending_result = (self._epoch, action)
            self._cond.notify_all()
        _dbg(f"submit 接受 epoch={self._epoch} 动作={action.describe()}")
        return True, None

    # -- 主循环 ----------------------------------------------------------- #
    def _run_match(self, epoch: int) -> None:
        stop = self._stop
        try:
            deciders: list[Any] = [
                PacedDecider(build(self.decider, self.mode), self.ai_delay, stop)
                for _ in range(SEATS)
            ]
            deciders[self.human_seat] = HumanDecider(self, epoch)
            rng = random.Random(self.seed)
            dealer = 0
            for index in range(self.rounds):
                if epoch != self._epoch:
                    return
                outcome = R.run_round(
                    deciders,
                    dealer=dealer,
                    round_no=index + 1,
                    base_score=self.base_score,
                    rng=rng,
                    observer=lambda st: self._observe(st, epoch),
                    prior_scores=tuple(self._cumulative),
                    rounds_total=self.rounds,
                )
                if epoch != self._epoch:
                    return
                for seat in range(SEATS):
                    self._cumulative[seat] += outcome.scores[seat]
                rec = {
                    "round_no": outcome.round_no,
                    "dealer": outcome.dealer,
                    "winner": outcome.winner,
                    "fan": outcome.fan,
                    "detail": list(outcome.detail),
                    "scores": list(outcome.scores),
                    "cumulative": list(self._cumulative),
                    "is_flow": outcome.is_flow,
                    "turns": outcome.turns,
                }
                self.rounds_played.append(rec)
                is_last = index == self.rounds - 1
                # 非最后一局：广播结果后**停下等人类点「开下一局」**，不自动续局。
                self.awaiting_next = not is_last
                self.bus.publish({
                    "type": "round_end",
                    "result": rec,
                    "cumulative": list(self._cumulative),
                    "awaiting_next": self.awaiting_next,
                    "next_round_no": None if is_last else index + 2,
                })
                dealer = next_dealer(dealer, outcome)
                if not is_last:
                    # 等 /api/next_round（或新一场使 epoch 失效）。0.5s 轮询兜底，
                    # 即使错过 notify，也会在 0.5s 内因 epoch 变化而退出，绝不死锁。
                    with self._round_gate:
                        marker = self._advance_seq
                        while self._advance_seq == marker:
                            if epoch != self._epoch:
                                return
                            self._round_gate.wait(timeout=0.5)
                    if epoch != self._epoch:
                        return
                    self.awaiting_next = False
            self.status = "finished"
            self.finished = True
            self.bus.publish({"type": "match_end", "rounds": self.rounds_played,
                              "cumulative": list(self._cumulative)})
        except _MatchCancelled:
            # 本场被新一代取代：静默退出（不报错、不改状态）。
            return
        except Exception as exc:  # noqa: BLE001
            self.status = f"error: {exc}"
            self.bus.publish({"type": "error", "message": str(exc)})


def _parse_action(payload: dict[str, Any]):
    from majiang.rules.action import Action

    kind = payload.get("kind")
    tile = payload.get("tile")
    tile_idx = None if tile in (None, "") else T.parse(tile)
    tiles = tuple(T.parse(c) for c in (payload.get("tiles") or []))
    return Action(kind=kind, tile=tile_idx, tiles=tiles, gang_kind=payload.get("gang_kind"))


# --------------------------------------------------------------------------- #
# 自检：脚本化「人类」跑完一场，验证线程与阻塞逻辑
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sess = GameSession(rounds=2, seed=7)
    sess.start()

    deadline = time.time() + 60
    answered = 0
    while time.time() < deadline:
        st = sess.current_state
        if st and not sess.finished and sess._pending_allowed is not None:
            # 随便挑一个合法动作：优先 pass，否则第一个
            with sess._lock:
                allowed = sorted(sess._pending_allowed, key=lambda k: (k[0] != "pass",))
            kind, tile, tiles, gang = allowed[0]
            payload = {
                "kind": kind,
                "tile": None if tile is None else T.to_code(tile),
                "tiles": list(T.to_codes(tiles)),
                "gang_kind": gang,
            }
            ok, err = sess.submit(payload)
            if ok:
                answered += 1
            else:
                time.sleep(0.05)
        if sess.finished:
            break
        time.sleep(0.05)

    print(f"status={sess.status} 人类决策={answered} 局数={len(sess.rounds_played)}")
    for r in sess.rounds_played:
        print(" ", r["round_no"], "winner=", r["winner"], "fan=", r["fan"],
              "scores=", r["scores"], "cum=", r["cumulative"])
