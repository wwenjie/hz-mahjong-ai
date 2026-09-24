"""决策接口与兜底（tasks.md 4.9）。

运行时只依赖一个窄接口：给定局面与合法动作，返回一个动作。真正的策略在分组 5 实现
并替换进来；本模块提供两个「一定要有」的组件：

- ``FirstLegalDecider``：可在真实对局中跑通的占位策略，同时也是异常兜底。
- ``GuardedDecider``：把任意决策器包成「绝不抛异常、绝不返回非法动作、超预算即降级」。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from majiang.rules.action import CHI, DISCARD, GANG, HU, PASS, PENG, Action
from majiang.rules.situation import Situation

# 兜底时的动作偏好：能胡就胡，其次杠，再次吃碰，然后出牌，最后才是过
FALLBACK_PREFERENCE: tuple[str, ...] = (HU, GANG, PENG, CHI, DISCARD, PASS)


@runtime_checkable
class Decider(Protocol):
    """决策器：返回 ``None`` 表示本局面不动作。"""

    name: str

    def choose(
        self,
        situation: Situation,
        actions: Sequence[Action],
        *,
        budget_ms: int,
    ) -> Action | None: ...


class FirstLegalDecider:
    """占位策略：按固定优先级取第一个合法动作。"""

    name = "first-legal"

    def choose(
        self,
        situation: Situation,
        actions: Sequence[Action],
        *,
        budget_ms: int,
    ) -> Action | None:
        return first_by_preference(actions)


def first_by_preference(
    actions: Sequence[Action],
    preference: Sequence[str] = FALLBACK_PREFERENCE,
) -> Action | None:
    for kind in preference:
        for action in actions:
            if action.kind == kind:
                return action
    return None


class GuardedDecider:
    """包装：异常、非法返回、超预算都降级到兜底动作。"""

    def __init__(
        self,
        inner: Decider,
        *,
        fallback: Decider | None = None,
        on_fallback: object | None = None,
    ) -> None:
        self._inner = inner
        self._fallback = fallback or FirstLegalDecider()
        self._on_fallback = on_fallback
        self.name = inner.name

    @property
    def last_reason(self) -> str:
        """转发内层决策器的决策理由（若其提供），用于留痕。"""
        return str(getattr(self._inner, "last_reason", ""))

    @property
    def last_detail(self) -> dict[str, object]:
        detail = getattr(self._inner, "last_detail", None)
        return dict(detail) if isinstance(detail, dict) else {}

    def configure(self, tournament: object) -> None:
        """把赛事配置转发给内层决策器（若其支持）。"""
        configure = getattr(self._inner, "configure", None)
        if callable(configure):
            configure(tournament)

    def choose(
        self,
        situation: Situation,
        actions: Sequence[Action],
        *,
        budget_ms: int,
    ) -> Action | None:
        started = time.monotonic()
        try:
            choice = self._inner.choose(situation, actions, budget_ms=budget_ms)
        except Exception as exc:  # noqa: BLE001 —— 决策异常绝不能让对局无响应
            return self._degrade(situation, actions, f"{type(exc).__name__}: {exc}", started)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if choice is None:
            return None
        if elapsed_ms > budget_ms:
            return self._degrade(situation, actions, f"超出预算 {elapsed_ms:.0f}ms", started)
        if choice not in actions:
            return self._degrade(situation, actions, f"返回了非法动作 {choice.describe()}", started)
        return choice

    def _degrade(
        self,
        situation: Situation,
        actions: Sequence[Action],
        reason: str,
        started: float,
    ) -> Action | None:
        if self._on_fallback is not None and callable(self._on_fallback):
            self._on_fallback(reason, (time.monotonic() - started) * 1000.0)
        return self._fallback.choose(situation, actions, budget_ms=0)


__all__ = [
    "Decider",
    "FALLBACK_PREFERENCE",
    "FirstLegalDecider",
    "GuardedDecider",
    "first_by_preference",
]
