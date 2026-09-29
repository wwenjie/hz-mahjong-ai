#!/usr/bin/env python3
"""agent-c · 独立复核 agentb-reviewer 的两条关键数字（不 import 其 work/**）。

复核对象（b-reviewer 2026-09-29 出牌策略审查）：
  D2：`_choose_discard` 排序不稳定 —— 反转候选顺序会改变**44.1%** 的出牌。
  F4：`visible_need` 不含已见张数 —— 乘 (4-seen)/4 后 **24.6%** 出牌改变。

我的做法：**自己重建**判定点（用 `majiang.sim.replay`），直接调 `HeuristicDecider`，
用**同一 situation** 分别喂入「升序」与「反转」候选列表，数改变率。
（D2 只需要排序稳定性 ⇒ 不需要对手信息；F4 用 monkeypatch，仅内存。）

只读；零平台请求。
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

from majiang.strategy import risk as risk_mod
from majiang.sim import replay
from majiang.strategy.policy import HeuristicDecider
from majiang.rules.situation import PHASE_DRAW

OUR = "u_a7f7c67bb14a"
SEATS = 4


def _load_rooms(n: int, seed: int) -> list[str]:
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for p in files:
        try:
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) == 4 and OUR in ids:
            out.append(p)
        if len(out) >= n:
            break
    return out


def check_d2(rooms: list[str]) -> tuple[int, int]:
    """反转候选顺序，数我方出牌决策改变率。"""
    dec = HeuristicDecider()
    changed = 0
    total = 0
    for path in rooms:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        mine = ids.index(OUR)
        for event, state in replay.iter_before_each_event(doc):
            if event.get("type") != replay.DISCARDED or not state.opened:
                continue
            if int(event.get("seat", -1)) != mine:
                continue
            try:
                from majiang.rules.action import legal_actions
                sit = state.situation_for(mine, phase=PHASE_DRAW)
            except Exception:
                continue
            try:
                acts = list(legal_actions(sit))
            except Exception:
                continue
            if len(acts) < 2:
                continue
            a = dec.choose(sit, acts, budget_ms=1800)
            b = dec.choose(sit, list(reversed(acts)), budget_ms=1800)
            if a is None or b is None:
                continue
            total += 1
            if (a.kind, getattr(a, "tile", None)) != (b.kind, getattr(b, "tile", None)):
                changed += 1
    return changed, total


def check_f4(rooms: list[str]) -> tuple[int, int]:
    """visible_need × (4-seen)/4，数出牌改变率（monkeypatch，仅内存）。"""
    from majiang.rules import shanten as sm
    orig = risk_mod.visible_need

    def patched(tile: int) -> float:
        return orig(tile)

    # 需要一个能拿到 seen 的上下文 ⇒ 用分步：先在原口径记录选择，再在改口径记录。
    # 这里用更直接的方式：把 (4-seen)/4 折进 visible_need 需要 seen(tile)，
    # 而 visible_need 签名只有 tile ⇒ 只能借助 situation。故改为「重跑两次 decider」
    # 通过注入全局 seen 表实现。
    seen_tbl = {"seen": None}

    def visible_need_patched(tile: int) -> float:
        base = orig(tile)
        seen = seen_tbl["seen"]
        if seen is None:
            return base
        return base * (4 - min(4, seen[tile])) / 4.0

    dec = HeuristicDecider()
    changed = 0
    total = 0
    risk_mod.visible_need = orig
    for path in rooms:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        mine = ids.index(OUR)
        for event, state in replay.iter_before_each_event(doc):
            if event.get("type") != replay.DISCARDED or not state.opened:
                continue
            if int(event.get("seat", -1)) != mine:
                continue
            try:
                from majiang.rules.action import legal_actions
                sit = state.situation_for(mine, phase=PHASE_DRAW)
                acts = list(legal_actions(sit))
            except Exception:
                continue
            if len(acts) < 2:
                continue
            seen_tbl["seen"] = None
            risk_mod.visible_need = orig
            a = dec.choose(sit, acts, budget_ms=1800)
            try:
                seen = sm.visible_counts(
                    list(sit.hand.counts), [m.tiles for m in sit.all_melds], sit.discards
                )
            except Exception:
                continue
            seen_tbl["seen"] = seen
            risk_mod.visible_need = visible_need_patched
            b = dec.choose(sit, acts, budget_ms=1800)
            risk_mod.visible_need = orig
            if a is None or b is None:
                continue
            total += 1
            if (a.kind, getattr(a, "tile", None)) != (b.kind, getattr(b, "tile", None)):
                changed += 1
    risk_mod.visible_need = orig
    return changed, total


def main() -> int:
    rooms = _load_rooms(40, 20260929)
    print(f"房数 = {len(rooms)}（seed=20260929）")
    print("=" * 70)
    c2, t2 = check_d2(rooms)
    print(f"D2 反转候选顺序改变出牌：{c2}/{t2} = {c2 / t2:.1%}" if t2 else "D2 无样本")
    print(f"    （b-reviewer 报 44.1%）")
    print("-" * 70)
    c4, t4 = check_f4(rooms)
    print(f"F4 seen 因子改变出牌：{c4}/{t4} = {c4 / t4:.1%}" if t4 else "F4 无样本")
    print(f"    （b-reviewer 报 24.6%；A 报约 25%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
