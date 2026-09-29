#!/usr/bin/env python3
"""agent-c · 超时余量仪器（A 19:05 / B' 18:40 请求；用户 18:58 授权窗口内自主推进）。

**问题**：A/B' 发现「超时」在服务端不是「丢窗口」而是「能胡就自动胡、否则打最右」。
若我们的贵模型超预算降级到 `FirstLegalDecider`，可能**比服务端自动胡更差**。
所以在接贵模型前，必须量化「我们离超时还有多远」，以及**真机上到底发生过几次超时**。

**做法**（只读；零平台请求）：
  1. 从 `logs/*.jsonl` 取全部 `decision.made`，按 phase 统计 `elapsed_ms / budget_ms`；
  2. 从 `data/auto_sessions/*/events/*.json` 取我们座位的服务端 `timeout` 事件（按 kind 分桶）；
  3. 交叉：我们**自己**判定的超预算决策数（`elapsed_ms > budget_ms`）vs 服务端实际 timeout 数。

用法::

    nice -n 19 uv run python agent/verify/timeout_margin_probe.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

OUR = "u_a7f7c67bb14a"
SEATS = 4


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))
    return ys[k]


def scan_decisions() -> tuple[dict, Counter, int]:
    by_phase: dict[str, list[tuple[float, float]]] = defaultdict(list)
    deciders = Counter()
    n = 0
    for path in glob.glob("logs/*.jsonl"):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if "\"decision.made\"" not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("user_id") != OUR:
                        continue
                    ph = d.get("phase") or "?"
                    el = d.get("elapsed_ms")
                    bu = d.get("budget_ms")
                    if el is None or bu is None:
                        continue
                    by_phase[ph].append((float(el), float(bu)))
                    deciders[d.get("decider") or "?"] += 1
                    n += 1
        except Exception:
            continue
    return by_phase, deciders, n


def scan_server_timeouts() -> tuple[Counter, Counter, int]:
    """我们座位的服务端 timeout 事件，按 kind 分桶。"""
    by_kind = Counter()
    by_phase_guess = Counter()
    rooms = 0
    for path in glob.glob("data/auto_sessions/*/events/*.json"):
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != SEATS or OUR not in ids:
            continue
        rooms += 1
        mine = ids.index(OUR)
        for blk in doc.get("blocks") or []:
            for e in (blk.get("events") or []):
                if e.get("type") != "timeout":
                    continue
                if int(e.get("seat", -1)) != mine:
                    continue
                kind = ((e.get("data") or {}).get("kind")) or "?"
                by_kind[kind] += 1
                win = ((e.get("data") or {}).get("window")) or ""
                by_phase_guess[f"{kind}{'/' + win if win else ''}"] += 1
    return by_kind, by_phase_guess, rooms


def main() -> int:
    print("=" * 88)
    print("超时余量仪器（只读；零平台请求）")
    print("=" * 88)
    by_phase, deciders, n = scan_decisions()
    print(f"decision.made 总数（我方）= {n}")
    print(f"decider 分布 = {dict(deciders)}")
    print()
    print(f"{'phase':<16}{'n':>8}{'budget':>9}{'p50':>9}{'p99':>9}{'max':>10}"
          f"{'超预算':>8}{'p99/budget':>12}{'余量(max)':>11}")
    print("-" * 88)
    over_total = 0
    for ph in sorted(by_phase, key=lambda k: -len(by_phase[k])):
        rows = by_phase[ph]
        els = [e for e, _ in rows]
        bus = [b for _, b in rows]
        budget = bus[0]
        over = sum(1 for e, b in rows if e > b)
        over_total += over
        p99 = pct(els, 0.99)
        print(f"{ph:<16}{len(els):>8}{budget:>9.0f}{pct(els,0.5):>9.2f}{p99:>9.2f}"
              f"{max(els):>10.2f}{over:>8}{p99 / budget:>12.1%}{budget - max(els):>11.1f}")
    print("-" * 88)
    print(f"我方判定超预算（elapsed_ms > budget_ms）总数 = {over_total}")
    print()
    by_kind, by_phase_guess, rooms = scan_server_timeouts()
    print(f"服务端 timeout 事件（我方座位，房数={rooms}）：")
    print(f"  按 kind = {dict(by_kind)}")
    print(f"  按 kind/window = {dict(by_phase_guess)}")
    print()
    print("判读：")
    print("  · p99/budget ≪ 100% ⇒ 余量充足，贵模型有空间（A 19:05：出牌约 5 倍余量）。")
    print("  · 我方判定超预算数若 > 0 且服务端 timeout 数 ≈ 0 ⇒ 我们的自降级阈值比服务端更紧，"
          "降级过度保守（可能白白丢掉好决策）。")
    print("  · 服务端 timeout 数 > 0 ⇒ 已发生过真实超时，需看对应决策是否被降级。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
