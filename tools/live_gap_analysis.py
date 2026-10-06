#!/usr/bin/env python3
"""线上对局差距体检（B' 2026-10-06 14:37 固化，用户指令）。

回答：「我们 vs 强 bot 输在哪」——每天/每次版本变更后跑一遍出体检报告。

数据源：
    data/auto_sessions/sessions.jsonl           对局元数据
    data/auto_sessions/<room>/events/*.json     逐事件日志

口径：
- 敌我识别：座位名含「玄武」→ 我方；其余 → 对手（默认全对手，--strong-only 只留强 bot）
- 强 bot 名单：--strong-names（默认从 seats 名字启发式：凤凰/麻将科学/朱雀/白虎 等非「玄武」常客，
  可按赛季成绩单修正）
- 指标（全部过 tools/ci_gate.py 的守门规则：n≥30 且 95%CI 不含 0 才报方向）：
  胡率 / 均分 / 均番(胡时) / 爆头率 / fan≥4 率 / 财飘率 / 胡牌巡目(近似=该座位 discard 数)
- 按天分桶，输出趋势表

用法：
    uv run python tools/live_gap_analysis.py [--hours 48] [--out agent/out/live-gap-report.txt]
    uv run python tools/live_gap_analysis.py --strong-only   # 只对强 bot 对照组
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_gate import compare_two_groups  # noqa: E402

OUR_NAMES = ("玄武",)  # 我方 bot 座位名关键字
STRONG_DEFAULT = ("凤凰", "麻将科学", "朱雀", "白虎", "青龙")  # 强 bot 候选（非我方的常客）
GOD = "白"


def prop_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return p, p - 1.96 * se, p + 1.96 * se


def prop_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float, bool]:
    """两比例差 + 95%CI + 是否显著。n<30 判不足。"""
    if n1 < 30 or n2 < 30:
        return 0.0, 0.0, 0.0, False
    p1 = k1 / n1
    p2 = k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return d, lo, hi, not (lo <= 0 <= hi)


class SideStats:
    def __init__(self) -> None:
        self.scores: list[float] = []
        self.wins = 0
        self.fans: list[float] = []
        self.win_turns: list[float] = []
        self.baotou = 0      # detail 含「爆头」
        self.fan_ge4 = 0     # fan >= 4
        self.caipiao = 0     # detail 含「财飘」
        self.god_discards = 0  # 打出财神次数（巡段级）
        self.god_rounds = 0    # 摸到过财神的巡段数


def analyze_file(path: str, strong_only: bool, strong_names: tuple[str, ...]) -> tuple[SideStats, SideStats] | None:
    try:
        doc = json.load(open(path))
    except Exception:
        return None
    seats = doc.get("seats", [])
    if len(seats) != 4:
        return None
    our_seats = {i for i, s in enumerate(seats) if any(k in (s.get("name") or "") for k in OUR_NAMES)}
    if not our_seats:
        return None
    if strong_only:
        opp_seats = {i for i, s in enumerate(seats)
                     if i not in our_seats and any(k in (s.get("name") or "") for k in strong_names)}
        if not opp_seats:
            return None  # 本房没有强 bot，跳过
    else:
        opp_seats = {i for i in range(4)} - our_seats

    ours, opps = SideStats(), SideStats()
    for blk in doc.get("blocks", []):
        disc_count: Counter = Counter()
        # 财神持有追踪（巡段级）
        start_hands = blk.get("start_hands") or []
        has_god = {s: bool(h and GOD in h) for s, h in enumerate(start_hands)}
        drawn_god = {s: False for s in range(4)}
        discarded_god = {s: False for s in range(4)}
        for ev in blk.get("events", []):
            t = ev.get("type")
            seat = ev.get("seat")
            tile = ev.get("tile")
            if t == "tile_drawn" and tile == GOD and seat is not None:
                drawn_god[seat] = True
            elif t == "tile_discarded" and seat is not None:
                if tile == GOD:
                    discarded_god[seat] = True
                disc_count[seat] += 1
            elif t == "round_ended":
                d = ev.get("data") or {}
                scores = d.get("scores")
                if not scores:
                    continue
                winner = ev.get("seat")
                fan = d.get("fan") or 0
                details = [str(x) for x in (d.get("detail") or [])]
                for s in range(4):
                    if s in our_seats:
                        st = ours
                    elif s in opp_seats:
                        st = opps
                    else:
                        continue
                    st.scores.append(float(scores[s]))
                    if s == winner and not d.get("draw"):
                        st.wins += 1
                        st.fans.append(float(fan))
                        st.win_turns.append(float(disc_count[s]))
                        if "爆头" in details:
                            st.baotou += 1
                        if fan >= 4:
                            st.fan_ge4 += 1
                        if "财飘" in details:
                            st.caipiao += 1
        # 财神：巡段结算
        for s in range(4):
            if has_god.get(s) or drawn_god[s]:
                if s in our_seats:
                    ours.god_rounds += 1
                    if discarded_god[s]:
                        ours.god_discards += 1
                elif s in opp_seats:
                    opps.god_rounds += 1
                    if discarded_god[s]:
                        opps.god_discards += 1
    return ours, opps


def merge_into(dst: SideStats, src: SideStats) -> None:
    dst.scores.extend(src.scores)
    dst.wins += src.wins
    dst.fans.extend(src.fans)
    dst.win_turns.extend(src.win_turns)
    dst.baotou += src.baotou
    dst.fan_ge4 += src.fan_ge4
    dst.caipiao += src.caipiao
    dst.god_discards += src.god_discards
    dst.god_rounds += src.god_rounds


def report_day(label: str, ours: SideStats, opps: SideStats) -> list[str]:
    n_our = len(ours.scores)
    n_opp = len(opps.scores)
    if n_our == 0 or n_opp == 0:
        return [f"### {label}：无数据"]

    def m(xs):
        return sum(xs) / len(xs) if xs else 0.0

    lines = [f"### {label}（我方座位记录 {n_our} / 对手 {n_opp}，我方胡 {ours.wins} / 对手胡 {opps.wins}）", ""]

    # 比例类指标（带 CI）
    rows = [
        ("胡率", ours.wins, n_our, opps.wins, n_opp),
        ("爆头率(占胡牌)", ours.baotou, max(ours.wins, 1), opps.baotou, max(opps.wins, 1)),
        ("fan≥4率(占胡牌)", ours.fan_ge4, max(ours.wins, 1), opps.fan_ge4, max(opps.wins, 1)),
        ("财飘率(占胡牌)", ours.caipiao, max(ours.wins, 1), opps.caipiao, max(opps.wins, 1)),
        ("财神打出率", ours.god_discards, max(ours.god_rounds, 1), opps.god_discards, max(opps.god_rounds, 1)),
    ]
    lines.append(f"  {'指标':<16}{'我方':>10}{'对手':>10}{'差':>18}  判定")
    for name, k1, n1, k2, n2 in rows:
        p1 = k1 / max(n1, 1)
        p2 = k2 / max(n2, 1)
        d, lo, hi, sig = prop_diff(k1, n1, k2, n2)
        verdict = "✅显著" if sig else ("样本不足" if n1 < 30 or n2 < 30 else "不显著")
        lines.append(f"  {name:<16}{p1*100:>9.1f}%{p2*100:>9.1f}%{d*100:>+12.1f}pp  {verdict}")

    # 连续指标（走 ci_gate）
    lines.append("")
    v_score = compare_two_groups(ours.scores, opps.scores, label="均分(每座位每局)", min_n=30)
    lines.append(f"  均分：我方 {m(ours.scores):+.2f} vs 对手 {m(opps.scores):+.2f}  "
                 f"diff {v_score.diff:+.2f} CI[{v_score.ci_lo:+.2f},{v_score.ci_hi:+.2f}] "
                 f"{'✅' if v_score.ok else '❌'}")
    if ours.fans and opps.fans:
        v_fan = compare_two_groups(ours.fans, opps.fans, label="均番(胡时)", min_n=30)
        lines.append(f"  均番(胡时)：我方 {m(ours.fans):.2f} vs 对手 {m(opps.fans):.2f}  "
                     f"diff {v_fan.diff:+.2f} CI[{v_fan.ci_lo:+.2f},{v_fan.ci_hi:+.2f}] "
                     f"{'✅' if v_fan.ok else '❌'}")
    if ours.win_turns and opps.win_turns:
        v_turn = compare_two_groups(ours.win_turns, opps.win_turns, label="胡牌巡目", min_n=30)
        lines.append(f"  胡牌巡目(近似)：我方 {m(ours.win_turns):.1f} vs 对手 {m(opps.win_turns):.1f}  "
                     f"diff {v_turn.diff:+.2f} CI[{v_turn.ci_lo:+.2f},{v_turn.ci_hi:+.2f}] "
                     f"{'✅' if v_turn.ok else '❌'}")
    lines.append("")
    return lines


def opp_scores_safe(opps: SideStats) -> float:
    return sum(opps.scores) / len(opps.scores) if opps.scores else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=int, default=48, help="回溯小时数（按文件 mtime）")
    ap.add_argument("--strong-only", action="store_true", help="只对强 bot 对照组")
    ap.add_argument("--strong-names", default=",".join(STRONG_DEFAULT),
                    help="强 bot 座位名关键字（逗号分隔）")
    ap.add_argument("--out", default="agent/out/live-gap-report.txt")
    args = ap.parse_args()

    strong_names = tuple(x for x in args.strong_names.split(",") if x)
    cutoff = datetime.now().timestamp() - args.hours * 3600
    files = sorted(f for f in glob.glob("data/auto_sessions/*/events/*.json")
                   if os.path.getmtime(f) >= cutoff)
    print(f"近 {args.hours}h 事件文件：{len(files)} 个（strong_only={args.strong_only}）", flush=True)

    by_day: dict[str, list[str]] = defaultdict(list)
    for f in files:
        day = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%m-%d")
        by_day[day].append(f)

    lines = [
        "=" * 72,
        f"线上差距体检（{datetime.now():%Y-%m-%d %H:%M} 生成；近 {args.hours}h；"
        f"{'只对照强 bot：' + ','.join(strong_names) if args.strong_only else '对照全部对手'}）",
        "口径：我方=座位名含「玄武」；巡目=该座位已 discard 数（近似）；财神=「白」",
        "守门：比例类 n≥30 且 95%CI 不含 0 才报方向；连续类走 tools/ci_gate.py",
        "=" * 72,
        "",
    ]

    for day in sorted(by_day):
        ours_all, opps_all = SideStats(), SideStats()
        used = 0
        for f in by_day[day]:
            r = analyze_file(f, args.strong_only, strong_names)
            if r is None:
                continue
            merge_into(ours_all, r[0])
            merge_into(opps_all, r[1])
            used += 1
        lines.append(f"## {day}（{used}/{len(by_day[day])} 房有效）")
        lines.extend(report_day(day, ours_all, opps_all))

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n报告已写 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
