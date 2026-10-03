"""财神打法对照（S2 衍生，coordinator 22:42 自加，23:05 确认启动）。

任务：回答「手持财神时，bot 的做牌路径和我们差在哪」——为什么他们能把财神
转成爆头/财飘，我们转不成。

口径：20 个头部 bot（与 s2_fan_gap_analysis.py 同名单）vs 我们（凤凰-5531），
其他对手单列 OTHER 作参照。全量事件流 data/auto_sessions/*/events/*.json。

切片：
  1. 第 4 摸时点：财神数(0/1/≥2) × 向听(0/1/2/3+) → 本局最终胡率。
     （固定时点避免「赢家提前结束」截断；与 analyze_god_conversion.py 同口径）
  2. 听牌时点（出牌后 14 张扣打出）：财神数 × 听口种数 / 可见张数。
     （若财神手的听口更窄，则答案具体：财神价值=扩听口，我们没用上）

用法:
  setsid .venv/bin/python agent/verify/god_conversion_gap.py \
      > agent/out/god-conversion-gap.txt 2>&1 &
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles
from majiang.rules import win as win_module
from majiang.sim import replay

SEATS = 4
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"
MILESTONE = 4
GOD_BUCKETS = ("0张", "1张", "≥2张")
SHANTEN_BUCKETS = (0, 1, 2, 3)

US = "我们"
TOP = "头部bot"
OTHER = "其他对手"

TOP_BOTS = [
    "玄武-2346", "三杯猫", "铳一色14", "歪比巴卜肉蛋葱鸡", "爆头研究所",
    "Astra-0", "腾蛇-0638", "康陶应雀", "凤凰-2626", "glm-flash",
    "麒麟-7780", "白虎-0211", "豆包豆包帮我把其他AI电源拔掉",
    "Nomad", "双白平胡", "Kimi-K4.1", "菜菜子", "走马", "今晚打老虎",
    "晴总总，该请桂语山房了",
]
OUR_NAME = "凤凰-5531"
OUR_UID = "u_a7f7c67bb14a"


def god_bucket(gods: int) -> str:
    return "0张" if gods == 0 else ("1张" if gods == 1 else "≥2张")


def shanten_bucket(value: int) -> int:
    return min(max(value, 0), 3)


def live_copies(state: replay.ReplayState, seat: int, waits: tuple[int, ...]) -> int:
    visible = shanten_module.visible_counts(
        list(state.seats[seat].hand),
        [meld.tiles for other in state.seats for meld in other.melds],
        [list(other.discards) for other in state.seats],
    )
    return sum(max(0, tiles.COPIES_PER_KIND - visible[tile]) for tile in waits)


def seat_group(uid: str, top_uids: set[str]) -> str:
    if uid == OUR_UID:
        return US
    return TOP if uid in top_uids else OTHER


def scan(payload: dict, top_uids: set[str], cells: dict, waits_d: dict,
         counters: Counter) -> None:
    seats_meta = payload.get("seats") or []
    ids = [str(s.get("user_id", "")) for s in seats_meta]
    if len(ids) != SEATS or OUR_UID not in ids:
        return
    if not any(u in top_uids for u in ids):
        return  # 只统计至少有 1 个头部 bot 在场的房，对齐 S2 口径
    official = {
        int(e.get("round_no", 0) or 0): e for e in (payload.get("rounds") or [])
    }

    for state, events in replay.iter_rounds(payload):
        milestones: dict[int, tuple[int, int]] = {}
        draws: Counter = Counter()
        for event in events:
            kind = str(event.get("type"))
            seat = event.get("seat")
            if kind == DRAWN and isinstance(seat, int) and 0 <= seat < SEATS:
                draws[seat] += 1
                if draws[seat] == MILESTONE:
                    ss = state.seats[seat]
                    try:
                        v = shanten_module.shanten_any(ss.hand, len(ss.melds))
                    except Exception:
                        v = -1
                    milestones[seat] = (ss.hand[tiles.GOD], v)
            elif kind == DISCARDED and isinstance(seat, int) and 0 <= seat < SEATS:
                tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
                if tile is not None:
                    counts = list(state.seats[seat].hand)
                    counts[tile] -= 1
                    mc = len(state.seats[seat].melds)
                    try:
                        v = shanten_module.shanten_any(counts, mc)
                    except Exception:
                        v = -1
                    if v == 0:
                        wl = win_module.winning_draws(counts, mc)
                        if wl:
                            grp = seat_group(ids[seat], top_uids)
                            key = (grp, god_bucket(counts[tiles.GOD]))
                            entry = waits_d[key]
                            entry[0] += 1
                            entry[1] += len(wl)
                            entry[2] += live_copies(state, seat, wl)
            replay.apply_event(state, event)

        entry = official.get(state.round_no) or {}
        winner = None if entry.get("is_draw") else entry.get("winner")
        for seat in range(SEATS):
            grp = seat_group(ids[seat], top_uids)
            counters[f"{grp} 参与局数"] += 1
            if seat not in milestones:
                counters[f"{grp} 未到第{MILESTONE}摸"] += 1
                continue
            gods, sh = milestones[seat]
            if sh < 0:
                counters[f"{grp} 向听不可算"] += 1
                continue
            cell = cells[(grp, god_bucket(gods), shanten_bucket(sh))]
            cell[0] += 1
            if isinstance(winner, int) and winner == seat:
                cell[1] += 1


def report_cells(cells: dict) -> None:
    print("== 交叉表：第 4 摸时（财神数 × 向听）→ 本局最终胡牌率 ==")
    groups = (US, TOP, OTHER)
    header = f"{'向听':>4} {'财神':>5} |"
    for g in groups:
        header += f" {g+' n':>8} {g+'胡率':>9} |"
    header += " TOP−我们"
    print(header)
    for sh in SHANTEN_BUCKETS:
        for gods in GOD_BUCKETS:
            row = f"{sh:>4} {gods:>5} |"
            ns = {}
            rates = {}
            skip = False
            for g in groups:
                n, w = cells[(g, gods, sh)]
                ns[g] = n
                rates[g] = (w / n) if n else 0.0
                if n < 40:
                    skip = True
            if skip:
                continue
            for g in groups:
                row += f" {ns[g]:>8} {rates[g]:>9.2%} |"
            row += f" {rates[TOP] - rates[US]:+8.2%}"
            print(row)
        print()


def report_waits(waits_d: dict) -> None:
    print("\n== 听牌时的听口（出牌后判定，含可见张数）==")
    print(f"{'组别':<8}{'财神':>5}{'样本':>8}{'听口种数':>10}{'可见张数':>10}")
    for grp in (US, TOP, OTHER):
        for gods in GOD_BUCKETS:
            n, kinds, copies = waits_d[(grp, gods)]
            if not n:
                continue
            print(f"{grp:<8}{gods:>5}{n:>8}{kinds / n:>10.2f}{copies / n:>10.2f}")
        print()


def main() -> int:
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    print(f"事件流: {len(files)}", flush=True)

    # 第一遍：name → uid
    name_to_uid: dict[str, str] = {}
    for fp in files:
        try:
            d = json.load(open(fp))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception:
            pass
    top_uids = {name_to_uid[n] for n in TOP_BOTS if n in name_to_uid}
    print(f"头部 bot 命中: {len(top_uids)}/{len(TOP_BOTS)}", flush=True)
    if OUR_UID != name_to_uid.get(OUR_NAME, OUR_UID):
        print(f"⚠️ 我们 uid 不符: 参数 {OUR_UID} vs 名册 {name_to_uid.get(OUR_NAME)}",
              flush=True)

    cells: dict = defaultdict(lambda: [0, 0])
    waits_d: dict = defaultdict(lambda: [0, 0, 0])
    counters: Counter = Counter()

    for i, fp in enumerate(files):
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)
        try:
            payload = json.load(open(fp))
        except Exception:
            continue
        scan(payload, top_uids, cells, waits_d, counters)

    print(f"\n文件 {len(files)} 个（向听取每座第 {MILESTONE} 次摸牌之后）。")
    print("原始计数：" + " ".join(f"{k}={v}" for k, v in sorted(counters.items())) + "\n")
    report_cells(cells)
    report_waits(waits_d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
