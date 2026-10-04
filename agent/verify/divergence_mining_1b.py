"""1b：在头部强 bot 的决策点上跑 v6，按 巡目×向听×财神×动作类型 分桶挖分歧（A 21:25 分工表）。

数据源：data/auto_sessions/*/events/*.json（真实对局事件流）。
方法：
  - 用 sim/replay.py 的 iter_rounds 逐事件重建「该事件发生之前」的局面
    （iter_rounds 只产出状态与事件序列，apply_event 由本脚本逐步调用）；
  - 当事件是强 bot 的 peng/chi/pass（对别家弃牌的响应）或 tile_discarded（出牌）时，
    以该 bot 座位为 observer 构造 Situation，喂给 v6（versions.build），比较 v6 选择 vs bot 实际动作；
  - 副露响应分 peng 窗口与 chi 窗口两层（与线上窗口顺序一致）；
  - 分桶：动作类型 × 巡目段 × 向听 × 是否有财神。巡目 = 该 seat 已摸牌数（自摸巡）；
    全局巡目段（总摸牌数//4）另备。
产物：agent/out/divergence-1b.txt

口径说明（诚实声明）：
  - 「v6 会怎么做」= 离线 v6 在重建局面上 argmax；真实 bot 动作来自事件流。
  - pass 事件 = bot 主动放弃窗口；timeout 不算 bot 决策（服务端兜底），单独计数不混入分歧率。
  - chi 窗口只在「上家弃牌」时出现；按平台顺序：tile_discarded 后先 peng 窗口（其余三家），
    后 chi 窗口（下家）。下家的 pass 归属判定：看该 seat 在窗口链里 pass 的次数——
    一次 pass 时两窗各记一个决策点（v6 两窗都算），两次 pass 时第一次归 peng、第二次归 chi。
用法: setsid .venv/bin/python agent/verify/divergence_mining_1b.py > agent/out/divergence-1b.txt 2>&1
"""
import json
import glob
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")

from majiang.sim import replay as R
from majiang.rules import tiles, shanten as sh
from majiang.rules.action import DISCARD, PENG, CHI
from majiang.rules.action import legal_actions
from majiang.rules.situation import PHASE_RESPONSE_PENG, PHASE_RESPONSE_CHI, PHASE_DRAW
from majiang.strategy import versions
from majiang.strategy.policy import Mode

TOP_BOTS = [
    "玄武-2346","三杯猫","铳一色14","歪比巴卜肉蛋葱鸡","爆头研究所",
    "Astra-0","腾蛇-0638","康陶应雀","凤凰-2626","glm-flash",
    "麒麟-7780","白虎-0211","豆包豆包帮我把其他AI电源拔掉",
    "Nomad","双白平胡","Kimi-K4.1","菜菜子","走马","今晚打老虎",
    "晴总总，该请桂语山房了",
]

TURN_BUCKETS = [(0, 5, "1-5"), (6, 10, "6-10"), (11, 15, "11-15"), (16, 99, "16+")]


def turn_bucket(turn_no: int) -> str:
    for lo, hi, name in TURN_BUCKETS:
        if lo <= turn_no <= hi:
            return name
    return "16+"


def shanten_bucket(shanten: int) -> str:
    if shanten <= 0:
        return "听牌"
    if shanten == 1:
        return "1向听"
    if shanten == 2:
        return "2向听"
    return "3+向听"


def action_label(action) -> str:
    if action is None:
        return "none"
    k = action.kind
    if k == DISCARD:
        return f"discard:{tiles.to_code(action.tile)}"
    if k in (PENG, CHI):
        return k
    return k


def main(sample_n: int = 0) -> None:
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    if sample_n > 0 and len(files) > sample_n:
        import random
        random.seed(42)
        files = random.sample(files, sample_n)
        print(f"采样: {sample_n}/{len(sorted(glob.glob('data/auto_sessions/*/events/*.json')))} 房", flush=True)
    print(f"事件流: {len(files)}", flush=True)

    decider = versions.build("v6", Mode.QUALIFIER)
    print(f"决策器: {decider.name}", flush=True)

    # name -> uid
    name_to_uid = {}
    for fpath in files:
        try:
            d = json.load(open(fpath))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception:
            pass
    targets = {name_to_uid[n]: n for n in TOP_BOTS if n in name_to_uid}
    print(f"目标 bot: {len(targets)}/20", flush=True)

    # 桶：key = (动作类型, 巡目段, 向听, 财神有无)
    buckets = defaultdict(lambda: [0, 0])  # [n, divergent]
    totals = Counter()   # "出牌:n" / "出牌:div" 等
    window_stats = Counter()
    errors = Counter()
    # 每 seat 已摸牌数（每个 round state 独立）
    seat_draws = [0, 0, 0, 0]

    for fi, fpath in enumerate(files):
        if fi % 1000 == 0:
            print(f"  {fi}/{len(files)}", flush=True)
        try:
            payload = json.load(open(fpath))
        except Exception:
            continue
        seats = payload.get("seats", [])
        seat_of_uid = {}
        for si, s in enumerate(seats):
            u = s.get("user_id")
            if u in targets:
                seat_of_uid[si] = targets[u]
        if not seat_of_uid:
            continue

        for state, events in R.iter_rounds(payload):
            seat_draws = [0, 0, 0, 0]
            n_events = len(events)
            for idx, event in enumerate(events):
                etype = event.get("type")
                seat = event.get("seat")

                # ---- 目标 bot 的决策点判定（在 apply_event 之前）----
                if isinstance(seat, int) and seat in seat_of_uid:

                    # 出牌决策点
                    if etype == "tile_discarded":
                        tile = R._tile_of(event.get("tile"))
                        if tile is not None and state.opened:
                            try:
                                sit = state.situation_for(seat, phase=PHASE_DRAW, drawn=None)
                                hand_counts = list(state.seats[seat].hand)
                                meld_count = len(state.seats[seat].melds)
                                shanten = sh.shanten_any(hand_counts, meld_count=meld_count)
                                god_n = sum(
                                    hand_counts[t]
                                    for t in range(tiles.TILE_KINDS)
                                    if tiles.is_god(t)
                                )
                                tb = turn_bucket(seat_draws[seat])
                                sb = shanten_bucket(shanten)
                                gb = "有财神" if god_n > 0 else "无财神"
                                acts = legal_actions(sit)
                                if acts:
                                    chosen = decider.choose(sit, acts, budget_ms=600)
                                    v6_lab = action_label(chosen)
                                    bot_lab = f"discard:{tiles.to_code(tile)}"
                                    key = ("出牌", tb, sb, gb)
                                    buckets[key][0] += 1
                                    totals["出牌:n"] += 1
                                    if v6_lab != bot_lab:
                                        buckets[key][1] += 1
                                        totals["出牌:div"] += 1
                                else:
                                    errors["no_actions_discard"] += 1
                            except Exception as exc:
                                # 不吞异常类型（10-04 tiles.label bug 曾把 24 万点全吞成 n=0）
                                errors[f"decide_discard:{type(exc).__name__}"] += 1

                    # 副露响应决策点
                    elif etype in ("peng", "chi", "pass"):
                        # 被响应的牌 = 往前最近的 tile_discarded（别家打的）
                        offered = None
                        discarder = None
                        for j in range(idx - 1, -1, -1):
                            pj = events[j]
                            if pj.get("type") == "tile_discarded":
                                offered = R._tile_of(pj.get("tile"))
                                discarder = pj.get("seat")
                                break
                            if pj.get("type") in ("peng", "chi", "gang", "tile_drawn"):
                                break
                        if offered is not None and discarder != seat and state.opened:
                            is_next = (discarder + 1) % 4 == seat
                            if etype == "peng":
                                windows = [("peng", PHASE_RESPONSE_PENG)]
                            elif etype == "chi":
                                windows = [("chi", PHASE_RESPONSE_CHI)]
                            else:  # pass
                                if is_next:
                                    # 该 seat 在本响应链里的 pass 序号
                                    chain_idx = 0
                                    for j in range(idx, min(n_events, idx + 4)):
                                        pj = events[j]
                                        if pj.get("type") in (
                                            "tile_drawn","peng","chi","gang","tile_discarded",
                                        ):
                                            break
                                        if pj.get("seat") == seat and pj.get("type") == "pass":
                                            if j == idx:
                                                break
                                            chain_idx += 1
                                    if chain_idx == 0:
                                        windows = [
                                            ("peng", PHASE_RESPONSE_PENG),
                                            ("chi", PHASE_RESPONSE_CHI),
                                        ]
                                    else:
                                        windows = [("chi", PHASE_RESPONSE_CHI)]
                                else:
                                    windows = [("peng", PHASE_RESPONSE_PENG)]

                            for wname, phase in windows:
                                window_stats[f"{wname}:n"] += 1
                                try:
                                    sit = state.situation_for(
                                        seat, phase=phase,
                                        offered=offered, responding=(seat,),
                                    )
                                    hand_counts = list(state.seats[seat].hand)
                                    meld_count = len(state.seats[seat].melds)
                                    shanten = sh.shanten_any(hand_counts, meld_count=meld_count)
                                    god_n = sum(
                                        hand_counts[t]
                                        for t in range(tiles.TILE_KINDS)
                                        if tiles.is_god(t)
                                    )
                                    tb = turn_bucket(seat_draws[seat])
                                    sb = shanten_bucket(shanten)
                                    gb = "有财神" if god_n > 0 else "无财神"
                                    acts = legal_actions(sit)
                                    if not acts:
                                        errors[f"no_actions_{wname}"] += 1
                                        continue
                                    chosen = decider.choose(sit, acts, budget_ms=600)
                                    v6_lab = action_label(chosen)
                                    bot_lab = etype if etype in ("peng", "chi") else "pass"
                                    key = (f"{wname}窗口", tb, sb, gb)
                                    buckets[key][0] += 1
                                    totals[f"{wname}窗口:n"] += 1
                                    if v6_lab != bot_lab:
                                        buckets[key][1] += 1
                                        totals[f"{wname}窗口:div"] += 1
                                except Exception:
                                    errors[f"decide_{wname}"] += 1

                # ---- 推进状态与巡目计数（所有事件都要 apply，不只目标 bot 的）----
                if etype == "tile_drawn" and isinstance(seat, int) and 0 <= seat < 4:
                    seat_draws[seat] += 1
                R.apply_event(state, event)

    # ---------- 输出 ----------
    print("\n=== 决策点总量 ===")
    for k in ("出牌", "peng窗口", "chi窗口"):
        n = totals[f"{k}:n"]
        dv = totals[f"{k}:div"]
        rate = dv / n * 100 if n else 0
        print(f"  {k}: n={n}, 分歧={dv} ({rate:.1f}%)")

    print("\n=== 分歧率 Top 桶（n>=100 才排）===")
    rows = []
    for (atype, tb, sb, gb), (n, dv) in buckets.items():
        if n < 100:
            continue
        rows.append((dv / n, n, dv, atype, tb, sb, gb))
    rows.sort(reverse=True)
    print(f"{'动作':<8s} {'巡目':<6s} {'向听':<8s} {'财神':<6s} {'n':>7s} {'分歧':>7s} {'分歧率':>7s}")
    for rate, n, dv, atype, tb, sb, gb in rows[:25]:
        print(f"{atype:<8s} {tb:<6s} {sb:<8s} {gb:<6s} {n:>7d} {dv:>7d} {rate*100:>6.1f}%")

    print("\n=== 全桶明细 ===")
    for (atype, tb, sb, gb), (n, dv) in sorted(buckets.items(), key=lambda x: -x[1][0]):
        rate = dv / n * 100 if n else 0
        print(f"  {atype} | {tb} | {sb} | {gb} | n={n} 分歧={dv} ({rate:.1f}%)")

    print(f"\n窗口统计: {dict(window_stats)}")
    print(f"异常计数: {dict(errors)}")
    print("ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(sample_n=_n)
