#!/usr/bin/env python
"""自摸口径的听口可达性（A 2026-10-06 17:28③ 交办，量已定义）。

**口径（照抄 A 17:28③）**：在**我方已听牌但最终未胡**的局里（对照=bot 同口径），报三个分布，分桶 有财神/无财神 × 副露有/无：

| 量 | 定义 | 若它为主 ⇒ 结论 |
|---|---|---|
| (i) 听口剩余张数 | 听口牌的 `4 − 可见`（可见=自己手牌+四家弃牌+四家副露）的总和 | 可达性/被持有 ⇒ 机制件=听口选择 |
| (ii) 听口类型 | 两面/嵌张/边张/单骑/对倒 占比 | 愚形 ⇒ 机制件=听口质量 |
| (iii) 到听后的剩余巡数 | 局终巡目 − 到听巡目 | 时机 ⇒ 回到速度×时机 |

**预登记判据（照抄）**：**(iii) 为主（普遍 ≤1 巡）⇒ 时机问题**；**(i)/(ii) 为主（有巡数却胡不了）⇒ 听口可达性/愚形问题**。

产物：`agent/out/wait-accessibility.txt`。
"""
from __future__ import annotations

import collections
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent" / "verify"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles, win as win_mod  # noqa: E402
from majiang.sim import replay as R  # noqa: E402

import stage_a_dataset_export as EXP  # noqa: E402

OUR = "u_a7f7c67bb14a"
GOD_ID = tiles.GOD
KINDS = tiles.TILE_KINDS


def build_targets(files):
    name_to_uid = {}
    for fpath in files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception:  # noqa: BLE001
            pass
    return {name_to_uid[n]: n for n in EXP.TOP_BOTS if n in name_to_uid}


def wait_types(counts: list[int], meld_count: int) -> dict[str, int]:
    """对一手 13 张听牌，返回各听口类型的张数 {两面, 嵌张, 边张, 单骑, 对倒}。

    对每张能胡的牌 tile（`winning_draws` 返回），判定其类型：尝试把 tile 当作
    顺子完成（两面/嵌张/边张）、将牌对子（单骑）、刻子完成（对倒）。
    若同一张牌可多解，取优先级：两面 > 对倒 > 嵌张 > 边张 > 单骑。
    """
    try:
        draws = win_mod.winning_draws(counts, meld_count)
    except ValueError:
        return {}
    types: dict[str, int] = collections.Counter()
    for t in draws:
        ttype = classify(counts, t)
        types[ttype] += 1
    return dict(types)


def classify(counts: list[int], tile: int) -> str:
    """粗分类一张听口牌的类型（启发式，仅供占比分布用）。"""
    # 字牌（含财神）不可能是顺子/刻子的中间张（除对倒/单骑）
    if tile >= 27:
        # 字牌只能单骑（等对子）或对倒（等刻子）
        if counts[tile] >= 2:
            return "对倒"
        return "单骑"
    # 数牌：看相邻
    rank = tile % 9
    left2 = counts[tile - 2] if rank >= 2 else 0
    left1 = counts[tile - 1] if rank >= 1 else 0
    right1 = counts[tile + 1] if rank <= 7 else 0
    right2 = counts[tile + 2] if rank <= 6 else 0
    # 两面：左1+右1 都有（等中间张不算——那是嵌张）
    if left1 >= 1 and right1 >= 1:
        return "两面"
    # 边张：12 等 3 / 89 等 7
    if rank == 2 and left2 >= 1 and left1 >= 1:  # 等 3（手中 1,2）
        return "边张"
    if rank == 6 and right1 >= 1 and right2 >= 1:  # 等 7（手中 8,9）
        return "边张"
    # 嵌张：左右各空一（左2+右1 或 左1+右2）
    if (left2 >= 1 and right1 >= 1) or (left1 >= 1 and right2 >= 1):
        return "嵌张"
    # 对倒：手中已有对子（≥2）
    if counts[tile] >= 2:
        return "对倒"
    return "单骑"


def process_room(path, targets, stats, recs, seen_rounds):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stats["bad_json"] += 1
        return
    room_id = doc.get("room_id") or Path(path).stem
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    if len(ids) != 4 or OUR not in ids:
        stats["no_our_seat"] += 1
        return
    my_seat = ids.index(OUR)
    bot_seats = [i for i, u in enumerate(ids) if u in targets]
    if not bot_seats:
        stats["no_target_bot"] += 1
        return
    try:
        rounds_meta = {r.get("round_no"): r for r in (doc.get("rounds") or [])}
        rounds = list(R.iter_rounds(doc))
    except Exception:  # noqa: BLE001
        stats["iter_rounds"] += 1
        return

    for state, events in rounds:
        rno = getattr(state, "round_no", None)
        rkey = (room_id, rno)
        if rkey in seen_rounds:
            stats["dup_round"] += 1
            continue
        seen_rounds.add(rkey)

        rmeta = rounds_meta.get(rno) or {}
        winner = rmeta.get("winner") if not rmeta.get("is_draw") else None

        start_god = {s: state.seats[s].hand[GOD_ID] for s in range(4)}
        turn_cnt: collections.Counter = collections.Counter()
        # 每座位的首次到听巡目 + 到听时刻的听口快照（听口牌/类型/剩余张数）
        first_tenpai_turn: dict[int, int | None] = {s: None for s in range(4)}
        tenpai_snapshot: dict[int, dict] = {}
        memo: dict = {}

        for ev in events:
            seat = ev.get("seat")
            if seat is None:
                continue
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                stats["apply_event"] += 1
                break
            if ev.get("type") in ("tile_discarded", "peng", "chi", "gang", "ming_gang", "an_gang", "bu_gang"):
                turn_cnt[seat] += 1
            # 检查到听（摸牌/出牌/副露后）
            if ev.get("type") in ("tile_drawn", "tile_discarded", "peng", "chi", "gang", "ming_gang", "an_gang", "bu_gang"):
                if first_tenpai_turn[seat] is None:
                    ss = state.seats[seat]
                    try:
                        sh = shanten_mod.shanten(ss.hand, len(ss.melds), memo=memo)
                    except Exception:  # noqa: BLE001
                        continue
                    if sh == 0:
                        first_tenpai_turn[seat] = turn_cnt[seat]
                        # 听口快照：手牌必须是 13 张（出牌后）才能算 winning_draws
                        if sum(ss.hand) == 13 - 3 * len(ss.melds):
                            try:
                                draws = win_mod.winning_draws(list(ss.hand), len(ss.melds))
                                # 可见计数：自己手牌 + 四家弃牌 + 四家副露
                                visible = list(state.seats[seat].hand)  # 先算自己手牌
                                for s2 in range(4):
                                    for t in state.seats[s2].discards:
                                        visible[t] += 1
                                    for meld in state.seats[s2].melds:
                                        for t in meld.tiles:
                                            visible[t] += 1
                                remaining = sum(4 - visible[t] for t in draws if visible[t] < 4)
                                types = wait_types(list(ss.hand), len(ss.melds))
                                tenpai_snapshot[seat] = {
                                    "draws": list(draws),
                                    "remaining": remaining,
                                    "types": types,
                                }
                            except Exception:  # noqa: BLE001
                                pass

        end_turn = max(turn_cnt.values()) if turn_cnt else 0
        # 只记录「已听牌但未胡」的座位
        for s in range(4):
            if s != my_seat and s not in bot_seats:
                continue
            if first_tenpai_turn[s] is None:
                continue  # 未到听
            if winner == s:
                continue  # 已胡（我们只关心「听了但没胡」）
            ss = state.seats[s]
            snap = tenpai_snapshot.get(s) or {}
            recs.append({
                "seat": s,
                "is_bot": s in bot_seats,
                "is_me": s == my_seat,
                "god": start_god[s],
                "meld": len(ss.melds),
                "tenpai_turn": first_tenpai_turn[s],
                "end_turn": end_turn,
                "remaining_turns": end_turn - first_tenpai_turn[s],
                "wait_remaining": snap.get("remaining"),
                "wait_types": snap.get("types") or {},
                "wait_kinds": len(snap.get("draws") or []),
            })


def god_bucket(n): return "god=0" if n == 0 else ("god=1" if n == 1 else "god≥2")
def meld_bucket(m): return "meld=0" if m == 0 else "meld≥1"


def pct_dist(counter: dict, keys) -> str:
    n = sum(counter.get(k, 0) for k in keys)
    if not n:
        return "—"
    return "/".join(f"{counter.get(k, 0)/n:.0%}" for k in keys)


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    print(f"事件文档 {len(files)} 个", flush=True)
    targets = build_targets(files)
    print(f"目标 bot: {len(targets)}/{len(EXP.TOP_BOTS)}", flush=True)

    stats = collections.Counter()
    recs: list[dict] = []
    seen_rounds: set = set()
    for f in files:
        process_room(f, targets, stats, recs, seen_rounds)

    print(f"去重局数 = {len(seen_rounds)}", flush=True)
    print(f"已听未胡座位局 = {len(recs)}", flush=True)

    my = [r for r in recs if r["is_me"]]
    bot = [r for r in recs if r["is_bot"]]
    print(f"我方 {len(my)} / bot {len(bot)}", flush=True)

    out = []
    out.append(f"自摸口径听口可达性（{len(seen_rounds)} 去重局；已听未胡：我方 {len(my)} / bot {len(bot)}）")
    out.append("")

    # (iii) 到听后剩余巡数
    out.append("== (iii) 到听后剩余巡数（局终巡目 − 到听巡目）==")
    out.append("侧 | n | p50 | p90 | mean | ≤1巡占比 | ≤2巡占比")
    out.append("---|---|-----|-----|------|----------|----------")
    for name, rows in (("bot", bot), ("我方", my)):
        v = [r["remaining_turns"] for r in rows]
        if not v:
            continue
        srt = sorted(v)
        p50 = srt[len(srt) // 2]
        p90 = srt[int(len(srt) * 0.9)]
        mean = statistics.mean(v)
        le1 = sum(1 for x in v if x <= 1) / len(v)
        le2 = sum(1 for x in v if x <= 2) / len(v)
        out.append(f"{name:>4} | {len(v):>4} | {p50:>3} | {p90:>3} | {mean:>4.1f} | {le1:>8.1%} | {le2:>8.1%}")
    out.append("判据：(iii) 为主（普遍 ≤1 巡）⇒ 时机问题")
    out.append("")

    # (i) 听口剩余张数
    out.append("== (i) 听口剩余张数（听口牌的 4−可见 之和）==")
    out.append("侧 | n(有快照) | p50 | p90 | mean")
    out.append("---|-----------|-----|-----|-----")
    for name, rows in (("bot", bot), ("我方", my)):
        v = [r["wait_remaining"] for r in rows if r["wait_remaining"] is not None]
        if not v:
            continue
        srt = sorted(v)
        p50 = srt[len(srt) // 2]
        p90 = srt[int(len(srt) * 0.9)]
        out.append(f"{name:>4} | {len(v):>9} | {p50:>3} | {p90:>3} | {statistics.mean(v):>4.1f}")
    out.append("判据：(i) 为主（有巡数却胡不了 + 剩余张数低）⇒ 听口可达性/被持有问题")
    out.append("")

    # (ii) 听口类型
    out.append("== (ii) 听口类型（已听未胡者的听牌型分布；顺序：两面/对倒/嵌张/边张/单骑）==")
    out.append("侧 | n | 两面/对倒/嵌张/边张/单骑 占比")
    out.append("---|---|------------------------------")
    TYPE_KEYS = ("两面", "对倒", "嵌张", "边张", "单骑")
    for name, rows in (("bot", bot), ("我方", my)):
        agg: dict[str, int] = collections.Counter()
        for r in rows:
            for k, v in r["wait_types"].items():
                agg[k] += v
        out.append(f"{name:>4} | {len(rows):>4} | {pct_dist(agg, TYPE_KEYS)}")
    out.append("判据：(ii) 为主（愚形占比高）⇒ 听口质量问题")
    out.append("")

    # 分桶
    out.append("== 分桶（起手财神 × 终局副露；已听未胡）==")
    out.append("分桶 | bot n | bot 剩余巡mean | bot 剩余张mean | 我方 n | 我方剩余巡mean | 我方剩余张mean")
    out.append("-----|-------|---------------|----------------|--------|----------------|-----------------")
    buckets: dict[tuple, dict] = collections.defaultdict(lambda: {"bot": [], "me": []})
    for r in recs:
        key = (god_bucket(r["god"]), meld_bucket(r["meld"]))
        if r["is_bot"]:
            buckets[key]["bot"].append(r)
        elif r["is_me"]:
            buckets[key]["me"].append(r)
    for key in sorted(buckets.keys()):
        v = buckets[key]
        bt = [r["remaining_turns"] for r in v["bot"]]
        bw = [r["wait_remaining"] for r in v["bot"] if r["wait_remaining"] is not None]
        mt = [r["remaining_turns"] for r in v["me"]]
        mw = [r["wait_remaining"] for r in v["me"] if r["wait_remaining"] is not None]
        out.append(
            f"{key[0]:>6} {key[1]:>6} | {len(v['bot']):>5} | "
            f"{f'{statistics.mean(bt):.1f}' if bt else '—':>13} | {f'{statistics.mean(bw):.1f}' if bw else '—':>14} | "
            f"{len(v['me']):>6} | {f'{statistics.mean(mt):.1f}' if mt else '—':>14} | {f'{statistics.mean(mw):.1f}' if mw else '—':>15}"
        )

    text = "\n".join(out)
    print(text)
    (ROOT / "agent/out/wait-accessibility.txt").write_text(text, encoding="utf-8")
    print(f"\n-> agent/out/wait-accessibility.txt", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
