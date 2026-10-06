#!/usr/bin/env python
"""到听速度分桶差异（A 2026-10-06 16:49 [待C] 交办）。

**口径**（照抄 A 16:49②）：
1. **量**：每个座位「首次到听（shanten==0）的巡目」，真机全量局；
2. **两侧同口径**：我方 vs 头部 bot（全知视角，从 start_hands 重放）；
3. **分桶**：`财神数 (0 / 1 / ≥2)` × `是否已副露 (0 / ≥1)`；
4. **同时报**：到听巡目分布（p50/p90）与到听率（在给定巡数内到听的比例），不只是均值。

**预登记决策树**（A 16:49③，出数后按此执行，不重议口径）：
| 结果 | 判读 | 动作 |
|---|---|---|
| 有财神桶里我们比 bot 慢 ≥0.5 巡 | (甲) 到听慢 | 立项「到听速度」机制件 |
| 两侧无差异（<0.2 巡） | (乙) 听口质量 | 转「听口在自摸口径下的可达性」 |
| 无财神桶也慢 | 全属性牌效慢 | 回早巡牌效（需先给机制门） |

**巡目定义**：座位级「出手序数」（该座位第几次出牌，含开局弃牌），从 start_hands 重放时逐事件累计。

**判定方式**：每次手牌变化（摸牌/出牌/副露/杠）后，用 `shanten(counts, meld_count)` 算向听；
首次 `shanten==0` 时的巡目记为「到听巡目」。

**产物**：`agent/out/tenpai-speed.txt`。
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
from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay as R  # noqa: E402

import stage_a_dataset_export as EXP  # noqa: E402

OUR = "u_a7f7c67bb14a"

GOD_ID = tiles.GOD
GOD_TILES = {GOD_ID}


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


def god_bucket(n: int) -> str:
    if n == 0:
        return "god=0"
    if n == 1:
        return "god=1"
    return "god≥2"


def meld_bucket(m: int) -> str:
    return "meld=0" if m == 0 else "meld≥1"


def process_room(path, targets, stats, recs, seen_rounds):
    """重放一房，对每个座位记录首次到听巡目。同一 (room_id, round_no) 跨文档去重。"""
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
        # 每局每座位的到听记录（每局独立）
        tenpai_turn: dict[int, int | None] = {s: None for s in range(4)}
        turn_cnt: collections.Counter = collections.Counter()  # 每座位出手序数
        memo: dict = {}
        # 分桶键用**起手**状态：起手财神数（固有属性，避免「到听时点财神已打出」偏差）；
        # 副露用「该局该座位是否有过副露」（终态）
        start_god: dict[int, int] = {s: sum(state.seats[s].hand[t] for t in GOD_TILES if t < len(state.seats[s].hand)) for s in range(4)}

        for ev in events:
            seat = ev.get("seat")
            if seat is None:
                continue
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                stats["apply_event"] += 1
                break

            etype = ev.get("type", "")
            # 出手事件 = 出牌（tile_discarded）或副露响应（peng/chi/gang 等改变手牌）
            # 巡目 = 该座位第几次「主动动作」（出牌或副露）
            if etype == "tile_discarded" or etype in ("peng", "chi", "gang", "ming_gang", "an_gang", "bu_gang"):
                turn_cnt[seat] += 1

            # 手牌变化后检查到听（摸牌/出牌/副露后都算）
            if etype in ("tile_drawn", "tile_discarded", "peng", "chi", "gang", "ming_gang", "an_gang", "bu_gang"):
                if tenpai_turn[seat] is None:
                    ss = state.seats[seat]
                    try:
                        sh = shanten_mod.shanten(ss.hand, len(ss.melds), memo=memo)
                    except Exception:  # noqa: BLE001
                        continue
                    if sh == 0:
                        tenpai_turn[seat] = turn_cnt[seat]

        # 记录结果（**所有座位都进 recs**，未到听 turn=None —— 到听率分母需要）
        for s in range(4):
            recs.append({
                "seat": s,
                "is_bot": s in bot_seats,
                "turn": tenpai_turn[s],          # None = 本局未到听
                "god": start_god[s],              # 起手财神数
                "meld": len(state.seats[s].melds),  # 终态副露数（该局是否副露过）
                "room": Path(path).stem,
            })
            if tenpai_turn[s] is not None:
                stats["tenpai_recorded"] += 1
            else:
                stats["never_tenpai"] += 1


def percentile(data, p):
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


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

    print(f"\n== 汇总 ==", flush=True)
    print(f"去重局数 = {len(seen_rounds)}（dup_round 跳过 {stats['dup_round']}）", flush=True)
    print(f"记录到听点数 = {stats['tenpai_recorded']}", flush=True)
    print(f"未到听点数 = {stats['never_tenpai']}", flush=True)

    # 分桶统计（n=该侧总座位局数含未到听；tenpai=到听数；turns=到听者的巡目列表）
    bucket: dict[tuple, dict] = collections.defaultdict(
        lambda: {"bot_turns": [], "my_turns": [], "bot_n": 0, "my_n": 0, "bot_tenpai": 0, "my_tenpai": 0})
    for r in recs:
        gb = god_bucket(r["god"])
        mb = meld_bucket(r["meld"])
        key = (gb, mb)
        v = bucket[key]
        if r["is_bot"]:
            v["bot_n"] += 1
            if r["turn"] is not None:
                v["bot_tenpai"] += 1
                v["bot_turns"].append(r["turn"])
        else:
            v["my_n"] += 1
            if r["turn"] is not None:
                v["my_tenpai"] += 1
                v["my_turns"].append(r["turn"])

    # 输出
    out = []
    out.append(f"到听速度分桶差异（到听 {stats['tenpai_recorded']} 座位局 / 未到听 {stats['never_tenpai']}）")
    out.append("分桶=起手财神数 × 该局是否副露过；巡目=该座位出手序数；到听率=到听座位局/该侧总座位局")
    out.append("")
    out.append("分桶           |  bot n(率)      |  bot p50/p90/mean |  我方 n(率)     |  我方 p50/p90/mean | 巡目差(我-bot)")
    out.append("---------------|-----------------|-------------------|-----------------|--------------------|---------------")
    for key in sorted(bucket.keys()):
        v = bucket[key]
        bt = v["bot_turns"]
        mt = v["my_turns"]
        if not v["bot_n"] and not v["my_n"]:
            continue
        b_rate = f"{v['bot_tenpai']}/{v['bot_n']}({v['bot_tenpai']/v['bot_n']:.0%})" if v["bot_n"] else "—"
        m_rate = f"{v['my_tenpai']}/{v['my_n']}({v['my_tenpai']/v['my_n']:.0%})" if v["my_n"] else "—"
        b_stat = f"{percentile(bt,50):.1f}/{percentile(bt,90):.1f}/{statistics.mean(bt):.2f}" if bt else "—"
        m_stat = f"{percentile(mt,50):.1f}/{percentile(mt,90):.1f}/{statistics.mean(mt):.2f}" if mt else "—"
        diff = f"{statistics.mean(mt)-statistics.mean(bt):+.2f}" if (bt and mt) else "—"
        out.append(f"{key[0]:>6} {key[1]:>7} | {b_rate:>15} | {b_stat:>17} | {m_rate:>15} | {m_stat:>18} | {diff:>13}")

    # === A 18:23②：分布形状（p50/p75/p90/p95 + 累计到听率），分桶 有财神 × 副露 ===
    # 有财神 = god≥1（照抄 A 原文「有财神 × 副露有/无」两桶）
    out.append("")
    out.append("== 分布形状（A 18:23②）：有财神 × 副露，p50/p75/p90/p95 + 累计到听率（≤5/≤8/≤11/≤14 巡）==")
    shape_bucket: dict[tuple, dict] = collections.defaultdict(
        lambda: {"bot_turns": [], "my_turns": [], "bot_n": 0, "my_n": 0})
    for r in recs:
        gb = "god≥1" if r["god"] >= 1 else "god=0"
        mb = meld_bucket(r["meld"])
        key = (gb, mb)
        v = shape_bucket[key]
        if r["is_bot"]:
            v["bot_n"] += 1
            if r["turn"] is not None:
                v["bot_turns"].append(r["turn"])
        else:
            v["my_n"] += 1
            if r["turn"] is not None:
                v["my_turns"].append(r["turn"])

    def cum_rate(turns, total, cap):
        """≤cap 巡内到听的座位局占该侧总局数（含未到听）的比例。"""
        if not total:
            return None
        return sum(1 for t in turns if t <= cap) / total

    out.append("分桶         | 侧  | n(局) | p50 | p75 | p90 | p95 | ≤5巡 | ≤8巡 | ≤11巡 | ≤14巡")
    out.append("-------------|-----|-------|-----|-----|-----|-----|------|------|-------|-------")
    for key in sorted(shape_bucket.keys()):
        v = shape_bucket[key]
        for side, turns, n in (("bot", v["bot_turns"], v["bot_n"]), ("我方", v["my_turns"], v["my_n"])):
            if not n:
                continue
            if turns:
                cells = (f"{percentile(turns, p):.0f}" for p in (50, 75, 90, 95))
                p50, p75, p90, p95 = cells
            else:
                p50 = p75 = p90 = p95 = "—"
            cums = (f"{cum_rate(turns, n, c):.0%}" for c in (5, 8, 11, 14))
            c5, c8, c11, c14 = cums
            out.append(f"{key[0]:>5} {key[1]:>7} | {side:>3} | {n:>5} | {p50:>3} | {p75:>3} | {p90:>3} | {p95:>3} | {c5:>4} | {c8:>4} | {c11:>5} | {c14:>5}")
        out.append("-------------|-----|-------|-----|-----|-----|-----|------|------|-------|-------")
    text = "\n".join(out)
    print(text)
    (ROOT / "agent/out/tenpai-speed.txt").write_text(text, encoding="utf-8")
    print(f"\n-> agent/out/tenpai-speed.txt", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
