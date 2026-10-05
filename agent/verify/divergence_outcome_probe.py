"""分歧点后果对拍 —— A 2026-10-04 14:35 ③ 交办（[待C]，只读、不占通道、采样即可）。

问题：1b v2 实测我方与头部 bot 出牌分歧率 19.5%（13,429/68,780），且全在 1-3 向听做牌路径
（听牌后高度一致）。「分歧率高 ≠ 我们错」——必须量后果。

方法：replay 真机事件流的我方出牌点，用 v6（= 线上决策器，1b v2 同款）离线决策，
比较「v6 会打什么」vs「我方实际打了什么」：
  - 一致点：v6 选择 == 实际打出（我方行为与离线 v6 重合）；
  - 分歧点：v6 选择 != 实际打出（我方行为偏离离线 v6）。
对每个点，继续 replay 到局终，量结局：
  a. 本局我方是否胡 / 均番；
  b. 本局对手任一胡率（= 1 − 我方胡 − 流局）/ 对手均番；
  c. 分歧点距局终剩余巡数（余巡）——后果窗口长度；
  d. 分桶：巡目段 × 向听 × 财神数（照 1b 口径）。

判读（A 预登记）：
  - 「分歧点的我们胡率」显著低于一致点（差 ≥3pp）且对手在同手更高 ⇒ 我们的做牌路径确有系统性缺陷
    ⇒ 把该桶交 A 做机制件（出牌层排序键 / 财神 buffer 保存策略）；
  - 两组结局无差别 ⇒ 分歧是路径多样性、不是缺陷 ⇒ 封存线索，全力转 S3。

诚实声明：
  - 「实际打出」来自事件流（我方真机动作），「v6 会打什么」是离线重建局面的 argmax。
    两者分歧 = 「真机决策偏离离线 v6」，原因可以是运行时窗口/超时/状态差——本探针只量**后果**，
    不归因分歧来源。
  - 一致性标注与 1b v2 同源（同 v6、同预算 600ms、同 Situation 构造）⇒ 分歧率应复现 ~19.5%
    作为仪表自检（偏差 >3pp 先查仪表再报数）。
  - 只统计**我方出牌点**（1b v2 的最大分歧源）；peng/chi 窗口分歧率低（2.2~5.1%）且
    窗口动作后果口径不同，不在本探针。

口径：
  - 巡目段：1-5 / 6-10 / 11-15 / 16+（照 1b）；
  - 向听：听牌(0) / 1向听 / 2向听 / 3+向听（照 1b）；
  - 财神：有 / 无（照 1b）；
  - 结局：is_draw ⇒ 流局；否则 winner == 我方座位 ⇒ 我方胡，winner 其它 ⇒ 对手胡。
  - 均番只对胡牌局计（流局局番数记 0 不计入均番分母）。
  - 余巡 = 分歧点后该局我方剩余摸牌数（我方每摸一次算一巡）。

数据源：data/auto_sessions/*/events/*.json，与 1b v2 同批（采样 ~2000 房 ⇒ 预期
我方出牌点 ~7 万、分歧点 ~1.3 万，结局可算）。单核成本：v6 choose 42-157ms/点 × 7 万点
≈ 1-3h（主要成本在一致/分歧标注，replay 到局终几乎免费）。分块幂等落盘
agent/out/div-outcome-chunks/cs<N>-n<M>/，DONE 收尾。

用法：`.venv/bin/python agent/verify/divergence_outcome_probe.py [--rooms N] [--chunk-size N]`
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import shanten as sh  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay as R  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"
WIN = "round_ended"  # 局终事件：data={dealer, detail, draw, fan, round_no, scores}

TURN_BUCKETS = ("1-5", "6-10", "11-15", "16+")


def turn_bucket(n: int) -> str:
    return TURN_BUCKETS[0] if n <= 5 else (TURN_BUCKETS[1] if n <= 10 else (TURN_BUCKETS[2] if n <= 15 else TURN_BUCKETS[3]))


def sh_bucket(s: int) -> str:
    if s == 0:
        return "听牌"
    if s == 1:
        return "1向听"
    if s == 2:
        return "2向听"
    return "3+向听"


def god_bucket(g: int) -> str:
    return "有财神" if g > 0 else "无财神"


def process_batch(batch, chunk_path, decider):
    # 每桶（巡目段×向听×财神×一致/分歧）：[n, 我方胡n, 对手胡n, 流局n, 我方番sum, 对手番sum,
    #   我方胡番sum(均番用), 对手胡番sum, 余巡sum, 先到听我方n, 先到听对手n, 先到听流局n]
    # 先到听：分歧点视为「出手时刻」，从该时刻起 replay，第一个达到 0 向听的座位（我方/任一对手/流局前无人）。
    #   只对向听 ≥1 的点统计（听牌点「先到听」无意义——我方已听）。
    cells: dict = collections.defaultdict(lambda: [0] * 12)
    rooms = 0
    errors = collections.Counter()
    for path in batch:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        rooms += 1
        try:
            rounds = list(R.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            # 本局局终事件（从事件流尾部找 round_ended）
            outcome = None  # round_ended event
            for ev in reversed(events):
                if ev.get("type") == WIN:
                    outcome = ev
                    break
            # 局终字段：data.draw / data.fan；winner = round_ended.seat（胡牌家座位）
            o_is_draw = False
            o_winner = None
            o_fan = 0
            if outcome is not None:
                odata = outcome.get("data") or {}
                o_is_draw = bool(odata.get("draw"))
                o_fan = int(odata.get("fan") or 0)
                w = outcome.get("seat")
                if isinstance(w, int) and 0 <= w < 4:
                    o_winner = w
            draws = [0, 0, 0, 0]
            for idx, ev in enumerate(events):
                et = ev.get("type")
                seat = ev.get("seat")
                if not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et == DRAWN:
                    draws[seat] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et != DISCARDED or ids[seat] != OUR:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                # 我方出牌点：标注 + 量后果
                if not state.opened:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                tile = R._tile_of(ev.get("tile"))  # noqa: SLF001
                if tile is None:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                try:
                    sit = state.situation_for(seat, phase=PHASE_DRAW, drawn=None)
                    acts = legal_actions(sit)
                    chosen = decider.choose(sit, acts, budget_ms=600)
                except Exception:  # noqa: BLE001
                    errors["choose"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                divergent = chosen is None or chosen.tile != tile
                hand_counts = list(state.seats[seat].hand)
                meld_n = len(state.seats[seat].melds)
                try:
                    sh0 = sh.shanten_any(hand_counts, meld_n)
                except Exception:  # noqa: BLE001
                    errors["shanten"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if sh0 is None or sh0 < 0:
                    errors["shanten_neg"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                god_n = hand_counts[tiles.GOD]
                tb, sb, gb = turn_bucket(draws[seat]), sh_bucket(sh0), god_bucket(god_n)
                agree = "分歧" if divergent else "一致"
                key = (tb, sb, gb, agree)
                c = cells[key]
                c[0] += 1

                # ---- 后果：从当前点到局终 ----
                # 余巡 + 先到听：从当前事件 index+1 起向前扫（用 deepcopy 的 state 副本，
                # 不污染主循环 state）
                import copy as _copy
                st2 = _copy.deepcopy(state)
                our_remaining_draws = 0
                first_tenpai = None  # "us" / "opp" / None
                for ev2 in events[idx + 1:]:
                    et2 = ev2.get("type")
                    seat2 = ev2.get("seat")
                    if et2 == WIN:
                        break
                    if not isinstance(seat2, int) or not (0 <= seat2 < 4):
                        try:
                            R.apply_event(st2, ev2)
                        except Exception:  # noqa: BLE001
                            break
                        continue
                    if et2 == DRAWN:
                        if seat2 == seat:
                            our_remaining_draws += 1
                        # 摸牌后检查该座位是否到听（只对向听 ≥1 的起点统计先到听）
                        if first_tenpai is None and sh0 >= 1:
                            ss = st2.seats[seat2]
                            try:
                                v = sh.shanten_any(list(ss.hand), len(ss.melds))
                            except Exception:  # noqa: BLE001
                                v = -1
                            if v == 0:
                                first_tenpai = "us" if seat2 == seat else "opp"
                    try:
                        R.apply_event(st2, ev2)
                    except Exception:  # noqa: BLE001
                        break
                # 结局入桶
                if o_is_draw or o_winner is None:
                    c[3] += 1
                elif o_winner == seat:
                    c[1] += 1
                    c[6] += o_fan
                else:
                    c[2] += 1
                    c[7] += o_fan
                c[8] += our_remaining_draws
                if sh0 >= 1:  # 只对向听 ≥1 的点统计先到听
                    if first_tenpai == "us":
                        c[9] += 1
                    elif first_tenpai == "opp":
                        c[10] += 1
                    else:
                        c[11] += 1

                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
    payload = {
        "rooms": rooms,
        "cells": {"|".join(k): v for k, v in cells.items()},
        "errors": dict(errors),
    }
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分歧点后果对拍（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=200)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    chunk_dir = ROOT / "agent" / "out" / "div-outcome-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    print(f"事件流 {len(files)} 房 / {n_chunks} 块 -> {chunk_dir}", flush=True)

    decider = versions.build("v6", Mode.QUALIFIER)
    print(f"决策器: {decider.name}", flush=True)

    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path, decider)
        print(f"chunk {ci:04d}: {rooms} rooms", flush=True)

    # 合并
    cells: dict = collections.defaultdict(lambda: [0] * 12)
    total_rooms = 0
    err_all = collections.Counter()
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["cells"].items():
            m = cells[k]
            for i in range(len(v)):
                m[i] += v[i]
        for k, v in d.get("errors", {}).items():
            err_all[k] += v

    print(f"\nrooms={total_rooms} errors={dict(err_all)}")
    # 自检：一致/分歧率
    tot_div = sum(v[0] for k, v in cells.items() if k.endswith("|分歧"))
    tot_agr = sum(v[0] for k, v in cells.items() if k.endswith("|一致"))
    tot = tot_div + tot_agr
    if tot:
        print(f"自检：分歧率 = {tot_div}/{tot} = {tot_div / tot * 100:.1f}%（1b v2 参考值 19.5%）")

    print(f"\n== 分歧 vs 一致 结局对拍（巡目段 × 向听 × 财神 × 一致性）==")
    hdr = (f"{'巡目':<6}{'向听':<7}{'财神':<6}{'一致':<5}{'n':>7}{'我方胡%':>8}{'对手胡%':>8}"
           f"{'流局%':>7}{'我方均番':>9}{'对手均番':>9}{'余巡':>6}{'先到听我%':>9}{'先到听对%':>9}")
    print(hdr)
    for k in sorted(cells):
        tb, sb, gb, agree = k.split("|")
        n, uw, ow, dr, _uws, _ows, uwf, owf, rem, ftu, fto, ftd = cells[k]
        if not n:
            continue
        uw_pct = uw / n * 100
        ow_pct = ow / n * 100
        dr_pct = dr / n * 100
        uw_fan = uwf / uw if uw else 0.0
        ow_fan = owf / ow if ow else 0.0
        ft_base = ftu + fto + ftd
        ftu_pct = ftu / ft_base * 100 if ft_base else 0.0
        fto_pct = fto / ft_base * 100 if ft_base else 0.0
        print(f"{tb:<6}{sb:<7}{gb:<6}{agree:<5}{n:>7}{uw_pct:>7.1f}%{ow_pct:>7.1f}%"
              f"{dr_pct:>6.1f}%{uw_fan:>9.2f}{ow_fan:>9.2f}{rem / n:>6.1f}{ftu_pct:>8.1f}%{fto_pct:>8.1f}%")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
