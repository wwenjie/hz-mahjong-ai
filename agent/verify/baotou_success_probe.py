#!/usr/bin/env python
"""弃胡后**成功率**测量（A 2026-10-06 00:05③ 预登记的第三诊断：「瓶颈在弃胡之后」）。

**承接**：`baotou_funnel_probe.py` 已量出漏斗——L0=1523 能胡点、L2=248 有机会（16.3%）、
L3=162 实际弃胡（65.3%）、L4=86 有机会但被挡。两判据都不低 ⇒ 瓶颈不在「机会少」也不在「判据严」，
而在**弃胡之后能否真的爆头/上链**。本探针量这最后一段。

**口径**（只读 replay，真实结局，零平台请求）：
对每个 L2 点（有爆头机会的能胡点），从该点**续看本局真实事件流到 round_ended**：
- 我方结局：`baotou`（最终真爆头）/ `piao`（财飘，链≥1 或 piao_count≥1）/ `hu_plain`（普通胡，非爆头非飘）/ `no_hu`（未胡）
- 对手结局：`opp_hu`（任一对手胡）/ `draw`（流局）
- 弃胡后**巡数代价**：从该点到局终的巡数（谁先到）

**分组**（回答「弃胡值不值」）：
- **L3 组（实际弃胡）**：弃胡后的真实结局分布——成功率 = (baotou+piao+hu_plain)/L3；
- **L4 组（被挡）**：若当时弃胡会怎样**无法直接看**（真实历史里我们胡了）——但可看「这些点我们普通胡了，番数 vs 若爆头的番数」（`fan_now` vs `fan*2`），量「被挡的机会成本」。

**分桶**：财神数 × 番数（巡目意义不大——能胡点 96% 在 11+ 巡）。

产物：`agent/out/baotou-success-chunks/cs<N>-n<M>/`，DONE 收尾。

用法：`.venv/bin/python agent/verify/baotou_success_probe.py [--rooms N] [--chunk-size N]`
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

from majiang.rules.action import HU, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay as R  # noqa: E402
from majiang.strategy import risk  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"
ROUND_ENDED = "round_ended"


def god_bucket(g: int) -> str:
    return "0" if g == 0 else ("1" if g == 1 else "2+")


def fan_bucket(f: int) -> str:
    return "1" if f <= 1 else ("2" if f == 2 else "4+")


def process_batch(batch, chunk_path, decider):
    # 每桶（财神×番数×组[L3/L4]）：[n, baotou, piao, hu_plain, no_hu, opp_hu, draw, fan_now_sum, fan_if_baotou_sum, turns_used_sum]
    cells: dict = collections.defaultdict(lambda: [0] * 10)
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
        me = ids.index(OUR)
        try:
            rounds = list(R.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            # 先找本局 round_ended（结局）
            outcome = None
            for ev in reversed(events):
                if ev.get("type") == ROUND_ENDED:
                    outcome = ev
                    break
            o_draw = False
            o_winner = None
            o_fan = 0
            o_detail = ""
            if outcome is not None:
                od = outcome.get("data") or {}
                o_draw = bool(od.get("draw"))
                o_fan = int(od.get("fan") or 0)
                o_detail = str(od.get("detail") or "")
                w = outcome.get("seat")
                if isinstance(w, int) and 0 <= w < 4:
                    o_winner = w

            # 逐个我方摸牌点扫，命中 L2 就量结局
            for idx, ev in enumerate(events):
                if ev.get("type") != DRAWN or ev.get("seat") != me:
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
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                try:
                    sit = state.situation_for(me, phase=PHASE_DRAW, drawn=tile)
                    acts = legal_actions(sit)
                except Exception:  # noqa: BLE001
                    errors["situation"] += 1
                    continue
                if not any(a.kind == HU for a in acts):
                    continue
                # 漏斗三条件复算到 L2
                try:
                    hand = sit.hand
                    waiting = list(hand.counts)
                    waiting[tile] -= 1
                    current = decider._fan_of(sit, waiting, tile)  # noqa: SLF001
                    if not current.hu:
                        continue
                    if decider._piao_candidate(sit, waiting, tile) is not None:  # noqa: SLF001
                        continue
                    target = (
                        decider._baotou_by_discard(sit, waiting, tile)  # noqa: SLF001
                        if decider.config.chase_baotou
                        else None
                    )
                    if target is None:
                        continue
                    # 是 L2 点：判 L3/L4
                    risks = decider._risks(sit)  # noqa: SLF001
                    survival = risk.lap_survival(risks)
                    gain = float(decider._win_points(current.fan * 2, sit))  # noqa: SLF001
                    loss = float(decider._loss_points(sit))  # noqa: SLF001
                    threshold = decider._piao_threshold(gain, loss, sit)  # noqa: SLF001
                    gave_up = survival > threshold and not sit.is_restricted
                    grp = "L3" if gave_up else "L4"

                    # 弃胡后巡数代价：数从该点到局终的我方后续摸牌次数
                    turns_used = 0
                    for ev2 in events[idx + 1 :]:
                        if ev2.get("type") == DRAWN and ev2.get("seat") == me:
                            turns_used += 1
                        if ev2.get("type") == ROUND_ENDED:
                            break

                    # 真机下一步动作核对（仪表自检：v6 复演 vs 真机是否一致）
                    # real_act: "hu_now"=下一事件即局终且我方胡（真机直接胡）；
                    #           "discard_target"=真机打了爆头目标牌（真机也弃胡）；
                    #           "discard_other"=真机打了别的牌；"round_end_other"=其他结局
                    real_act = "round_end_other"
                    for ev2 in events[idx + 1 :]:
                        t2 = ev2.get("type")
                        if t2 == ROUND_ENDED:
                            real_act = "hu_now" if ev2.get("seat") == me else "round_end_other"
                            break
                        if t2 == DISCARDED and ev2.get("seat") == me:
                            real_act = "discard_target" if R._tile_of(ev2.get("tile")) == target else "discard_other"  # noqa: SLF001
                            break
                        if t2 in ("peng", "chi", "gang", "pass", "timeout") and ev2.get("seat") == me:
                            # 我方中间动作（碰/吃/杠）不终止寻找，但也非出牌
                            continue

                    # 我方结局分类
                    if o_draw or o_winner is None:
                        my_result = "no_hu"  # 流局
                        opp_hu = 0
                        draw = 1
                    elif o_winner == me:
                        # 我方胡：爆头/财飘/普通
                        if "爆头" in o_detail:
                            my_result = "baotou"
                        elif "财飘" in o_detail or "飘" in o_detail:
                            my_result = "piao"
                        else:
                            my_result = "hu_plain"
                        opp_hu = 0
                        draw = 0
                    else:
                        my_result = "no_hu"
                        opp_hu = 1
                        draw = 0

                    key = f"{god_bucket(int(hand.god_count))}|{fan_bucket(int(current.fan))}|{grp}"
                    c = cells[key]
                    c[0] += 1
                    c[{"baotou": 1, "piao": 2, "hu_plain": 3, "no_hu": 4}[my_result]] += 1
                    c[5] += opp_hu
                    c[6] += draw
                    c[7] += int(current.fan)
                    c[8] += int(current.fan) * 2  # 若爆头（番翻倍）
                    c[9] += turns_used
                    # 真机一致性计数（跨桶汇总到 __real__ 键）
                    rc = cells[f"__real__|{grp}"]
                    rc[0] += 1
                    rc[{"hu_now": 1, "discard_target": 2, "discard_other": 3, "round_end_other": 4}[real_act]] += 1
                except Exception:  # noqa: BLE001
                    errors["funnel"] += 1
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"rooms": rooms, "cells": dict(cells), "errors": dict(errors)}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(chunk_path)
    return rooms


def merge_report(chunk_dir: Path) -> None:
    cells: dict = collections.defaultdict(lambda: [0] * 10)
    total_rooms = 0
    err_all = collections.Counter()
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d.get("errors", {}).items():
            err_all[k] += v
        for k, v in d["cells"].items():
            m = cells[k]
            for i in range(len(v)):
                m[i] += v[i]

    # 汇总 L3 / L4（排除 __real__ 自检键）
    def agg(grp):
        n = ba = pi = hp = nh = oh = dr = fs = fb = tu = 0
        for k, v in cells.items():
            if k.startswith("__real__") or not k.endswith("|" + grp):
                continue
            n += v[0]; ba += v[1]; pi += v[2]; hp += v[3]; nh += v[4]
            oh += v[5]; dr += v[6]; fs += v[7]; fb += v[8]; tu += v[9]
        return n, ba, pi, hp, nh, oh, dr, fs, fb, tu

    print(f"rooms={total_rooms}  errors={dict(err_all)}", flush=True)
    # 真机一致性自检（v6 复演判定 vs 真机实际动作）
    for grp in ("L3", "L4"):
        rc = cells.get(f"__real__|{grp}")
        if rc and rc[0]:
            n = rc[0]
            print(
                f"自检-真机动作[{grp}]：n={n}  真机直接胡={rc[1]}({rc[1]/n*100:.0f}%)  "
                f"真机也弃胡打目标牌={rc[2]}({rc[2]/n*100:.0f}%)  真机打别牌={rc[3]}({rc[3]/n*100:.0f}%)  其他={rc[4]}",
                flush=True,
            )
    print(
        "  （判读：L3 组应高『真机也弃胡』、L4 组应高『真机直接胡』——背离说明真机时代/状态与 v6 复演不一致，结局列仅作参考）",
        flush=True,
    )
    print(flush=True)
    for grp, label in (("L3", "实际弃胡"), ("L4", "有机会但被挡")):
        n, ba, pi, hp, nh, oh, dr, fs, fb, tu = agg(grp)
        if n == 0:
            print(f"== {grp}（{label}）：无样本 ==", flush=True)
            continue
        win = ba + pi + hp
        print(f"== {grp}（{label}）n={n} ==", flush=True)
        print(f"  弃胡后我方结局：爆头 {ba}({ba/n*100:.1f}%)  财飘 {pi}({pi/n*100:.1f}%)  普通胡 {hp}({hp/n*100:.1f}%)  未胡 {nh}({nh/n*100:.1f}%)", flush=True)
        print(f"  我方成功率（最终胡牌）= {win}/{n} = {win/n*100:.1f}%", flush=True)
        print(f"  其中爆头+财飘（链上）= {ba+pi}/{n} = {(ba+pi)/n*100:.1f}%", flush=True)
        print(f"  对手胡 {oh}({oh/n*100:.1f}%)  流局 {dr}({dr/n*100:.1f}%)", flush=True)
        print(f"  弃胡点当时番均 {fs/n:.2f}（若爆头≈{fb/n:.2f}）  弃胡后我方平均再摸 {tu/n:.1f} 巡到局终", flush=True)
        print(flush=True)
    print("== 分桶明细（财神|番数|组）：n 爆头/财飘/普胡/未胡 对胡 流局 ==", flush=True)
    for k in sorted(cells):
        if k.startswith("__real__"):
            continue
        v = cells[k]
        print(
            f"  {k} | n={v[0]:>3}  爆{v[1]} 飘{v[2]} 胡{v[3]} 未{v[4]}  对胡{v[5]} 流{v[6]}",
            flush=True,
        )
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="弃胡后成功率测量（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        import random

        random.seed(args.seed)
        files = random.sample(files, min(args.rooms, len(files)))

    chunk_dir = ROOT / "agent" / "out" / "baotou-success-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    print(f"事件流 {len(files)} 房 / {n_chunks} 块 -> {chunk_dir}", flush=True)

    if args.merge_only:
        merge_report(chunk_dir)
        return 0

    decider = versions.build("v6", Mode.QUALIFIER)
    print(f"决策器: {decider.name}  chase_baotou={decider.config.chase_baotou}", flush=True)

    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path, decider)
        print(f"chunk {ci:04d}: {rooms} rooms", flush=True)

    done = sum(1 for _ in chunk_dir.glob("chunk-*.json"))
    if done >= n_chunks:
        merge_report(chunk_dir)
    else:
        print(f"已落盘 {done}/{n_chunks} 块（全部齐后用 --merge-only 合并）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
