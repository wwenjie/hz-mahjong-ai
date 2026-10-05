#!/usr/bin/env python
"""爆头/弃胡路径**漏斗测量**（A 2026-10-06 00:05 交办 [待C]，只读、零平台请求）。

**问题**：我方爆头率只有强 bot 的 1/3。A 00:05① 读代码定位到「弃胡求爆头」窄路
（`_choose_win_or_piao`，要同时满足三条件才弃胡），结论：**不是调参问题，是覆盖率问题**——
瓶颈在「(A) 很少有机会」还是「(B) 有机会但判据太严」必须先量出来。

**漏斗口径**（与 A 00:05② 表逐行对应，分母逐层写出）：

| 层 | 定义 | 记什么 |
|---|---|---|
| L0 | `legal_actions` 含 HU 的点 | 能胡点总数 |
| L1 | 其中 `_piao_candidate` 已非 None | 已在飘（不在本漏斗内） |
| L2 | 其中 `_baotou_by_discard` 有解 | 有机会（条件 2 满足） |
| L3 | L2 中 `survival > threshold` | 实际弃胡 |
| L4 | L2 中 `survival ≤ threshold` | 有机会但被挡掉 |

**决策点 = 我方座位 tile_discarded 且当时 legal_actions 含 HU**（v6 = 线上决策器、1b v2 / 后果对拍同款）。
对每个 L0 点用**只读复刻**（直接调 decider 的 `_piao_candidate`/`_baotou_by_discard`/`_piao_threshold`/`_risks`）
重算三条件——不构造 Action、不改决策路径，纯测量。

**分桶**（A 00:05②）：财神数（0 / 1 / ≥2）× 巡目（1-5 / 6-10 / ≥11）× 当前番数（1 / 2 / ≥4）。
番数 = `compute_fan(能胡的那手)`（与生产 `_fan_of` 同源）。

**仪表自检**（A 00:05④ 强制）：探针判定的「能胡点」上，生产 `v6.choose()` 在该局面必须返回 HU 类动作
（HU 直接胡 / DISCARD 弃胡求爆头都算「认出了能胡」）；若返回 PASS/其他 ⇒ 记 `selfcheck_mismatch`
（只读复刻与生产不一致，口径作废）。一致性应 ≈100%。

**预登记分叉判据**（A 00:05③，出数后不重新讨论口径）：
- L2/L0 <10% ⇒ 瓶颈 (A) 很少有机会 ⇒ 机制件走做牌路径/排序键，不动阈值；
- L2/L0 高但 L3/L2 <30% ⇒ 瓶颈 (B) 判据太严 ⇒ 改 `_piao_threshold` 结构（路径②，非缩系数）；
- 两者都不低但爆头率仍 1/3 ⇒ 瓶颈在弃胡之后 ⇒ 另开「弃胡后成功率」诊断。

**成本**：每 L0 点 ≈ 一次 choose（自检）+ 三条件复算（cheap）——L0 点占出牌点 ~8-15%（胡率口径），
800 房 ≈ 800×8.7×0.12 ≈ 830 个 L0 点，单核 <10min。分块幂等落盘
`agent/out/baotou-funnel-chunks/cs<N>-n<M>/`，DONE 收尾。

用法：`.venv/bin/python agent/verify/baotou_funnel_probe.py [--rooms N] [--chunk-size N] [--merge-only]`
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


def turn_bucket(n: int) -> str:
    return "1-5" if n <= 5 else ("6-10" if n <= 10 else "11+")


def god_bucket(g: int) -> str:
    return "0" if g == 0 else ("1" if g == 1 else "2+")


def fan_bucket(f: int) -> str:
    return "1" if f <= 1 else ("2" if f == 2 else "4+")


def process_batch(batch, chunk_path, decider):
    # 每桶（财神×巡目×番数）：[L0, L1, L2, L3, L4]
    cells: dict = collections.defaultdict(lambda: [0] * 5)
    rooms = 0
    errors = collections.Counter()
    selfcheck_total = 0
    selfcheck_ok = 0
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
            for ev in events:
                et = ev.get("type")
                seat = ev.get("seat")
                if not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                # 只处理我方**摸牌点**（HU 只在摸牌后给出，_choose_win_or_piao 在此运行）；
                # 其余事件 apply 后继续
                if et != DRAWN or ids[seat] != OUR:
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
                # 先 apply 摸牌（手牌变 14 张），再构造 drawn 局面
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                try:
                    sit = state.situation_for(seat, phase=PHASE_DRAW, drawn=tile)
                    acts = legal_actions(sit)
                except Exception:  # noqa: BLE001
                    errors["situation"] += 1
                    continue
                if not any(a.kind == HU for a in acts):
                    # 非能胡点：继续
                    continue
                # ---- L0：能胡点 ----
                # 生产决策自检：choose 必须认出能胡（返回 HU 或 DISCARD-弃胡求爆头）
                selfcheck_total += 1
                try:
                    chosen = decider.choose(sit, acts, budget_ms=600)
                    if chosen is not None and chosen.kind in (HU, "discard"):
                        selfcheck_ok += 1
                    else:
                        errors["selfcheck_mismatch"] += 1
                except Exception:  # noqa: BLE001
                    errors["choose"] += 1
                # 漏斗三条件复算（drawn 已知）
                try:
                    hand = sit.hand
                    waiting = list(hand.counts)
                    waiting[tile] -= 1
                    current = decider._fan_of(sit, waiting, tile)  # noqa: SLF001
                    if not current.hu:
                        # 能胡但 fan 校验失败（边缘）：不算 L0 的可操作点
                        errors["fan_not_hu"] += 1
                        continue
                    # 分桶键
                    god_n = int(hand.god_count)
                    turn = turn_bucket(state.draws)  # 全局摸牌数近似巡目
                    fb = fan_bucket(int(current.fan))
                    key = f"{god_bucket(god_n)}|{turn}|{fb}"
                    c = cells[key]
                    c[0] += 1  # L0
                    # L1：已在飘（_piao_candidate 非 None ⇒ 不在本漏斗）
                    if decider._piao_candidate(sit, waiting, tile) is not None:  # noqa: SLF001
                        c[1] += 1
                        continue
                    # L2：_baotou_by_discard 有解（chase_baotou 默认 True）
                    target = (
                        decider._baotou_by_discard(sit, waiting, tile)  # noqa: SLF001
                        if decider.config.chase_baotou
                        else None
                    )
                    if target is None:
                        # 无机会：L2 不进
                        continue
                    c[2] += 1  # L2 有机会
                    # L3/L4：survival vs threshold
                    risks = decider._risks(sit)  # noqa: SLF001
                    survival = risk.lap_survival(risks)
                    gain = float(decider._win_points(current.fan * 2, sit))  # noqa: SLF001
                    loss = float(decider._loss_points(sit))  # noqa: SLF001
                    threshold = decider._piao_threshold(gain, loss, sit)  # noqa: SLF001
                    if survival > threshold and not sit.is_restricted:
                        c[3] += 1  # L3 实际弃胡
                    else:
                        c[4] += 1  # L4 有机会但被挡
                except Exception:  # noqa: BLE001
                    errors["funnel"] += 1
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "rooms": rooms,
                "cells": {k: v for k, v in cells.items()},
                "errors": dict(errors),
                "selfcheck_total": selfcheck_total,
                "selfcheck_ok": selfcheck_ok,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(chunk_path)
    return rooms


def merge_report(chunk_dir: Path) -> None:
    cells: dict = collections.defaultdict(lambda: [0] * 5)
    total_rooms = 0
    err_all = collections.Counter()
    sc_total = 0
    sc_ok = 0
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d.get("errors", {}).items():
            err_all[k] += v
        sc_total += d.get("selfcheck_total", 0)
        sc_ok += d.get("selfcheck_ok", 0)
        for k, v in d["cells"].items():
            m = cells[k]
            for i in range(len(v)):
                m[i] += v[i]

    L0 = sum(v[0] for v in cells.values())
    L1 = sum(v[1] for v in cells.values())
    L2 = sum(v[2] for v in cells.values())
    L3 = sum(v[3] for v in cells.values())
    L4 = sum(v[4] for v in cells.values())

    print(f"rooms={total_rooms}  errors={dict(err_all)}", flush=True)
    print(
        f"自检：能胡点 choose 认出率 = {sc_ok}/{sc_total} = "
        f"{(sc_ok / sc_total * 100) if sc_total else 0:.1f}%（应 ≈100%；低 ⇒ 只读复刻口径作废）",
        flush=True,
    )
    print(flush=True)
    print("== 漏斗（全体能胡点）==", flush=True)
    print(f"  L0 能胡点              = {L0}", flush=True)
    print(f"  L1 已在飘（出漏斗）    = {L1}  ({L1 / L0 * 100 if L0 else 0:.1f}% of L0)", flush=True)
    in_funnel = L0 - L1
    print(f"  漏斗内（L0−L1）        = {in_funnel}", flush=True)
    print(
        f"  L2 有机会（弃牌可成爆头）= {L2}  ({L2 / in_funnel * 100 if in_funnel else 0:.1f}% of 漏斗内）",
        flush=True,
    )
    print(
        f"  L3 实际弃胡（survival>阈）= {L3}  ({L3 / L2 * 100 if L2 else 0:.1f}% of L2)",
        flush=True,
    )
    print(
        f"  L4 有机会但被挡（≤阈）   = {L4}  ({L4 / L2 * 100 if L2 else 0:.1f}% of L2)",
        flush=True,
    )
    print(flush=True)
    print("== 预登记分叉判据读数 ==", flush=True)
    l2_rate = L2 / in_funnel if in_funnel else 0
    l3_rate = L3 / L2 if L2 else 0
    print(f"  L2/漏斗内 = {l2_rate:.1%}  （<10% ⇒ 瓶颈A 很少有机会）", flush=True)
    print(f"  L3/L2    = {l3_rate:.1%}  （<30% ⇒ 瓶颈B 判据太严）", flush=True)
    if l2_rate < 0.10:
        print("  ⇒ 判读：瓶颈(A) 很少有机会 ⇒ 机制件走做牌路径/排序键，不动阈值", flush=True)
    elif l3_rate < 0.30:
        print("  ⇒ 判读：瓶颈(B) 有机会但判据太严 ⇒ 改 _piao_threshold 结构（路径②）", flush=True)
    else:
        print("  ⇒ 判读：两者都不低 ⇒ 瓶颈在弃胡之后，另开『弃胡后成功率』诊断", flush=True)
    print(flush=True)
    print("== 分桶明细（财神|巡目|番数）：L0/L1/L2/L3/L4 ==", flush=True)
    for k in sorted(cells):
        v = cells[k]
        g, t, f = k.split("|")
        print(
            f"  财神{g:>2} 巡{t:>3} 番{f:>2} | L0={v[0]:>4} L1={v[1]:>3} L2={v[2]:>3} "
            f"L3={v[3]:>3} L4={v[4]:>3}"
            + (f"  L3/L2={v[3] / v[2] * 100:.0f}%" if v[2] else ""),
            flush=True,
        )
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="爆头/弃胡漏斗测量（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        import random

        random.seed(args.seed)
        files = random.sample(files, min(args.rooms, len(files)))

    chunk_dir = ROOT / "agent" / "out" / "baotou-funnel-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    print(f"事件流 {len(files)} 房 / {n_chunks} 块 -> {chunk_dir}", flush=True)

    if args.merge_only:
        merge_report(chunk_dir)
        return 0

    decider = versions.build("v6", Mode.QUALIFIER)
    print(f"决策器: {decider.name}  chase_baotou={decider.config.chase_baotou}", flush=True)

    for ci in range(n_chunks):
        if ci % args.shards != args.shard:
            continue
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path, decider)
        print(f"chunk {ci:04d}: {rooms} rooms", flush=True)

    done = sum(1 for _ in chunk_dir.glob("chunk-*.json"))
    if done >= n_chunks and args.shards == 1:
        merge_report(chunk_dir)
    else:
        print(f"shard {args.shard} 完成；已落盘 {done}/{n_chunks} 块（全部齐后用 --merge-only 合并）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
