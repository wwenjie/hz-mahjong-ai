"""G1 真机 replay 反事实探针 —— A 2026-10-04 02:15 交办（[待C]，不占 A/B 通道、不上平台）。

问题：财神听口缺口（真机：我们 1 财神听口 4.53 种 vs 头部 bot 6.08）在自对弈场不存在
（v5 自对弈已 5.571），且线性加性权重不咬合（boost=2.0 只动 +0.02%）。
⇒ 只有真机数据能回答「换个排序键能不能把 4.53 拉上去」。

方法：在真机事件流的我方「打出后 0 向听（听牌）且打出后手持财神 ≥1」决策点上：
  (i) 实际所选的听口种数 len(winning_draws)（A 口径，与 god_conversion_gap.py 同源）；
  (ii) 反事实：同决策点所有「打出后仍 0 向听」的候选中，按**字典序**
       （先比听口种数、再比听口剩余可见张数、牌种码升序兜底确定性）重排会选的听口种数。
  同时报：反事实改判的决策点占比（A 要求：<5% 则缺口集中少数点，与 M1 尾部结论同型）。

预登记门槛（A 02:15 原文）：反事实听口种数 4.53 → **≥5.5（+25%）** 才起真机限臂；
否则判「加性/字典序两种形式都推不动」，财神听口轴关闭。

口径钉死（与 god_conversion_gap.py 完全同源，使「实际」列应复现 4.53 作为仪表自检）：
  - 决策点 = 我方座位 tile_discarded，打出后 shanten_any==0 且打出后手 counts[GOD]≥1。
  - 听口种数 = len(winning_draws(打出后手, meld_count))；可见张数 = live_copies 同 god 探针。
  - 财神分桶：1张 / ≥2张（0 张不在本探针范围）。
  - 数据源 = data/auto_sessions/*/events/*.json 全量（与 4.53 同批，不做 arm 过滤——
    4.53 就是这批的读数；反事实问的是「同一批点上换排序键」）。

自检：合并后「实际」列 1 财神桶应 ≈4.53（god 探针 23:12 读数）；偏差 >0.1 则先查仪表再报数。

抗回收：分块幂等落盘 agent/out/g1-godwait-cf-chunks/cs<N>-n<M>/，DONE 收尾。
成本预估：json parse + replay 主导（只有听牌+财神点做 winning_draws），单核 ~30-45min。

用法：`.venv/bin/python agent/verify/g1_godwait_counterfactual.py [--rooms N] [--chunk-size N]`
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

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules import win as win_mod  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"


def process_batch(batch, chunk_path):
    # 每桶 [n, actual_kinds_sum, cf_kinds_sum, actual_copies_sum, cf_copies_sum,
    #       changed_n, cands_sum, kinds_ge2_diff_n]
    # kinds_ge2_diff_n：反事实听口种数比实际多 ≥2 的点数（看幅度分布右尾）
    agg: dict = collections.defaultdict(lambda: [0] * 8)
    rooms = 0
    skipped = collections.Counter()
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
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            for ev in events:
                et = ev.get("type")
                seat = ev.get("seat")
                if not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                is_target = et == DISCARDED and ids[seat] == OUR
                if not is_target:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                # 我方打出点
                before = list(state.seats[seat].hand)
                meld_n = len(state.seats[seat].melds)
                if sum(before) != tiles.HAND_SIZE + 1 - tiles.MELD_SLOTS * meld_n:
                    skipped["hand_size"] += 1
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                after_actual = list(state.seats[seat].hand)
                diff = [t for t in range(tiles.TILE_KINDS) if before[t] - after_actual[t] == 1]
                if len(diff) != 1:
                    skipped["diff"] += 1
                    continue
                gods = after_actual[tiles.GOD]
                if gods < 1:
                    continue
                memo: dict = {}
                try:
                    s0 = shanten_mod.shanten(after_actual, meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    skipped["shanten"] += 1
                    continue
                if s0 != 0:
                    continue
                # 可见计数（与 god 探针同口径：本手 + 全部副露 + 全部弃牌）
                visible = shanten_mod.visible_counts(
                    after_actual,
                    [meld.tiles for other in state.seats for meld in other.melds],
                    [list(other.discards) for other in state.seats],
                )

                def kinds_copies(hand_counts):
                    wl = win_mod.winning_draws(hand_counts, meld_n)
                    kinds = len(wl)
                    copies = sum(max(0, tiles.COPIES_PER_KIND - visible[t]) for t in wl)
                    return kinds, copies

                actual_k, actual_c = kinds_copies(after_actual)
                # 反事实候选：全体打出后仍 0 向听的候选
                best = None  # (kinds, copies, -tile) 取 max ⇒ 牌种码小者兜底
                n_cands = 0
                for t, c in enumerate(before):
                    if c <= 0:
                        continue
                    cand = list(before)
                    cand[t] -= 1
                    try:
                        if shanten_mod.shanten(cand, meld_n, memo=memo) != 0:
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                    n_cands += 1
                    k, cp = kinds_copies(cand)
                    key = (k, cp, -t)
                    if best is None or key > best:
                        best = key
                if best is None:
                    skipped["no_cand"] += 1
                    continue
                cf_k, cf_c = best[0], best[1]
                bucket = "1张" if gods == 1 else "≥2张"
                a = agg[bucket]
                a[0] += 1
                a[1] += actual_k
                a[2] += cf_k
                a[3] += actual_c
                a[4] += cf_c
                a[5] += 1 if cf_k != actual_k or cf_c != actual_c else 0
                a[6] += n_cands
                a[7] += 1 if cf_k - actual_k >= 2 else 0
    payload = {"rooms": rooms, "agg": dict(agg), "skipped": dict(skipped)}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G1 真机 replay 反事实探针（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=200)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    chunk_dir = ROOT / "agent" / "out" / "g1-godwait-cf-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    print(f"事件流 {len(files)} 房 / {n_chunks} 块 -> {chunk_dir}", flush=True)

    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path)
        print(f"chunk {ci:04d}: {rooms} rooms", flush=True)

    merged: dict = collections.defaultdict(lambda: [0] * 8)
    total_rooms = 0
    skip_all = collections.Counter()
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["agg"].items():
            m = merged[k]
            for i in range(len(v)):
                m[i] += v[i]
        for k, v in d.get("skipped", {}).items():
            skip_all[k] += v

    print(f"\nrooms={total_rooms} skipped={dict(skip_all)}")
    print(f"{'财神':>4} {'n':>6} {'实际种数':>8} {'反事实种数':>10} {'实际张数':>8} "
          f"{'反事实张数':>10} {'改判占比':>8} {'均候选数':>8} {'+≥2种点数':>9}")
    for k in sorted(merged):
        n, aks, cks, acs, ccs, ch, cs, ge2 = merged[k]
        if not n:
            continue
        print(f"{k:>4} {n:>6} {aks/n:>8.2f} {cks/n:>10.2f} {acs/n:>8.2f} "
              f"{ccs/n:>10.2f} {ch/n*100:>7.1f}% {cs/n:>8.2f} {ge2:>9}")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
