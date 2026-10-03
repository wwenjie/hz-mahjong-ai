"""S4-measure：好型率（perfect n-away）与我们的选择分歧 —— A 2026-10-03 21:25 分工表 C 项。

问题（survey #3 的前测，不改任何档位）：在 2 向听同向听层，对每个决策点、每个候选
（打出后向听 == s_actual）计算**好型率**：
  对每个同向听候选 c（打出某张后的手牌）：
    枚举全部进张 t（剩余可见 >0 且使向听从 s_actual 降 1 的摸牌）；
    对每张进张，模拟摸入后再枚举打出，问「能否达到 0 向听（听牌）」；
    若能，再问此时听口是否含两面搭子（好型听）；
    perfect_rate(c) = 能落到「好型听」的进张剩余张数 / 全部进张剩余张数。
然后看我们实际所选候选的 perfect_rate 在同向听候选集合里的 rank/差距，
与对手同口径对比。若我们系统性选到低 perfect_rate 候选 ⇒ 好型率是有效第二排序键；
若双方都差不多 ⇒ 换比较量这条路也要重估。

口径钉死（与 C31-R / gap 批同源，保证可比）：
  - 数据源 = data/auto_sessions/*/events/*.json，arm_map 过滤 v5 臂（同一批房）。
  - 决策点 = 我方座位 tile_discarded，且打出后向听 == 2（M1 靶子层，同 C31-R）。
  - 好型定义：听牌形中去掉雀头后，剩余搭子含至少一个两面（嵌/边/单钓/对碰=非好型）。
    财神在手一律算好型（财神可补成任意搭子，听口至少等价两面）。
  - 分桶键与 gap 批同构：(grp, god, n_bucket)，grp ∈ {our, opp}，god = 打出前手含财神。

抗回收：分块幂等落盘 agent/out/s4-perfect-v5-chunks/cs<N>-n<M>/，DONE 标记收尾。
单核 ~1h 预估（2 向听点 × 候选 × 进张 × 打出枚举，memo 每决策点用完即弃）。

用法：`.venv/bin/python agent/verify/s4_perfect_rate_probe.py [--rooms N] [--chunk-size N]`
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
from majiang.rules import win as win_mod  # noqa: E402  # (保留引用，听口判定备用)
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
DISCARDED = "tile_discarded"
DRAWN = "tile_drawn"
COPIES = tiles.COPIES_PER_KIND


def visible_counts(state) -> list[int]:
    vis = [0] * tiles.TILE_KINDS
    for seat in state.seats:
        for t in seat.discards:
            vis[t] += 1
        for m in seat.melds:
            for t in m.tiles:
                vis[t] += 1
    return vis


def _has_ryanmen(counts: list[int], meld_n: int) -> bool:
    """听牌形（0 向听）是否含两面搭子：去掉雀头后存在同花色相邻对 (t,t+1) 各≥1。

    粗口径但稳定：财神在手一律算好型（财神可补成任意搭子）。
    """
    c = list(counts)
    if c[tiles.GOD] > 0:
        return True
    suits = [(0, 9), (9, 18), (18, 27)]
    for head in range(tiles.TILE_KINDS):
        if c[head] < 2:
            continue
        c[head] -= 2
        try:
            for lo, hi in suits:
                for t in range(lo, hi - 1):
                    if c[t] >= 1 and c[t + 1] >= 1:
                        return True
        finally:
            c[head] += 2
    return False


def perfect_rate(after: list[int], meld_n: int, s: int, vis: list[int], memo: dict) -> tuple[int, int]:
    """同向听候选 after（打出后手牌）的好型率 = 能落到好型听的进张张数 / 全部进张张数。

    返回 (perfect_count, total_count)。进张 = 摸入后向听 < s 的牌种（按剩余可见张数加权）。
    """
    perfect = 0
    total = 0
    for t in range(tiles.TILE_KINDS):
        remaining = COPIES - vis[t] - after[t]
        if remaining <= 0:
            continue
        after[t] += 1
        try:
            sh = shanten_mod.shanten_any(after, meld_n, memo=memo)
        except Exception:  # noqa: BLE001
            after[t] -= 1
            continue
        if sh is None or sh >= s:
            after[t] -= 1
            continue
        total += remaining
        reached_tenpai = False
        for d in range(tiles.TILE_KINDS):
            if after[d] <= 0:
                continue
            after[d] -= 1
            try:
                sh2 = shanten_mod.shanten(list(after), meld_n, memo=memo)
            except Exception:  # noqa: BLE001
                after[d] += 1
                continue
            if sh2 == 0 and _has_ryanmen(after, meld_n):
                reached_tenpai = True
                after[d] += 1
                break
            after[d] += 1
        if reached_tenpai:
            perfect += remaining
        after[t] -= 1
    return perfect, total


def process_batch(batch, chunk_path):
    # 每桶 [n, perf_sum, tot_sum, chosen_rank_sum, chosen_perf_sum, chosen_tot_sum, best_perf_sum]
    # chosen_rank：按 perfect_rate 排序，实际所选在同向听候选中排第几（0=最优）
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    rooms = 0
    skipped = collections.Counter()
    for path in batch:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4:
            continue
        rooms += 1
        try:
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            draw_idx = [0, 0, 0, 0]
            for ev in events:
                et = ev.get("type")
                seat = ev.get("seat")
                if not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et == DRAWN:
                    draw_idx[seat] += 1
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et != DISCARDED:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                before = list(state.seats[seat].hand)
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                after_actual = state.seats[seat].hand
                diff = [t for t in range(tiles.TILE_KINDS) if before[t] - after_actual[t] == 1]
                if len(diff) != 1:
                    continue
                meld_n = len(state.seats[seat].melds)
                if sum(before) != tiles.HAND_SIZE + 1 - tiles.MELD_SLOTS * meld_n:
                    skipped["hand_size"] += 1
                    continue
                memo: dict = {}
                try:
                    s_actual = shanten_mod.shanten(list(after_actual), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    skipped["shanten"] += 1
                    continue
                if s_actual != 2:
                    continue
                vis = visible_counts(state)
                vis[diff[0]] -= 1
                # 全体同向听候选（含实际所选）
                cands: list[tuple[int, int, int]] = []  # (tile, perfect, total)
                for t, c in enumerate(before):
                    if c <= 0:
                        continue
                    cand = list(before)
                    cand[t] -= 1
                    try:
                        s_c = shanten_mod.shanten(cand, meld_n, memo=memo)
                    except Exception:  # noqa: BLE001
                        continue
                    if s_c != s_actual:
                        continue
                    p, tot = perfect_rate(cand, meld_n, s_actual, vis, memo)
                    cands.append((t, p, tot))
                if len(cands) < 2:
                    skipped["lt2_cand"] += 1
                    continue
                # 按 perfect_rate 降序排（total=0 的候选 rate=0，排到尾）
                cands.sort(key=lambda x: (x[1] / x[2]) if x[2] else 0.0, reverse=True)
                rank = next((i for i, (t, _, _) in enumerate(cands) if t == diff[0]), len(cands))
                _, cp, ct = next(((t, p, tot) for t, p, tot in cands if t == diff[0]), (0, 0, 0))
                bp, _bt = cands[0][1], cands[0][2]
                grp = "our" if ids[seat] == OUR else "opp"
                n = draw_idx[seat]
                n_bucket = "n≤4" if n <= 4 else ("n5-8" if n <= 8 else ("n9-12" if n <= 12 else "n≥13"))
                god = "god" if before[tiles.GOD] > 0 else "nogod"
                key = f"{grp}||{god}||{n_bucket}"
                a = agg[key]
                a[0] += 1
                a[1] += sum(p for _, p, _ in cands)
                a[2] += sum(tot for _, _, tot in cands)
                a[3] += rank
                a[4] += cp
                a[5] += ct
                a[6] += bp
    agg_ser = {k: v for k, v in agg.items()}
    payload = {"rooms": rooms, "agg": agg_ser, "skipped": dict(skipped)}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S4-measure 好型率分歧前测（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--stride-start", type=int, default=0,
                    help="采样起始偏移：files[stride::step]，保证与旧采样批不重叠")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--arm-map", default=None)
    ap.add_argument("--arm", default=None)
    args = ap.parse_args(argv)

    arm_of = {}
    if args.arm_map:
        with open(args.arm_map, encoding="utf-8") as f:
            arm_of = json.load(f).get("map", {})

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if arm_of:
        keep = []
        for p in files:
            gid = Path(p).stem
            arm = arm_of.get(gid)
            if arm is None or arm == "mixed":
                continue
            if args.arm and arm != args.arm:
                continue
            keep.append(p)
        files = keep
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        start = args.stride_start % step
        files = files[start::step][: args.rooms]

    tag = ("-" + args.arm) if args.arm else ""
    stag = f"-s{args.stride_start}" if args.stride_start else ""
    chunk_dir = ROOT / "agent" / "out" / f"s4-perfect{tag}-chunks" / f"cs{args.chunk_size}-n{len(files)}{stag}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size

    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path)
        print(f"chunk {ci:04d}: {rooms} rooms -> {chunk_path}", flush=True)

    # 合并
    merged: dict = collections.defaultdict(lambda: [0] * 7)
    total_rooms = 0
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["agg"].items():
            m = merged[k]
            for i in range(7):
                m[i] += v[i]
    print(f"rooms={total_rooms}")
    print(f"{'key':>22} {'n':>6} {'avg_rank':>8} {'chosen_pr':>9} {'best_pr':>8} {'pool_pr':>8}")
    for k in sorted(merged):
        n, ps, ts, rs, cps, cts, bps = merged[k]
        if not n:
            continue
        print(f"{k:>22} {n:>6} {rs/n:>8.2f} {cps/cts if cts else 0:>9.3f} "
              f"{bps/ts if ts else 0:>8.3f} {ps/ts if ts else 0:>8.3f}")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
