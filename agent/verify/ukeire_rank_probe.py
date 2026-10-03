"""C31-R：同向听层内「实际选择按精确进张的 rank」——A 2026-10-02 18:55 指定诊断。

问题：在 2 向听 × n5-8 层，我们实际打出的牌，在全部同向听候选中按精确进张
（ukeire）排第几？rank=0 表示选了最优。

- 若我们常选 rank 0-1 却仍慢 ⇒ 瓶颈不在进张键，M1 方向要换；
- 若 rank 分布显著劣于对手 ⇒ 瓶颈在排序/估值（支持 M1 类修法）。

口径与 C31（ukeire_choice_probe.py）一致：同向听候选 = 打出后向听 == s_actual 的候选；
ukeire = 使手牌向听下降的牌张数（4 − 全场可见）。新增：
  rank = #{候选 ukeire > u_actual}（严格更优的候选数，0 基）。

分桶聚合（our/opp × 向听 × 财神 × 摸序段）：
  [n, rank_sum, rank0_count, rank_le1_count, rank_ge3_count, gap_sum]

2026-10-03 A 13:12 ⑥：新增 gap_sum —— gap = max_u − u_actual（同向听候选中精确进张
最大值减实际所选牌的进张）。rank 是序数（rank=1 可能只差 1 张），gap 量的是幅度。
判 M1 形态：幅度大 ⇒ 进张键在 2 向听层不是有效目标 ⇒ M1 转目标键；幅度小 ⇒ M1 关闭
（rank 差是统计显著但实践无关的口径伪影）。

抗回收：沿用 C31 的分块幂等落盘（agent/out/c31r-chunks/）。

用法：`.venv/bin/python agent/verify/ukeire_rank_probe.py [--rooms N] [--chunk-size N]`
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
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
DISCARDED = "tile_discarded"
DRAWN = "tile_drawn"
COPIES = 4


def visible_counts(state) -> list[int]:
    vis = [0] * tiles.TILE_KINDS
    for seat in state.seats:
        for t in seat.discards:
            vis[t] += 1
        for m in seat.melds:
            for t in m.tiles:
                vis[t] += 1
    return vis


def ukeire(after: list[int], meld_n: int, s: int, vis: list[int], memo: dict) -> int:
    total = 0
    for t in range(tiles.TILE_KINDS):
        remaining = COPIES - vis[t] - after[t]
        if remaining <= 0:
            continue
        after[t] += 1
        try:
            sh = shanten_mod.shanten_any(after, meld_n, memo=memo)
        except Exception:  # noqa: BLE001
            sh = None
        after[t] -= 1
        if sh is not None and sh < s:
            total += remaining
    return total


def process_batch(batch, chunk_path):
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    rooms = 0
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
                etype = ev.get("type")
                seat = ev.get("seat")
                if etype == DRAWN and isinstance(seat, int) and 0 <= seat < 4:
                    draw_idx[seat] += 1
                    replay.apply_event(state, ev)
                    continue
                if etype != DISCARDED or not isinstance(seat, int) or not (0 <= seat < 4):
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
                    continue
                # A 00:55 裁决：memo 生命周期挪到每个决策点（用完即弃），
                # 否则手牌无关的全局缓存在单块 20 房内就膨胀到 7GB+。
                memo: dict = {}
                try:
                    s_actual = shanten_mod.shanten(list(after_actual), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    continue
                # A 00:55 退一步方案：只对向听==2 的决策点算 rank（M1 靶子层），
                # 其余跳过——CPU 砍到 ~1/5，保留 M1 裁决所需的全部信息。
                if s_actual != 2:
                    continue
                vis = visible_counts(state)
                vis[diff[0]] -= 1
                u_actual = ukeire(list(after_actual), meld_n, s_actual, vis, memo)
                rank = 0
                n_cand = 0
                max_u = u_actual  # A 13:12 ⑥：gap 幅度 = max_u − u_actual
                for t, c in enumerate(before):
                    if c <= 0 or t == diff[0]:
                        continue
                    cand = list(before)
                    cand[t] -= 1
                    try:
                        s_c = shanten_mod.shanten(cand, meld_n, memo=memo)
                    except Exception:  # noqa: BLE001
                        continue
                    if s_c != s_actual:
                        continue
                    n_cand += 1
                    u_c = ukeire(cand, meld_n, s_actual, vis, memo)
                    if u_c > max_u:
                        max_u = u_c
                    if u_c > u_actual:
                        rank += 1
                gap = max_u - u_actual
                grp = "our" if ids[seat] == OUR else "opp"
                n = draw_idx[seat]
                n_bucket = "n≤4" if n <= 4 else ("n5-8" if n <= 8 else ("n9-12" if n <= 12 else "n≥13"))
                god = "god" if before[tiles.GOD] > 0 else "nogod"
                s_bucket = str(s_actual) if s_actual <= 2 else "3+"
                key = (grp, s_bucket, god, n_bucket)
                a = agg[key]
                a[0] += 1
                a[1] += rank
                a[2] += 1 if rank == 0 else 0
                a[3] += 1 if rank <= 1 else 0
                a[4] += 1 if rank >= 3 else 0
                a[5] += gap
    agg_ser = {"||".join(str(x) for x in k): v for k, v in agg.items()}
    payload = {"rooms": rooms, "agg": agg_ser}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="同向听层实际选择的 ukeire rank（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--arm-map", default=None,
                    help="game_id→臂映射 JSON（build_arm_map.py 产物）；给了就按臂过滤，mixed/未知剔除")
    ap.add_argument("--arm", default=None,
                    help="只统计指定臂（如 v5/v6）；与 --arm-map 联用")
    args = ap.parse_args(argv)

    arm_of = {}
    if args.arm_map:
        with open(args.arm_map, encoding="utf-8") as f:
            arm_of = json.load(f).get("map", {})

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if arm_of:
        keep = []
        for p in files:
            gid = Path(p).stem  # 事件流文件名即 game_id
            arm = arm_of.get(gid)
            if arm is None or arm == "mixed":
                continue
            if args.arm and arm != args.arm:
                continue
            keep.append(p)
        files = keep
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    tag = ("-" + args.arm) if args.arm else ""
    # A 13:12 ⑥：gap_sum 改变了 chunk 落盘格式（5 列→6 列），与旧 n=890 批（5 列）不兼容。
    # 用独立目录（-gap 后缀）重跑，保旧产物（rank 分布）可复算、不覆盖。
    chunk_dir = ROOT / "agent" / "out" / f"c31r{tag}-chunks" / f"cs{args.chunk_size}-n{len(files)}-gap"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size

    # memo 生命周期已挪进 process_batch 的每个决策点（A 00:55 裁决：一行级修法）。
    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path)
        print(f"chunk {ci:04d}: {rooms} rooms -> {chunk_path}", flush=True)

    # 合并（6 列：n, rank_sum, rank0, rank_le1, rank_ge3, gap_sum）
    merged: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    total_rooms = 0
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["agg"].items():
            m = merged[k]
            for i in range(6):
                m[i] += v[i]
    print(f"rooms={total_rooms}")
    print(f"{'grp':>4} {'向听':>4} {'财神':>6} {'摸序':>6} {'n':>6} {'均rank':>7} {'rank0%':>7} {'≤1%':>6} {'≥3%':>6} {'均gap':>7}")
    for k in sorted(merged):
        n, rs, r0, r1, r3, gs = merged[k]
        if not n:
            continue
        print(f"{k.replace('||',' '):>22} {n:>6} {rs/n:>7.2f} {r0/n*100:>6.1f}% {r1/n*100:>5.1f}% {r3/n*100:>5.1f}% {gs/n:>7.2f}")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
