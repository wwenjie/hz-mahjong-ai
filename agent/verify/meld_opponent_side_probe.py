"""副露率「对手侧」诊断——A 2026-10-03 00:10 裁决派给 B' 的活。

问题：对手在哪些决策点上做了副露、而我方在同条件下没有？与 C 的「放开闸门反事实」互补：
  C 答「闸门放开会怎样」；本探针答「对手在哪些点上做了我们没做的副露」。

口径：
- 事件流里别人 `tile_discarded` 后，若紧随的响应是本座位 `peng`/`chi`/`gang` ⇒ 该座位副露；
  若是 `pass` ⇒ 该座位响应窗口放弃（replay.py:371，pass 不改局面）。
  **关键**：pass 不区分「能碰而跳过」与「根本不能碰」——本探针用 replay 重放的手牌判定
  「当时手里是否真有对子（碰）/可吃结构（仅下家）」，只把「**能副露**」的点计入分母。
- 分桶（我方/对手 × 摸序段 n≤4/5-8/9-12/≥13 × 向听层）统计副露决策率，
  找「对手副露率 ≥60% 而我方 ≤20%」的分歧桶。

判据（对齐 A 三分法）：
  分歧桶集中在我方能碰而跳过 ⇒ 闸门问题；
  分歧桶对手副露事后看路线期望不优 ⇒ 对手过度副露（缺口是假象）；
  分歧桶对手副露确实更优 ⇒ 估值问题。

抗回收：分块幂等落盘 agent/out/meld-opp[-v5]-chunks/。

用法：`.venv/bin/python agent/verify/meld_opponent_side_probe.py [--arm-map M --arm v5] [--rooms N] [--chunk-size N]`
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
PASS = "pass"
MELD_TYPES = {"peng", "chi", "gang"}


def _can_chi(hand: list[int], t: int) -> bool:
    """手里是否有能吃掉 t 的两张（顺子）。t 为数牌索引；字牌(>=27)/财神不可吃。"""
    if t >= 27:
        return False
    suit_base = (t // 9) * 9
    for lo in (t - 2, t - 1, t):
        hi = lo + 2
        if lo < suit_base or hi > suit_base + 8:
            continue
        need = [x for x in (lo, lo + 1, hi) if x != t]
        if all(0 <= x < len(hand) and hand[x] > 0 for x in need):
            return True
    return False


def process_batch(batch, chunk_path, memo):
    # key: (grp, n_bucket, s_bucket) -> [可副露点数, 实际副露数]
    agg: dict = collections.defaultdict(lambda: [0, 0])
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
        for state, events in replay.iter_rounds(doc):
            if state is None:
                continue
            draw_idx = [0, 0, 0, 0]
            for i, ev in enumerate(events):
                etype = ev.get("type")
                seat = ev.get("seat")
                if etype == DISCARDED and isinstance(seat, int) and 0 <= seat < 4:
                    discarded_tile = ev.get("tile")
                    try:
                        t_idx = tiles.parse(discarded_tile)
                    except Exception:  # noqa: BLE001
                        t_idx = None
                    if t_idx is not None:
                        # 找紧随的响应（pass/peng/chi/gang），响应窗口内
                        for j in range(i + 1, min(i + 5, len(events))):
                            nev = events[j]
                            nt = nev.get("type")
                            ns = nev.get("seat")
                            if nt in (DISCARDED, DRAWN):
                                break
                            if not isinstance(ns, int) or not (0 <= ns < 4) or ns == seat:
                                continue
                            if nt in MELD_TYPES or nt == PASS:
                                # 判定 ns 此刻是否能副露 discarded_tile
                                hand = list(state.seats[ns].hand)
                                can_peng = hand[t_idx] >= 2
                                can_chi = (ns == (seat + 1) % 4) and _can_chi(hand, t_idx)
                                if can_peng or can_chi:
                                    grp = "our" if ids[ns] == OUR else "opp"
                                    n = draw_idx[ns]
                                    n_bucket = "n≤4" if n <= 4 else ("n5-8" if n <= 8 else ("n9-12" if n <= 12 else "n≥13"))
                                    meld_n = len(state.seats[ns].melds)
                                    try:
                                        s = shanten_mod.shanten(hand, meld_n, memo=memo)
                                    except Exception:  # noqa: BLE001
                                        s = None
                                    if s is not None and s >= 0:
                                        s_bucket = str(s) if s <= 2 else "3+"
                                        key = (grp, n_bucket, s_bucket)
                                        agg[key][0] += 1
                                        if nt in MELD_TYPES:
                                            agg[key][1] += 1
                                break
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                if etype == DRAWN and isinstance(seat, int) and 0 <= seat < 4:
                    draw_idx[seat] += 1
    agg_ser = {"||".join(str(x) for x in k): v for k, v in agg.items()}
    payload = {"rooms": rooms, "agg": agg_ser}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="副露率对手侧诊断（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=100)
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
        files = files[::step][: args.rooms]

    tag = ("-" + args.arm) if args.arm else ""
    chunk_dir = ROOT / "agent" / "out" / f"meld-opp{tag}-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size

    memo: dict = {}
    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path, memo)
        print(f"chunk {ci:04d}: {rooms} rooms", flush=True)

    merged: dict = collections.defaultdict(lambda: [0, 0])
    total_rooms = 0
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["agg"].items():
            merged[k][0] += v[0]
            merged[k][1] += v[1]
    print(f"rooms={total_rooms}")
    print(f"{'key':>22} {'可副露点':>8} {'实际副露':>8} {'副露率':>7}")
    for k in sorted(merged):
        n, m = merged[k]
        if not n:
            continue
        print(f"{k.replace('||',' '):>22} {n:>8} {m:>8} {m/n*100:>6.1f}%")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
