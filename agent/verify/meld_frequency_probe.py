"""副露率对手侧·第二层：可副露点频次的成因分解——A 2026-10-03 00:10 批准诊断范围内，B' 机械补数。

00:15 全量结果：我方在能副露的点上 100% 副露（闸门无约束），但可副露点 2184 vs 对手 25243
（座位校正后少 ~3.9 倍）。本探针回答：为什么对手有更多可副露点？

分解两个成因（每个「他人弃牌→我方响应窗口」事件都算）：
  (a) **机会率**：他人弃牌中，我们手里能碰/吃的比例（可副露点 / 面对的他人弃牌数）。
      若我方机会率 ≈ 对手 ⇒ 弃牌流错位（对手弃的牌配不上我们的牌）是主因；
  (b) **对子密度**：同一（摸序段 × 向听层）桶内，手牌中对子/刻子的平均数量。
      若我方对子密度 < 对手 ⇒ 留牌结构是主因（留中张 ⇒ 对子少）。

分桶：我方/对手 × 摸序段 × 向听层。
  - opp_opp：对手面对「其他对手」的弃牌（对照组）
  - opp_our：对手面对「我方」的弃牌（测我方弃牌是否更难被碰——弃牌流错位方向 A）
  - our：我方面对任何人弃牌

抗回收：分块幂等落盘 agent/out/meld-freq-chunks/。
用法：`.venv/bin/python agent/verify/meld_frequency_probe.py [--arm-map M --arm v5] [--rooms N] [--chunk-size N]`
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


def _can_chi(hand: list[int], t: int) -> bool:
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


def _pair_count(hand: list[int]) -> int:
    """对子数（2 张计 1 对；3/4 张各计 1 刻，也算可碰）。"""
    return sum(1 for c in hand if c >= 2)


def process_batch(batch, chunk_path, memo):
    # key: (side, discarder_grp, n_bucket, s_bucket) -> [面对弃牌数, 可副露数, 对子数和, 手数]
    # side: our / opp；discarder_grp: 弃牌者是 our/opp
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0])
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
            for ev in events:
                etype = ev.get("type")
                seat = ev.get("seat")
                if etype == DISCARDED and isinstance(seat, int) and 0 <= seat < 4:
                    try:
                        t_idx = tiles.parse(ev.get("tile"))
                    except Exception:  # noqa: BLE001
                        t_idx = None
                    if t_idx is not None:
                        d_grp = "our" if ids[seat] == OUR else "opp"
                        for ns in range(4):
                            if ns == seat:
                                continue
                            hand = list(state.seats[ns].hand)
                            if sum(hand) == 0:
                                continue
                            side = "our" if ids[ns] == OUR else "opp"
                            n = draw_idx[ns]
                            n_bucket = "n≤4" if n <= 4 else ("n5-8" if n <= 8 else ("n9-12" if n <= 12 else "n≥13"))
                            meld_n = len(state.seats[ns].melds)
                            try:
                                s = shanten_mod.shanten(hand, meld_n, memo=memo)
                            except Exception:  # noqa: BLE001
                                s = None
                            if s is None or s < 0:
                                continue
                            s_bucket = str(s) if s <= 2 else "3+"
                            can = hand[t_idx] >= 2 or ((ns == (seat + 1) % 4) and _can_chi(hand, t_idx))
                            key = (side, d_grp, n_bucket, s_bucket)
                            a = agg[key]
                            a[0] += 1
                            a[1] += 1 if can else 0
                            a[2] += _pair_count(hand)
                            a[3] += 1
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
    ap = argparse.ArgumentParser(description="可副露点频次成因分解（分块幂等续跑）")
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
    chunk_dir = ROOT / "agent" / "out" / f"meld-freq{tag}-chunks" / f"cs{args.chunk_size}-n{len(files)}"
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

    merged: dict = collections.defaultdict(lambda: [0, 0, 0, 0])
    total_rooms = 0
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["agg"].items():
            m = merged[k]
            for i in range(4):
                m[i] += v[i]
    print(f"rooms={total_rooms}")
    print(f"{'key':>28} {'面对弃牌':>8} {'可副露':>7} {'机会率':>7} {'均对子':>7}")
    for k in sorted(merged):
        faced, can, pairs, hands = merged[k]
        if not faced:
            continue
        print(f"{k.replace('||',' '):>28} {faced:>8} {can:>7} {can/faced*100:>6.1f}% {pairs/max(hands,1):>7.2f}")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
