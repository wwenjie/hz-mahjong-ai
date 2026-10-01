"""C30：到听后胡率按「到听时机/剩余摸数」分层。

只读、零平台请求。目的：C18 测出我们到听率缺口 7.8pp、转换率缺口 3.2pp；
但转换率可能只是因为「我们到听更晚、剩余机会更少」。本探针把「到听后胡」按
首次到听发生在本人第 n 摸 / 到听后剩余摸数 分层，分离组成效应。

口径：
- 每次本人摸牌后（手牌为 14-3×副露）用 `shanten_any` 判定是否听牌；
- 记录该座位**首次**听牌的摸序 n、本局该座位总摸数 D、剩余摸数 = D - n；
- 胡局判定用文档级 `rounds[]` 的 `winner/is_draw`（与 C18 同口径）。

用法：`.venv/bin/python agent/verify/convert_by_remaining_probe.py [--rooms N]`

**抗回收（本机会周期性回收长跑进程）**：`--chunk-size N` 时按文件序分块，
每块聚合并**原子落盘**到 `agent/out/c30-chunks/chunk-NNNN.json`，已存在的块跳过
（幂等续跑）。全部块完成后合并写 `agent/out/convert-by-remaining.log` 并落
`agent/out/c30-chunks/DONE` 标记。被回收后原样重跑即可接着跑。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys
from math import erf, sqrt

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
DRAWN = "tile_drawn"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    if not n1 or not n2:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return ((k2 / n2) - (k1 / n1)) / se if se else float("nan")


def bucket_first_n(n: int) -> str:
    if n <= 4:
        return "n≤4"
    if n <= 8:
        return "n5-8"
    if n <= 12:
        return "n9-12"
    return "n≥13"


def bucket_remaining(r: int) -> str:
    if r >= 8:
        return "rem≥8"
    if r >= 5:
        return "rem5-7"
    if r >= 2:
        return "rem2-4"
    return "rem≤1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=0,
                    help=">0 时按文件序分块落盘（幂等续跑）；0=一次性内存聚合（仅小规模用）")
    ap.add_argument("--cross", action="store_true",
                    help="同时聚合 首次到听摸序×剩余摸数 交叉桶（写入 c30x-chunks 独立目录，不撞旧块）")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.rooms:
        files = files[:: max(1, len(files) // args.rooms)][: args.rooms]

    chunk_dir = REPO / "agent" / "out" / ("c30x-chunks" if args.cross else "c30-chunks")
    if args.chunk_size > 0:
        return main_chunked(files, args.chunk_size, chunk_dir)

    agg = {
        "our": {"first": collections.defaultdict(lambda: [0, 0]), "rem": collections.defaultdict(lambda: [0, 0]), "tot": [0, 0]},
        "opp": {"first": collections.defaultdict(lambda: [0, 0]), "rem": collections.defaultdict(lambda: [0, 0]), "tot": [0, 0]},
    }
    rooms = 0
    rounds_seen = 0
    for path in files:
        rooms_delta, rounds_delta, chunk_agg = process_file(path)
        rooms += rooms_delta
        rounds_seen += rounds_delta
        merge_into(agg, chunk_agg)

    report(agg, rooms, rounds_seen, sys.stdout)
    return 0


def new_agg():
    return {
        "our": {"first": collections.defaultdict(lambda: [0, 0]), "rem": collections.defaultdict(lambda: [0, 0]),
                "cross": collections.defaultdict(lambda: [0, 0]), "tot": [0, 0]},
        "opp": {"first": collections.defaultdict(lambda: [0, 0]), "rem": collections.defaultdict(lambda: [0, 0]),
                "cross": collections.defaultdict(lambda: [0, 0]), "tot": [0, 0]},
    }


def process_file(path):
    """处理单个事件流文件，返回 (rooms_delta, rounds_delta, agg)。"""
    agg = new_agg()
    rounds_seen = 0
    try:
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0, 0, agg
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    if len(ids) != 4 or OUR not in ids:
        return 0, 0, agg
    me = ids.index(OUR)
    round_meta = {int(r.get("round_no", 0) or 0): r for r in (doc.get("rounds") or [])}
    try:
        rounds = list(replay.iter_rounds(doc))
    except Exception:  # noqa: BLE001
        return 0, 0, agg
    for state, events in rounds:
        rounds_seen += 1
        first = [None] * 4
        draw_idx = [0] * 4
        total_draws = [0] * 4
        for ev in events:
            try:
                replay.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                break
            if ev.get("type") != DRAWN:
                continue
            seat = ev.get("seat")
            if not isinstance(seat, int) or not 0 <= seat < 4:
                continue
            draw_idx[seat] += 1
            total_draws[seat] += 1
            if first[seat] is not None:
                continue
            rs = state.seats[seat]
            if sum(rs.hand) <= 0:
                continue
            try:
                sh = shanten_mod.shanten_any(rs.hand, len(rs.melds))
            except Exception:  # noqa: BLE001
                continue
            if sh <= 0:
                first[seat] = draw_idx[seat]
        meta = round_meta.get(int(state.round_no), {}) or {}
        winner = meta.get("winner")
        is_draw = bool(meta.get("is_draw"))
        for seat in range(4):
            if first[seat] is None:
                continue
            grp = "our" if seat == me else "opp"
            rem = total_draws[seat] - first[seat]
            won = (not is_draw) and winner == seat
            for key, val in (("first", bucket_first_n(first[seat])), ("rem", bucket_remaining(rem))):
                agg[grp][key][val][0] += 1
                if won:
                    agg[grp][key][val][1] += 1
            ck = f"{bucket_first_n(first[seat])}|{bucket_remaining(rem)}"
            agg[grp]["cross"][ck][0] += 1
            if won:
                agg[grp]["cross"][ck][1] += 1
            agg[grp]["tot"][0] += 1
            if won:
                agg[grp]["tot"][1] += 1
    return 1, rounds_seen, agg


def merge_into(dst, src):
    for grp in ("our", "opp"):
        for key in ("first", "rem", "cross"):
            for bucket, (n, w) in src[grp][key].items():
                dst[grp][key][bucket][0] += n
                dst[grp][key][bucket][1] += w
        dst[grp]["tot"][0] += src[grp]["tot"][0]
        dst[grp]["tot"][1] += src[grp]["tot"][1]


def agg_to_json(agg):
    out = {}
    for grp in ("our", "opp"):
        out[grp] = {
            "first": {k: list(v) for k, v in agg[grp]["first"].items()},
            "rem": {k: list(v) for k, v in agg[grp]["rem"].items()},
            "cross": {k: list(v) for k, v in agg[grp]["cross"].items()},
            "tot": list(agg[grp]["tot"]),
        }
    return out


def agg_from_json(data):
    agg = new_agg()
    for grp in ("our", "opp"):
        g = data.get(grp) or {}
        for k, v in (g.get("first") or {}).items():
            agg[grp]["first"][k] = list(v)
        for k, v in (g.get("rem") or {}).items():
            agg[grp]["rem"][k] = list(v)
        for k, v in (g.get("cross") or {}).items():
            agg[grp]["cross"][k] = list(v)
        agg[grp]["tot"] = list(g.get("tot") or [0, 0])
    return agg


def main_chunked(files, chunk_size, chunk_root) -> int:
    """分块幂等：每块落 chunk-NNNN.json；全齐后合并写总日志 + DONE 标记。"""
    # 块目录带块大小与文件数指纹：换参数不撞旧块
    chunk_dir = chunk_root / f"cs{chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    done_marker = chunk_dir / "DONE"
    n_chunks = (len(files) + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue  # 幂等：已完成块跳过
        batch = files[ci * chunk_size:(ci + 1) * chunk_size]
        rooms = 0
        rounds = 0
        agg = new_agg()
        for path in batch:
            rd, qd, ca = process_file(path)
            rooms += rd
            rounds += qd
            merge_into(agg, ca)
        payload = {"rooms": rooms, "rounds": rounds, "agg": agg_to_json(agg),
                   "files": len(batch)}
        tmp = chunk_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, chunk_path)  # 原子替换，不留半个文件
        print(f"[chunk {ci + 1}/{n_chunks}] rooms={rooms} rounds={rounds} 落盘", flush=True)

    # 合并全部块
    agg = new_agg()
    rooms = 0
    rounds = 0
    for ci in range(n_chunks):
        data = json.loads((chunk_dir / f"chunk-{ci:04d}.json").read_text(encoding="utf-8"))
        rooms += data["rooms"]
        rounds += data["rounds"]
        merge_into(agg, agg_from_json(data["agg"]))

    log_path = REPO / "agent" / "out" / ("convert-cross.log" if chunk_dir.name.startswith("c30x") else "convert-by-remaining.log")
    tmp = log_path.with_suffix(".log.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        report(agg, rooms, rounds, fh)
    os.replace(tmp, log_path)
    done_marker.write_text("done\n", encoding="utf-8")
    print(f"MERGED rooms={rooms} rounds={rounds} -> {log_path}", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


def report(agg, rooms, rounds_seen, out):
    def line(d, k):
        n, w = d[k]
        return n, w, (w / n if n else float("nan"))

    p = lambda *a, **kw: print(*a, **kw, file=out)
    p(f"房={rooms} 局={rounds_seen}")
    p("\n① 按首次到听摸序 n 分层（到听后胡率）")
    p(f"   {'桶':>6} {'我方 n':>8} {'我方胡':>7} {'我方率':>8} {'对手 n':>8} {'对手胡':>8} {'对手率':>8} {'z':>7}")
    for k in ("n≤4", "n5-8", "n9-12", "n≥13"):
        an, aw, ar = line(agg["our"]["first"], k)
        on, ow, orr = line(agg["opp"]["first"], k)
        z = two_prop_z(aw, an, ow, on)
        p(f"   {k:>6} {an:8d} {aw:7d} {ar:8.1%} {on:8d} {ow:8d} {orr:8.1%} {z:7.2f}")

    p("\n② 按到听后剩余摸数分层（到听后胡率）")
    p(f"   {'桶':>6} {'我方 n':>8} {'我方胡':>7} {'我方率':>8} {'对手 n':>8} {'对手胡':>8} {'对手率':>8} {'z':>7}")
    for k in ("rem≥8", "rem5-7", "rem2-4", "rem≤1"):
        an, aw, ar = line(agg["our"]["rem"], k)
        on, ow, orr = line(agg["opp"]["rem"], k)
        z = two_prop_z(aw, an, ow, on)
        p(f"   {k:>6} {an:8d} {aw:7d} {ar:8.1%} {on:8d} {ow:8d} {orr:8.1%} {z:7.2f}")

    an, aw = agg["our"]["tot"]
    on, ow = agg["opp"]["tot"]
    p(f"\n③ 总体（到听者中）：我方 {aw}/{an} = {aw / an:.1%}；对手 {ow}/{on} = {ow / on:.1%}；z={two_prop_z(aw, an, ow, on):+.2f}")

    if agg["our"]["cross"] or agg["opp"]["cross"]:
        p("\n④ 首次到听摸序 × 剩余摸数 交叉（到听后胡率）")
        p(f"   {'到听':>6} {'rem':>7} {'我方 n':>8} {'我方胡':>7} {'我方率':>8} {'对手 n':>8} {'对手胡':>8} {'对手率':>8} {'z':>7}")
        for fk in ("n≤4", "n5-8", "n9-12", "n≥13"):
            for rk in ("rem≥8", "rem5-7", "rem2-4", "rem≤1"):
                ck = f"{fk}|{rk}"
                an2, aw2, ar2 = line(agg["our"]["cross"], ck)
                on2, ow2, orr2 = line(agg["opp"]["cross"], ck)
                z = two_prop_z(aw2, an2, ow2, on2)
                p(f"   {fk:>6} {rk:>7} {an2:8d} {aw2:7d} {ar2:8.1%} {on2:8d} {ow2:8d} {orr2:8.1%} {z:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
