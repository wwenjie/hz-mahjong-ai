"""对手侧补数：C 2026-10-03 00:30 请求——「对手在『向听≥1 且副露不变向听』的窗口上副露率是多少？」

承接 C 的闸门诊断（真机 decision.made 审计，只覆盖我方）。对手侧无日志，只能从事件流重放。

口径：
- 别人 tile_discarded 后，座位 ns 的响应 = 紧随的 pass/peng/chi/gang/timeout。
  timeout 记为「拒副露」（00:21 更正：被拒副露以 timeout 静默表现）。
- 「能副露」判定：手里有对子（碰）或下家可吃（验结构）。
- 对每个能副露的窗：算 s0=副露前向听、s1=副露后向听（碰=去掉2张+副露；吃=去掉2张+副露）。
  窗口分类：s0==0（已听牌）/ s0>=1且s1<s0（向听改善）/ s0>=1且s1==s0（向听不变）/ s0>=1且s1>s0（向听变差）。
- 核心输出：在「s0>=1 且 s1==s0」窗口上，我方 vs 对手的实际副露率。
  C 的预测：若对手在这类窗大量副露 ⇒ EQUAL 放开有真机依据；若对手也不副露 ⇒ 差距在别的桶。

抗回收：分块幂等落盘 agent/out/meld-equal-window-chunks/。
用法：`.venv/bin/python agent/verify/meld_equal_window_probe.py [--arm-map M --arm v5] [--rooms N] [--chunk-size N]`
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
TIMEOUT = "timeout"
MELD_TYPES = {"peng", "chi", "gang"}


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


def _chi_options(hand: list[int], t: int):
    """所有能吃 t 的两张组合。"""
    opts = []
    if t >= 27:
        return opts
    suit_base = (t // 9) * 9
    for lo in (t - 2, t - 1, t):
        hi = lo + 2
        if lo < suit_base or hi > suit_base + 8:
            continue
        need = [x for x in (lo, lo + 1, hi) if x != t]
        if all(0 <= x < len(hand) and hand[x] > 0 for x in need):
            opts.append(tuple(need))
    return opts


def _shanten_best_after_discard(h2: list[int], meld_n1: int, memo) -> int | None:
    """副露后的向听：暗手处于「弃牌前」状态（比 13-3*meld 多 1 张），
    枚举打出每张牌取最优向听。"""
    best = None
    for d in range(len(h2)):
        if h2[d] <= 0:
            continue
        h2[d] -= 1
        try:
            s = shanten_mod.shanten(h2, meld_n1, memo=memo)
        except Exception:  # noqa: BLE001
            s = None
        h2[d] += 1
        if s is not None and (best is None or s < best):
            best = s
    return best


def _shanten_after_peng(hand: list[int], t: int, meld_n: int, memo) -> int | None:
    """碰 t 后并弃 1 张的最优向听：手里去 2 张 t，副露数 +1。"""
    h2 = list(hand)
    h2[t] -= 2
    return _shanten_best_after_discard(h2, meld_n + 1, memo)


def _shanten_after_chi(hand: list[int], t: int, meld_n: int, memo) -> int | None:
    """吃 t 后并弃 1 张的最优向听：枚举所有吃法 × 弃牌，取最优。"""
    best = None
    for need in _chi_options(hand, t):
        h2 = list(hand)
        for x in need:
            h2[x] -= 1
        s = _shanten_best_after_discard(h2, meld_n + 1, memo)
        if s is not None and (best is None or s < best):
            best = s
    return best


def process_batch(batch, chunk_path, memo):
    # key: (grp, win_class) -> [窗口数, 实际副露数]
    # win_class: tenpai(s0=0) / improve(s1<s0) / equal(s0>=1,s1==s0) / worse(s1>s0)
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
            for i, ev in enumerate(events):
                if ev.get("type") == DISCARDED:
                    dseat = ev.get("seat")
                    if isinstance(dseat, int) and 0 <= dseat < 4:
                        try:
                            t = tiles.parse(ev.get("tile"))
                        except Exception:  # noqa: BLE001
                            t = None
                        if t is not None:
                            for j in range(i + 1, min(i + 6, len(events))):
                                nev = events[j]
                                nt, ns = nev.get("type"), nev.get("seat")
                                if nt in (DISCARDED, DRAWN):
                                    break
                                if not isinstance(ns, int) or not (0 <= ns < 4) or ns == dseat:
                                    continue
                                if nt in MELD_TYPES or nt in (PASS, TIMEOUT):
                                    hand = list(state.seats[ns].hand)
                                    can_peng = hand[t] >= 2
                                    can_chi = (ns == (dseat + 1) % 4) and _can_chi(hand, t)
                                    if not (can_peng or can_chi):
                                        break
                                    meld_n = len(state.seats[ns].melds)
                                    try:
                                        s0 = shanten_mod.shanten(hand, meld_n, memo=memo)
                                    except Exception:  # noqa: BLE001
                                        break
                                    if s0 is None or s0 < 0:
                                        break
                                    cands = []
                                    if can_peng:
                                        s1 = _shanten_after_peng(hand, t, meld_n, memo)
                                        if s1 is not None:
                                            cands.append(s1)
                                    if can_chi:
                                        s1 = _shanten_after_chi(hand, t, meld_n, memo)
                                        if s1 is not None:
                                            cands.append(s1)
                                    if not cands:
                                        break
                                    s1 = min(cands)
                                    if s0 == 0:
                                        wc = "tenpai"
                                    elif s1 < s0:
                                        wc = "improve"
                                    elif s1 == s0:
                                        wc = "equal"
                                    else:
                                        wc = "worse"
                                    grp = "our" if ids[ns] == OUR else "opp"
                                    key = (grp, wc)
                                    agg[key][0] += 1
                                    if nt in MELD_TYPES:
                                        agg[key][1] += 1
                                    break
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
    agg_ser = {"||".join(str(x) for x in k): v for k, v in agg.items()}
    payload = {"rooms": rooms, "agg": agg_ser}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对手侧『向听不变窗』副露率（分块幂等续跑）")
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
    chunk_dir = ROOT / "agent" / "out" / f"meld-eqw{tag}-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size

    for ci in range(n_chunks):
        memo: dict = {}  # 按块清空（shanten memo 膨胀教训）
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
    print(f"{'key':>14} {'窗口数':>8} {'实际副露':>8} {'副露率':>7}")
    for k in sorted(merged):
        n, m = merged[k]
        if not n:
            continue
        print(f"{k.replace('||',' '):>14} {n:>8} {m:>8} {m/n*100:>6.1f}%")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
