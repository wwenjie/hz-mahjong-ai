"""C31：同向听层进张面（ukeire）对照——同为 X 向听，我们留的形进张更窄吗？

只读、零平台请求。承接的证据链：
- C29（全量）证「shanten 层出牌与对手持平」⇒ 缺口不在向听数；
- C18 证「中巡 n=4–10 到听率缺口峰值 10.1pp」；
- C29/C30 之后唯一没量过的就是**同向听内的进张面**：同样打成 2 向听，
  我们选的形「下一摸能降向听的牌数」是不是比对手少。

口径（严格限定在「同向听层」内比较，避免跨向听的 apples-to-oranges）：
- 对每个「摸牌后出牌」整点（`sum(hand) == 14 − 3×副露`）：
  1. 实际出牌后向听 s_actual；
  2. 候选集 = 所有**打出后向听 == s_actual** 的候选（即同向听层）；
  3. 每个候选的 ukeire = 使手牌向听 **下降** 的牌张数（4 − 全场可见：自家手牌/
     四家舍牌/四家副露；副露是公开信息，合规）；
  4. gap = max(ukeire) − ukeire(实际)；chosen_is_max = (gap == 0)。
- 分桶：向听层（1/2/3+）× 是否持财神 × 摸序段（n≤4 / 5–8 / 9–12 / ≥13）。

用法：`.venv/bin/python agent/verify/ukeire_choice_probe.py [--rooms N]`
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics as st
import sys
from math import erf, sqrt
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


def _phat(p: float) -> float:
    return 0.5 * (1.0 + erf(p / sqrt(2.0)))


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    se = sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    if not se:
        return float("nan"), float("nan")
    t = (st.mean(a) - st.mean(b)) / se
    return t, 2 * (1 - _phat(abs(t)))


def visible_counts(state) -> list[int]:
    """全场可见张数（四家舍牌 + 四家副露）。副露是公开信息，合规。"""
    vis = [0] * tiles.TILE_KINDS
    for seat in state.seats:
        for t in seat.discards:
            vis[t] += 1
        for m in seat.melds:
            for t in m.tiles:
                vis[t] += 1
    return vis


def ukeire(after: list[int], meld_n: int, s: int, vis: list[int], memo: dict) -> int:
    """摸到哪些牌能让向听 < s。`after` 是出牌后的手牌计数（13 张）。"""
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


def process_batch(batch, chunk_path, memo):
    """处理一批文件 → 聚合结果原子落盘 chunk_path。返回 (rooms, agg_dict, gaps_lists)。"""
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    gaps = {"our": [], "opp": []}
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
                try:
                    s_actual = shanten_mod.shanten(list(after_actual), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    continue
                if s_actual <= 0:
                    continue
                vis = visible_counts(state)
                vis[diff[0]] -= 1
                u_actual = ukeire(list(after_actual), meld_n, s_actual, vis, memo)
                best = u_actual
                pool_u: list[int] = []
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
                    u_c = ukeire(cand, meld_n, s_actual, vis, memo)
                    pool_u.append(u_c)
                    if u_c > best:
                        best = u_c
                # A 18:55 的问题：实际选择在「精确进张」键上排第几（1=最优）。
                # 若 rank 常为 1 却仍慢 ⇒ 瓶颈不在这个键上，M1 整个方向要换。
                rank_u = 1 + sum(1 for u in pool_u if u > u_actual)
                pool_n = len(pool_u) + 1
                grp = "our" if ids[seat] == OUR else "opp"
                n = draw_idx[seat]
                n_bucket = "n≤4" if n <= 4 else ("n5-8" if n <= 8 else ("n9-12" if n <= 12 else "n≥13"))
                god = "god" if before[tiles.GOD] > 0 else "nogod"
                s_bucket = str(s_actual) if s_actual <= 2 else "3+"
                key = (grp, s_bucket, god, n_bucket)
                a = agg[key]
                a[0] += 1
                a[1] += best - u_actual
                a[2] += 1 if best == u_actual else 0
                a[3] += u_actual
                a[4] += best
                a[5] += rank_u
                a[6] += 1 if rank_u <= 2 else 0
                a[7] += pool_n
                gaps[grp].append(best - u_actual)
    # 序列化 agg：key tuple → 字符串
    agg_ser = {"||".join(str(x) for x in k): v for k, v in agg.items()}
    payload = {"rooms": rooms, "agg": agg_ser, "gaps": gaps}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)  # 原子落盘
    return rooms, agg, gaps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="同向听层 ukeire 对照（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=100)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    chunk_root = ROOT / "agent" / "out" / "c31-chunks"
    # v2：agg 行扩到 8 列（加 rank_u / rank≤2 / 池大小）——旧格式 chunk 不复用。
    chunk_dir = chunk_root / f"v2-cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    done_marker = chunk_dir / "DONE"
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size

    memo: dict = {}
    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue  # 幂等：已完成块跳过
        batch = files[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        rooms, _, _ = process_batch(batch, chunk_path, memo)
        print(f"[chunk {ci+1}/{n_chunks}] rooms={rooms} 落盘", file=sys.stderr, flush=True)

    # 合并全部 chunk
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    gaps = {"our": [], "opp": []}
    rooms = 0
    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        rooms += payload["rooms"]
        for k_str, v in payload["agg"].items():
            k = tuple(k_str.split("||"))
            a = agg[k]
            for i in range(8):
                a[i] += v[i]
        gaps["our"].extend(payload["gaps"]["our"])
        gaps["opp"].extend(payload["gaps"]["opp"])

    print(f"房={rooms}")
    print(f"\n{'组':>4} {'向听':>4} {'财神':>6} {'摸序':>6} {'决策数':>7} {'均ukeire':>9} {'均最优':>7} {'均gap':>7} {'最优率':>7} {'均rank':>7} {'rank≤2':>7} {'均池':>6}")
    for key in sorted(agg):
        grp, s_b, god, n_b = key
        n, sg, nm, su, sb, sr, nr2, sp = agg[key]
        if n < 30:
            continue
        print(f"{grp:>4} {s_b:>4} {god:>6} {n_b:>6} {n:7d} {su/n:9.2f} {sb/n:7.2f} {sg/n:7.2f} {nm/n:7.1%} {sr/n:7.2f} {nr2/n:7.1%} {sp/n:6.1f}")
    t, p = welch(gaps["our"], gaps["opp"])
    print(f"\n全体：我方均gap={st.mean(gaps['our']):.3f} 对手均gap={st.mean(gaps['opp']):.3f} "
          f"差={st.mean(gaps['our'])-st.mean(gaps['opp']):+.3f} (t={t:+.2f}, p={p:.4f})")
    done_marker.write_text("done\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
