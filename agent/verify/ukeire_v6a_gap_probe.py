"""[待C] A 2026-10-03 20:30 交叉检验：同 1,090 房同 5 桶，把「我方实际选择」换成 **v6a 复算**选择，报 our gap。

问题（A 原话）：用同一个 gap 探针，把「我方实际选择」换成 v6a 复算的选择（同样的 1,090 房、
同样 5 桶），报 our gap。**预登记预测**：v6a 的 gap 应明显小于 v5 的 1.273（v5 是 3 切片、
v6a 是 5 切片，且例 6/8 上 v6a 已选到最优）。

- 预测成立 ⇒ 机制链闭合（切片宽度→gap），「加宽在结局上测平」⇒ M1 关闭加固；
- gap 不降 ⇒ v6a 与 v5 在同向听层选择几乎相同（切片不是瓶颈），M1 关闭得更彻底。

口径与 ukeire_rank_probe.py（gap_sum 版）一致：同向听候选 = 打出后向听 == s_after 的候选；
ukeire = 使手牌向听下降的牌张数（4 − 全场可见）；gap = max_u − u_actual。
**唯一差别**：our 桶的「实际选择」= v6a 决策器在同一局面复算出的弃牌，而非事件流里 v5 的实弃。

分桶聚合（our-v6a / our-actual / opp × 向听 × 财神 × 摸序段）：
  [n, rank_sum, rank0_count, rank_le1_count, rank_ge3_count, gap_sum]

注意（A 20:30 更正）：复现没有真实弃牌堆差异——本探针 visible 取自回放状态（含全场已打出
+ 副露），与真机快照同口径；但 v6a 复算是在「v5 实际走出的世界线」上做的反事实选择，
对手后续行为不再改变，这是单点反事实，不是完整反事实对局。

抗回收：分块幂等落盘（agent/out/c31r-v6a-gap-chunks/），与 C31-R 同模式。

用法：`.venv/bin/python agent/verify/ukeire_v6a_gap_probe.py [--rooms N] [--chunk-size N]`
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
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"
DISCARDED = "tile_discarded"
DRAWN = "tile_drawn"
COPIES = 4

# 全局只建一次决策器（PolicyConfig 无状态，choose 不残留局面；last_detail 每次 choose 重置）。
_V6A = versions.build("v6", Mode.QUALIFIER)


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


def v6a_pick(state, seat: int, before: list[int]) -> int | None:
    """v6a 在当前回放局面下的弃牌选择（反事实复算）。失败返回 None（跳过该点）。"""
    try:
        situation = state.situation_for(seat)
        if situation.is_restricted:
            # 抓打圈内只能打刚摸到的牌；此时不是自由弃牌决策点，跳过。
            return None
        actions = legal_actions(situation)
        discards = [a for a in actions if a.kind == DISCARD]
        if not discards:
            return None
        chosen = _V6A.choose(situation, actions, budget_ms=0)
        if chosen is None or chosen.kind != DISCARD or chosen.tile is None:
            return None
        # 合法性 sanity：必须在手牌里
        if before[chosen.tile] <= 0:
            return None
        return chosen.tile
    except Exception:  # noqa: BLE001
        return None


def process_batch(batch, chunk_path):
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    rooms = 0
    skipped_v6a_none = 0
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
                memo: dict = {}
                try:
                    s_after = shanten_mod.shanten(list(after_actual), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    continue
                if s_after != 2:
                    continue
                vis = visible_counts(state)
                vis[diff[0]] -= 1

                is_our = ids[seat] == OUR
                # our 桶：反事实复算 v6a 的选择；opp 桶保持事件流实弃（对照基线不变）
                if is_our:
                    pick = v6a_pick(state, seat, before)
                    if pick is None:
                        skipped_v6a_none += 1
                        continue
                    after_pick = list(before)
                    after_pick[pick] -= 1
                    memo2: dict = {}
                    try:
                        s_pick = shanten_mod.shanten(after_pick, meld_n, memo=memo2)
                    except Exception:  # noqa: BLE001
                        continue
                    if s_pick != s_after:
                        # v6a 选择了「掉向听」的牌：不在同向听层内，跳过
                        # （与探针口径一致——口径只量同向听层内的选牌质量）
                        continue
                    u_actual = ukeire(after_pick, meld_n, s_after, vis, memo)
                    eval_tile = pick
                else:
                    u_actual = ukeire(list(after_actual), meld_n, s_after, vis, memo)
                    eval_tile = diff[0]

                rank = 0
                max_u = u_actual
                for t, c in enumerate(before):
                    if c <= 0 or t == eval_tile:
                        continue
                    cand = list(before)
                    cand[t] -= 1
                    try:
                        s_c = shanten_mod.shanten(cand, meld_n, memo=memo)
                    except Exception:  # noqa: BLE001
                        continue
                    if s_c != s_after:
                        continue
                    u_c = ukeire(cand, meld_n, s_after, vis, memo)
                    if u_c > max_u:
                        max_u = u_c
                    if u_c > u_actual:
                        rank += 1
                gap = max_u - u_actual
                grp = "our-v6a" if is_our else "opp"
                n = draw_idx[seat]
                n_bucket = "n≤4" if n <= 4 else ("n5-8" if n <= 8 else ("n9-12" if n <= 12 else "n≥13"))
                god = "god" if before[tiles.GOD] > 0 else "nogod"
                key = (grp, "2", god, n_bucket)
                a = agg[key]
                a[0] += 1
                a[1] += rank
                a[2] += 1 if rank == 0 else 0
                a[3] += 1 if rank <= 1 else 0
                a[4] += 1 if rank >= 3 else 0
                a[5] += gap
    agg_ser = {"||".join(str(x) for x in k): v for k, v in agg.items()}
    payload = {"rooms": rooms, "skipped_v6a_none": skipped_v6a_none, "agg": agg_ser}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v6a 反事实选择的 ukeire gap 交叉检验（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--arm-map", default=None,
                    help="game_id→臂映射 JSON；给了就按臂过滤，mixed/未知剔除")
    ap.add_argument("--arm", default=None, help="只统计指定臂（如 v5）；与 --arm-map 联用")
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
    chunk_dir = ROOT / "agent" / "out" / f"c31r-v6a-gap{tag}-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size

    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path)
        print(f"chunk {ci:04d}: {rooms} rooms -> {chunk_path}", flush=True)

    merged: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    total_rooms = 0
    total_skip = 0
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        total_skip += d.get("skipped_v6a_none", 0)
        for k, v in d["agg"].items():
            m = merged[k]
            for i in range(6):
                m[i] += v[i]
    print(f"rooms={total_rooms} skipped_v6a_none={total_skip}")
    print(f"{'key':>28} {'n':>6} {'均rank':>7} {'rank0%':>7} {'≤1%':>6} {'≥3%':>6} {'均gap':>6}")
    for k in sorted(merged):
        n, rs, r0, r1, r3, gs = merged[k]
        if not n:
            continue
        print(f"{k:>28} {n:>6} {rs/n:>7.2f} {r0/n*100:>6.1f}% {r1/n*100:>5.1f}% "
              f"{r3/n*100:>5.1f}% {gs/n:>6.2f}")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
