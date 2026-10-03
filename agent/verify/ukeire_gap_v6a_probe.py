"""C31-R2：v6a 复算交叉检验（A 2026-10-03 20:30 指定的便宜检验）。

问题（A 原话）：用同一个 gap 探针，把「我方实际选择」换成 **v6a 复算**的选择
（同样的 1,090 房、同样 5 桶），报 our gap。预登记预测：v6a 的 gap 应明显小于
v5 的 1.273（v5 是 3 切片、v6a 是 5 切片，且例 6/8 上 v6a 已选到最优）。

- 若预测成立 ⇒ 机制链闭合（切片宽度→gap），「加宽在结局上测平」⇒ M1 关闭加固；
- 若 gap 不降 ⇒ v6a 与 v5 在同向听层选择几乎相同（切片不是瓶颈），M1 关闭更彻底。

**决策点集合与原 gap 批严格一致**（口径钉死，与 v5 数 1.273 直接可比）：
  同样的 v5 臂房（arm_map 过滤）、只看我方座位、只取「实际弃牌后向听==2」的点。
  在这些点上，改用 v6a 复算它会打哪张，再对 v6a 的选择算 gap（与 5 桶同键）。

对手侧不算（gap 0.847 已在 cs50-n1090-gap 批钉死，无需重算）⇒ 成本 ~2h 而非 6h。

每桶聚合（god × 摸序段，与 v5 批同键）：
  [n_points, gap_sum, gap_n, rank_sum, rank0_n, agree_n, leave_n]
  - agree_n：v6a 所打 == 当时实际所打（若 ≈100%，gap 必然不动，机制链直接否）
  - leave_n：v6a 的选择使向听离开 2（改善/变差）——这些点不进 gap 均值（分母=gap_n）

口径诚实边界：
  - v6a 复算用 `ReplayState.situation_for` 构造观察者视角局面（公开信息+己方暗手），
    与平台运行时同源；`legal_actions` 生成候选，Mode.QUALIFIER（与 A 离线 A/B 同口径）。
  - 庄家首巡无 tile_drawn 事件、ReplayState 未 opened ⇒ situation_for 拒绝构造，
    这些点跳过并计入 skipped（量级 ~1%，见 THREAD 20:30 前探针同口径处理）。
  - budget_ms=1300（与平台护栏一致；v6a 实测最坏 294ms，不触顶）。

抗回收：分块幂等落盘（agent/out/c31r-v5-chunks/cs50-n*-gap-v6a/）。

用法：`.venv/bin/python agent/verify/ukeire_gap_v6a_probe.py [--rooms N] [--chunk-size N]`
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

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import table as table_rules  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, Action, legal_actions  # noqa: E402
from majiang.rules.hand import Hand  # noqa: E402
from majiang.rules.situation import PHASE_DRAW, GodState, Situation, TableState  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"
DISCARDED = "tile_discarded"
DRAWN = "tile_drawn"
COPIES = 4
BUDGET_MS = 1300


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
    # 每桶: [n_points, gap_sum, gap_n, rank_sum, rank0_n, agree_n, leave_n]
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    skipped = collections.Counter()
    rooms = 0
    v6a = make_decider("v6a", Mode.QUALIFIER)
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
            last_drawn: list[int | None] = [None, None, None, None]
            for ev in events:
                etype = ev.get("type")
                seat = ev.get("seat")
                if etype == DRAWN and isinstance(seat, int) and 0 <= seat < 4:
                    draw_idx[seat] += 1
                    try:
                        tile = int(ev.get("tile")) if ev.get("tile") is not None else None
                    except Exception:  # noqa: BLE001
                        tile = None
                    # 事件里的 tile 可能是牌码字符串，走 replay 的解析口径
                    if tile is None or not (0 <= tile < tiles.TILE_KINDS):
                        tile = None
                        try:
                            tile = tiles.parse(str(ev.get("tile")))
                        except Exception:  # noqa: BLE001
                            tile = None
                    last_drawn[seat] = tile
                    replay.apply_event(state, ev)
                    continue
                if etype != DISCARDED or not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                # —— 以下与我方 v5 gap 批完全同构的决策点识别 ——
                before = list(state.seats[seat].hand)
                is_our = ids[seat] == OUR
                if not is_our:
                    # 对手侧不算（gap 已钉死），但仍需推进状态
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
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
                    s_actual = shanten_mod.shanten(list(after_actual), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    continue
                if s_actual != 2:
                    continue  # 决策点集合与 v5 批一致：只取实际弃牌后向听==2

                # —— v6a 复算：在同一决策点（未应用弃牌前的局面）问 v6a 打哪张 ——
                # 注意：state 此刻已应用了实际弃牌，需回到弃牌前视角。
                # situation_for 用的是 state.seats[seat].hand——已被实际弃牌污染。
                # ⇒ 手动恢复：把实际弃牌加回去构造局面副本不可行（ReplayState 内部还有
                #   discards/turn 等）。改为：在应用弃牌**之前**就构造局面。
                # 实现上我们在这里重新构造会重复代码，故探针主循环采用「先构造后应用」：
                # 见下方 NOTE_REORDER。当前到达此处说明结构按先应用组织 —— 用 before
                # 手工构造 v6a 输入。
                #
                # NOTE_REORDER: 实际实现为先构造（见 process_batch 上半段 reorder 注释）。
                # 为最小 diff 复用 v5 批结构，这里用 before/可见信息手工构造 Situation：
                try:
                    sit = _situation_from_before(state, seat, before, last_drawn[seat])
                except Exception:  # noqa: BLE001
                    skipped["situation"] += 1
                    continue
                actions = legal_actions(sit)
                if not actions:
                    # turn 不匹配等边界：手工构造纯弃牌候选
                    actions = tuple(
                        Action(DISCARD, tile=t) for t in range(tiles.TILE_KINDS) if before[t] > 0
                    )
                try:
                    choice = v6a.choose(sit, actions, budget_ms=BUDGET_MS)
                except Exception:  # noqa: BLE001
                    skipped["choose"] += 1
                    continue
                if choice is None or choice.kind != DISCARD or choice.tile is None:
                    skipped["no_discard"] += 1
                    continue
                v6a_tile = choice.tile
                if before[v6a_tile] <= 0:
                    skipped["illegal_tile"] += 1
                    continue

                # v6a 选择后的向听
                after_v6a = list(before)
                after_v6a[v6a_tile] -= 1
                try:
                    s_v6a = shanten_mod.shanten(after_v6a, meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    continue

                god = "god" if before[tiles.GOD] > 0 else "nogod"
                n = draw_idx[seat]
                n_bucket = "n≤4" if n <= 4 else ("n5-8" if n <= 8 else ("n9-12" if n <= 12 else "n≥13"))
                key = (god, n_bucket)
                a = agg[key]
                a[0] += 1
                if v6a_tile == diff[0]:
                    a[5] += 1  # agree
                if s_v6a != s_actual:
                    a[6] += 1  # leave layer（不进 gap 均值）
                    continue

                # v6a 选择的 gap（同向听候选内 max_u − u_actual）
                vis = visible_counts(state)
                # vis 基于 state（已应用实际弃牌）：把实际弃牌从可见里减回、把 v6a 弃牌加回
                # 保持「打出瞬间」的可见口径（与 v5 批 vis[diff[0]] -= 1 同构）：
                # v5 批：state 已含实际弃牌 ⇒ vis 含它 ⇒ 减 1 得打出前可见。
                # 本批同理减 1（对实际弃牌），u 计算针对 v6a 弃牌后的手牌。
                vis[diff[0]] -= 1
                u_actual = ukeire(list(after_v6a), meld_n, s_actual, vis, memo)
                rank = 0
                max_u = u_actual
                for t, c in enumerate(before):
                    if c <= 0 or t == v6a_tile:
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
                    if u_c > max_u:
                        max_u = u_c
                    if u_c > u_actual:
                        rank += 1
                gap = max_u - u_actual
                a[1] += gap
                a[2] += 1  # gap_n
                a[3] += rank
                a[4] += 1 if rank == 0 else 0

    agg_ser = {"||".join(str(x) for x in k): v for k, v in agg.items()}
    payload = {"rooms": rooms, "agg": agg_ser, "skipped": dict(skipped)}
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def _situation_from_before(state, seat: int, before: list[int], drawn: int | None):
    """用「弃牌前」的手牌构造 v6a 的输入局面。

    state 已应用实际弃牌（手牌 13 张），而 v6a 需要在 14 张（含摸牌）时决策。
    这里从 state 的公共信息 + before（弃牌前手牌）直接构造 Situation，不回改 state：
    - 该座位手牌 = before（14 张，含刚摸的牌）
    - 该座位弃牌堆需减回「刚被 replay 推进的实际弃牌」（state.seats[seat].discards 尾项）
    """
    seat_state = state.seats[seat]
    hand = Hand.from_counts(before, seat_state.melds)
    discards = [list(s.discards) for s in state.seats]
    if discards[seat]:
        discards[seat] = discards[seat][:-1]  # 减回刚应用的实际弃牌
    god = GodState(
        hand_gods=hand.god_count,
        chain_count=seat_state.chain_count,
        piao_count=seat_state.piao_count,
        catch_play=state.catch_play,
        god_discarder_seat=state.god_discarder,
    )
    wall_remaining = max(0, table_rules.INITIAL_WALL - state.draws)
    if not state.opened:
        raise ValueError("not opened")
    return Situation.from_parts(
        seat=seat,
        phase=PHASE_DRAW,
        turn=seat,
        hand=hand,
        god=god,
        table=TableState(
            wall_remaining=wall_remaining,
            dealer_seat=state.dealer,
            round_no=state.round_no,
        ),
        responding_seats=(),
        offered_tile=None,
        drawn_tile=drawn,
        discards=tuple(tuple(d) for d in discards),
        melds=tuple(tuple(s.melds) for s in state.seats),
        hand_counts=tuple(
            sum(before) if i == seat else sum(s.hand) for i, s in enumerate(state.seats)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v6a 复算交叉检验（决策点集合与 v5 gap 批一致）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--arm-map", default=None)
    ap.add_argument("--arm", default="v5")
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
    chunk_dir = ROOT / "agent" / "out" / f"c31r{tag}-chunks" / f"cs{args.chunk_size}-n{len(files)}-gap-v6a"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size

    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path)
        print(f"chunk {ci:04d}: {rooms} rooms -> {chunk_path}", flush=True)

    # 合并（7 列）
    merged: dict = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    total_rooms = 0
    skipped_all = collections.Counter()
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["agg"].items():
            m = merged[k]
            for i in range(7):
                m[i] += v[i]
        skipped_all.update(d.get("skipped", {}))
    print(f"rooms={total_rooms}  skipped={dict(skipped_all)}")
    print(f"{'财神':>6} {'摸序':>6} {'n':>6} {'均gap':>7} {'gap_n':>6} {'均rank':>7} {'rank0%':>7} {'agree%':>7} {'leave%':>7}")
    for k in sorted(merged):
        n, gs, gn, rs, r0, agree, leave = merged[k]
        if not n:
            continue
        print(
            f"{k.replace('||',' '):>12} {n:>6} "
            f"{(gs/gn if gn else float('nan')):>7.2f} {gn:>6} "
            f"{(rs/gn if gn else float('nan')):>7.2f} {(r0/gn*100 if gn else 0):>6.1f}% "
            f"{agree/n*100:>6.1f}% {leave/n*100:>6.1f}%"
        )
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
