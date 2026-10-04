"""分歧特征探针：v5-godwait 与 v5 的分歧点机制画像 —— A 2026-10-04 10:00 ④ 交办（[待C]）。

回答的问题（A 原文四项）：
  (a) 分歧点占总决策点的比例；
  (b) 分歧点是否都是「近并列」（top-2 的 v5 键 copies 差 ≤2）；
  (c) 改判前后 wait kinds / copies 的分布变化；
  (d) 分歧点集中在哪些桶（巡目段 × 财神数 × 向听）。

A 的判读（预登记）：若分歧点系统性是「近并列且改判后 kinds 更高」⇒ 机制 =
「近并列处优先听口种数」⇒ 可解释，才考虑真机限臂；若分歧点杂乱无章 ⇒ 判运气，封存该臂。

方法：离线 replay 真机事件流的我方出牌点；对每个点重建 Situation 后，
**只走一遍并列层**（与 policy._break_ties_by_ukeire 同代码路径的只读复刻），
为每个候选同时算 v5 键（copies）与 godwait 键（copies + boost×god_n×kinds），
由此同成本推出两个臂的选择（单臂 3 候选预算 42–157ms/候选；逐点跑两决策器成本 ~2×，
只读复刻可省一半且行为等价——选择函数 = max(key)，godwait 键 = v5 键 + 加成项，
v5-godwait 与 v5 走完全相同的 tie 层选择代码，差异只在键值）。

口径钉死：
  - 决策点 = 我方座位 tile_discarded，`state.opened` 后，`situation_for(seat, PHASE_DRAW)`。
  - 两臂配置照 `src/majiang/cli.py` v5 / v5-godwait（QUALIFIER 模式、god_wait_boost=2.0）。
  - 主排序 + tie 层参数化 = 同 policy.py 当前代码；若有改 policy.py 的行为请先重测。
  - 数据源 = data/auto_sessions/*/events/*.json 全量（真机采集批，与 4.53/4.57 同批）。
  - 分桶：巡目段（n≤4 / 5-8 / 9-12 / ≥13）× 财神数（0/1/≥2）× 向听（0/1/2/≥3）。
    财神 0 时两臂逐位相等（policy 不变量）⇒ 分歧必在财神 ≥1 桶；仍全量计数核对不变量。
  - 近并列（(b) 口径）= 分歧点的 v5 键 top-2 copies 差 ≤ 2。
  - 改判前后 kinds/copies（(c) 口径）= v5 所选候选的 (kinds, copies) vs godwait 所选的
    (kinds, copies)；kinds = len(winning_draws(打出后手))，向听 0 层才有定义；
    向听 ≥1 层 kinds 报打出后手的 winning_draws 长度（可空）。

成本实测修正（2026-10-04）：v5 的 tie 层在**所有**并列点（含无财神）都跑精确进张
（对拍实测：god=0 并列点跳过 tie 层 ⇒ 26/287=9.1% 复刻错判，不可省）。
⇒ 探针成本 ≈ v5 真机单臂成本（~12s/房量级），全量 7,366 房 ~25h 不可行
⇒ **走采样**（按 A 01:35 纪律）：~400 房 ≈ 1.4 万决策点，预期分歧点 ~30-80 个
（冒烟实测分歧率 ~0.3%），够 (a)-(d) 的机制画像判读；不用于冠军/轴生死判决。
抗回收：分块幂等落盘 agent/out/godwait-divergence-chunks/cs<N>-n<M>/，DONE 收尾。

用法：`.venv/bin/python agent/verify/godwait_divergence_probe.py [--rooms N] [--chunk-size N]`
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
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig  # noqa: E402

OUR = "u_a7f7c67bb14a"
DISCARDED = "tile_discarded"
DRAWN = "tile_drawn"
GOD_BOOST = 2.0  # cli.py v5-godwait
UKEIRE_MAX_SHANTEN = 3
UKEIRE_CANDIDATES = 3

N_BUCKETS = ("n≤4", "n5-8", "n9-12", "n≥13")


def n_bucket(n: int) -> str:
    return N_BUCKETS[0] if n <= 4 else (N_BUCKETS[1] if n <= 8 else (N_BUCKETS[2] if n <= 12 else N_BUCKETS[3]))


def god_bucket(g: int) -> str:
    return "0张" if g == 0 else ("1张" if g == 1 else "≥2张")


def sh_bucket(s: int) -> str:
    s = min(max(s, 0), 3)
    return f"{s}向听" if s < 3 else "≥3向听"


def build_deciders():
    v5 = HeuristicDecider(PolicyConfig.for_mode(
        Mode.QUALIFIER, tiebreak="exact-ukeire", wait_aware_tenpai=True,
        shape_value=True, ukeire_order="blocks", ukeire_max_shanten=UKEIRE_MAX_SHANTEN,
        ukeire_candidates=UKEIRE_CANDIDATES))
    return v5, None


def tie_layer_rows(decider: HeuristicDecider, situation, boost: float):
    """复刻 policy._break_ties_by_ukeire 的并列层，返回 (top_shanten, rows)。

    rows = [(tile, v5_copies, gw_copies, kinds)]——对每个参与精确进张比较的候选。
    不参与（非并列、超向听、候选数<2）返回 (top_shanten, [])。
    只读复刻：行为依赖 policy 当前代码；policy 改了本探针需重核。
    """
    candidates = [a for a in legal_actions(situation) if a.kind == DISCARD]
    if not candidates:
        return None, []
    scores = sorted(
        (decider._score_discard(situation, a) for a in candidates),  # noqa: SLF001
        key=lambda item: item.total, reverse=True)
    top_shanten = scores[0].shanten
    tied = [s for s in scores if s.shanten == top_shanten]
    if len(tied) < 2 or top_shanten > UKEIRE_MAX_SHANTEN:
        return top_shanten, []
    wait_aware = top_shanten == 0  # wait_aware_tenpai=True
    tied = sorted(tied, key=lambda item: -item.blocks)
    if not wait_aware:
        tied = tied[: max(1, UKEIRE_CANDIDATES)]
    visible = shanten_mod.visible_counts(
        situation.hand.counts,
        [meld.tiles for meld in situation.all_melds],
        situation.discards,
    )
    memo: dict = {}
    rows = []
    for score in tied:
        counts = list(situation.hand.counts)
        counts[score.tile] -= 1
        god_n = counts[tiles.GOD]
        if wait_aware:
            wl = win_mod.winning_draws(counts, situation.hand.meld_count)
            copies = sum(max(0, tiles.COPIES_PER_KIND - visible[t]) for t in wl)
        else:
            entries = shanten_mod.ukeire(counts, situation.hand.meld_count, visible=visible, memo=memo)
            copies = sum(copy for _, copy in entries)
        if boost > 0 and god_n >= 1:
            try:
                kinds = len(win_mod.winning_draws(counts, situation.hand.meld_count))
            except ValueError:
                kinds = 0
        else:
            kinds = 0
        gw_copies = copies + boost * god_n * kinds if (boost > 0 and god_n >= 1) else copies
        rows.append((score.tile, copies, gw_copies, kinds))
    return top_shanten, rows


def process_batch(batch, chunk_path):
    v5, gw = build_deciders()
    # 决策点总数（分母，(a)）与分歧计数按 (nb, gb, sb) 桶；另单记 tenpai 标志（sub_bucket）
    points: dict = collections.defaultdict(int)   # (nb, gb, sb) -> n
    diverge: dict = collections.defaultdict(int)  # (nb, gb, sb) -> div_n
    # 分歧点明细聚合：(nb, gb, sb) -> [n, near_tie_n, delta_kinds_sum, delta_v5copies_sum,
    #                                    v5_kinds_sum, gw_kinds_sum, v5_copies_sum, gw_copies_sum,
    #                                    boost_fired_n, top2_gap_sum]
    divs: dict = collections.defaultdict(lambda: [0] * 10)
    rooms = 0
    errors = collections.Counter()
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
            draws = [0, 0, 0, 0]
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
                    draws[seat] += 1
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et != DISCARDED or ids[seat] != OUR:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                # 我方出牌点：先取局面（打出前），两臂各选一次
                if not state.opened:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                try:
                    sit = state.situation_for(seat, phase="draw", drawn=None)
                except Exception:  # noqa: BLE001
                    errors["sit"] += 1
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                hand_counts = list(state.seats[seat].hand)
                meld_n = len(state.seats[seat].melds)
                try:
                    sh0 = shanten_mod.shanten_any(hand_counts, meld_n)
                except Exception:  # noqa: BLE001
                    sh0 = -1
                if sh0 is None or sh0 < 0:
                    errors["shanten"] += 1
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                god_n = hand_counts[tiles.GOD]
                nb, gb, sb = n_bucket(draws[seat]), god_bucket(god_n), sh_bucket(sh0)
                # 只读复刻整条选择链（与 policy._choose_discard 当前代码等价）：
                # 主排序 _score_discard（ms 级）→ 并列层精确进张（仅 tie+财神+向听≤3 才付）。
                # 不调用 choose()：其内部 tie 层结果无法取出，复算等于双倍成本。
                # 与真机口径差异（诚实声明）：离线不算 EXACT_UKEIRE_BUDGET_SEC 墙钟截断
                # （真机/A/B 有 0.6s 上限，超时率极低；此处取确定性全量）。
                try:
                    acts = legal_actions(sit)
                    cands = [a for a in acts if a.kind == DISCARD]
                    if not cands:
                        try:
                            replay.apply_event(state, ev)
                        except Exception:  # noqa: BLE001
                            break
                        continue
                    scores = sorted(
                        (v5._score_discard(sit, a) for a in cands),  # noqa: SLF001
                        key=lambda item: item.total, reverse=True)
                except Exception:  # noqa: BLE001
                    errors["score"] += 1
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                key = (nb, gb, sb)
                points[key] += 1
                top_sh = scores[0].shanten
                tied_n = sum(1 for s in scores if s.shanten == top_sh)
                # tie 层触发条件与真机 v5 完全一致：并列≥2 且向听≤3（**无财神条件**——
                # v5 自己的 tie 层在无财神并列时也会用精确进张重排，2026-10-04 对拍实测
                # 跳过它会导致 26/287=9.1% 的 v5 复刻错判）。财神只影响 godwait 键。
                if not (tied_n >= 2 and 0 <= top_sh <= UKEIRE_MAX_SHANTEN):
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                try:
                    _, rows = tie_layer_rows(v5, sit, boost=GOD_BOOST)
                except Exception:  # noqa: BLE001
                    errors["tie_layer"] += 1
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if not rows:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                v5_pick = max(rows, key=lambda r: r[1])[0]
                gw_pick = max(rows, key=lambda r: r[2])[0]
                if gw_pick == v5_pick:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                diverge[key] += 1
                d = divs[key]
                d[0] += 1
                rows_sorted_v5 = sorted(rows, key=lambda r: -r[1])
                gap = rows_sorted_v5[0][1] - rows_sorted_v5[1][1] if len(rows_sorted_v5) >= 2 else 99
                d[9] += gap
                if gap <= 2:
                    d[1] += 1
                r5 = next((r for r in rows if r[0] == v5_pick), None)
                rg = next((r for r in rows if r[0] == gw_pick), None)
                if r5 and rg:
                    d[2] += rg[3] - r5[3]           # Δkinds（gw 所选 − v5 所选）
                    d[3] += rg[1] - r5[1]           # Δv5-copies
                    d[4] += r5[3]
                    d[5] += rg[3]
                    d[6] += r5[1]
                    d[7] += rg[1]
                    if rg[2] != rg[1]:
                        d[8] += 1
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
    payload = {
        "rooms": rooms,
        "points": {"|".join(k): v for k, v in points.items()},
        "diverge": {"|".join(k): v for k, v in diverge.items()},
        "divs": {"|".join(k): v for k, v in divs.items()},
        "errors": dict(errors),
    }
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v5-godwait vs v5 分歧特征探针（分块幂等续跑）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=200)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    chunk_dir = ROOT / "agent" / "out" / "godwait-divergence-chunks" / f"cs{args.chunk_size}-n{len(files)}"
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

    # 合并
    points: dict = collections.defaultdict(int)
    diverge: dict = collections.defaultdict(int)
    divs: dict = collections.defaultdict(lambda: [0] * 10)
    total_rooms = 0
    err_all = collections.Counter()
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["points"].items():
            points[k] += v
        for k, v in d["diverge"].items():
            diverge[k] += v
        for k, v in d["divs"].items():
            m = divs[k]
            for i in range(len(v)):
                m[i] += v[i]
        for k, v in d.get("errors", {}).items():
            err_all[k] += v

    tot_points = sum(points.values())
    tot_div = sum(diverge.values())
    print(f"\nrooms={total_rooms} 决策点={tot_points} 分歧={tot_div} "
          f"({tot_div / tot_points * 100 if tot_points else 0:.2f}%) errors={dict(err_all)}")

    print(f"\n== (d) 分歧点分桶（巡目段 × 财神 × 向听；只列出有点的桶）==")
    print(f"{'桶':<18} {'决策点':>8} {'分歧':>6} {'分歧率':>7}")
    for k in sorted(points):
        n = points[k]
        dv = diverge.get(k, 0)
        if not n:
            continue
        print(f"{k:<18} {n:>8} {dv:>6} {dv / n * 100:>6.2f}%")

    print(f"\n== (b)(c) 分歧点机制画像（只列有分歧的桶）==")
    print(f"{'桶':<18} {'分歧n':>6} {'近并列%':>7} {'Δkinds':>7} {'Δcopies':>7} "
          f"{'v5kinds':>7} {'gwkinds':>7} {'v5cop':>6} {'gwcop':>6} {'加权生效%':>8} {'top2gap':>7}")
    all_d = [0] * 10
    for k in sorted(divs):
        d = divs[k]
        for i in range(10):
            all_d[i] += d[i]
        n = d[0]
        if not n:
            continue
        print(f"{k:<18} {n:>6} {d[1] / n * 100:>6.1f}% {d[2] / n:>7.2f} {d[3] / n:>7.2f} "
              f"{d[4] / n:>7.2f} {d[5] / n:>7.2f} {d[6] / n:>6.1f} {d[7] / n:>6.1f} "
              f"{d[8] / n * 100:>7.1f}% {d[9] / n:>7.2f}")
    n = all_d[0]
    if n:
        print(f"{'全体':<18} {n:>6} {all_d[1] / n * 100:>6.1f}% {all_d[2] / n:>7.2f} "
              f"{all_d[3] / n:>7.2f} {all_d[4] / n:>7.2f} {all_d[5] / n:>7.2f} "
              f"{all_d[6] / n:>6.1f} {all_d[7] / n:>6.1f} {all_d[8] / n * 100:>7.1f}% "
              f"{all_d[9] / n:>7.2f}")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
