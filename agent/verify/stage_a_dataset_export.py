"""Stage A′/B 共用数据底座 —— 1b v2 分歧点数据集导出（A 2026-10-05 22:15 ④ 交办 [待C]）。

任务（A 22:15 ④ 原文）：「把 1b v2 的**分歧点数据集导出成训练/拟合用的标准格式**
（每个点：局面特征 + bot 的选择 + 我们 v5 的选择）——这是 A′/B 的共用底座。」

口径与边界：
- **决策点 = 头部 bot（TOP_BOTS，与 1b 同名单）的出牌点**。标签 = 该点 bot 实际打出。
  这些点天然分成「v5 与 bot 一致（80.5%）/ 分歧（19.5%）」两类——两类都是拟合目标
  （A′ 拟合权重是为了**提高与 bot 的一致率**，需要全部点，不是只要分歧点）。
- **局面特征**：`src/majiang/strategy/features.extract(situation)` 的**定长 29 维**向量
  （FEATURES.md 钉死：数据生成与线上推理共用本模块，不另起特征集，避免 train-serve skew）。
  **注意**：A 22:15 提到的「186 万点 / 186 维」与 `features.FEATURE_COUNT = 29` 不符——
  本导出用的是 FEATURES.md 规定的 29 维（公开信息、无对手暗牌、无累计分）。
  186 维的出处请 A/B' 澄清（可能是把 one-hot 牌编码也算进去的另一种口径）。
- **候选面**（Stage A′ 的拟合目标）：对每个 legal discard 候选记
  `{tile, main_total, shanten, is_bot, is_v5, wait_copies, ukeire_exact, wait_kinds}`。
  - `main_total` = `_score_discard` 的统一主排序键（Stage A′ 拟合的就是它内部各项的权重）；
  - `wait_copies`（听牌层，真实 `_wait_copies`，含「减掉即将打出那张」的口径）/
    `ukeire_exact`（向听 ≥1 层，真实 `shanten.ukeire`）= 并列层的两个真实键；
  - 候选的内部**子项**（route/pair_value/meld_value/feed_cost/god_penalty）**不落盘**——
    它们由 `total` 在导出时已按 v5 现行权重合成，子项要在拟合时按需重算（候选面小、重算便宜），
    落盘只会把 29 维的局面向量复制 N 份、体积膨胀一个量级。
- **成本**：只在 bot 出牌点做特征 + 候选评分（主排序 ms 级 + 并列层精确键），不做对拍。
  ~186 万点 × ~20ms ≈ 10h 单核 ⇒ **走采样**（与 1b v2 同 seed 42 的 2000 房）≈ 90 分钟。
  分块幂等落盘 `agent/out/stage-a-dataset-chunks/cs<N>-n<M>/`，DONE 收尾。

产物 schema（JSONL，每行一个决策点）：
```
{
  "schema": "stage_a_decision_point.v1",
  "room": "<file stem>",          # 分组切分键（按房分组，A 22:15 ② 切分纪律）
  "round_no": int, "turn_bucket": "1-5|6-10|11-15|16+",
  "shanten": int, "god_n": int,    # 分层键（向听/财神）
  "bot_name": str, "bot_seat": int,
  "bot_tile": int,                 # 标签：bot 实际打出
  "v5_tile": int,                  # 我们 v5 的选择
  "agree": bool,                   # v5_tile == bot_tile
  "features": [float] * 29,        # FEATURES.md 规定的局面向量
  "candidates": [ {tile, main_total, shanten, is_bot, is_v5,
                   wait_copies, ukeire_exact, wait_kinds}, ... ]
}
```

用法：`.venv/bin/python agent/verify/stage_a_dataset_export.py [--rooms N] [--chunk-size N]`
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules import win as win_mod  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay as R  # noqa: E402
from majiang.strategy import features as features_mod  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.policy import Mode, _wait_copies  # noqa: E402

TOP_BOTS = [
    "玄武-2346", "三杯猫", "铳一色14", "歪比巴卜肉蛋葱鸡", "爆头研究所",
    "Astra-0", "腾蛇-0638", "康陶应雀", "凤凰-2626", "glm-flash",
    "麒麟-7780", "白虎-0211", "豆包豆包帮我把其他AI电源拔掉",
    "Nomad", "双白平胡", "Kimi-K4.1", "菜菜子", "走马", "今晚打老虎",
    "晴总总，该请桂语山房了",
]

DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"
UKEIRE_MAX_SHANTEN = 3

TURN_BUCKETS = ((0, 5, "1-5"), (6, 10, "6-10"), (11, 15, "11-15"), (16, 99, "16+"))


def turn_bucket(n: int) -> str:
    for lo, hi, name in TURN_BUCKETS:
        if lo <= n <= hi:
            return name
    return "16+"


def candidate_rows(decider, situation, bot_tile, v5_tile, visible):
    """对每个 legal discard 候选算主排序键 + 并列层真实键。

    返回 (rows, errors_counter)。键的口径与 v5 决策内部一致（真实 `_wait_copies` /
    `shanten.ukeire`，不是复刻）。
    """
    errors = collections.Counter()
    cands = [a for a in legal_actions(situation) if a.kind == DISCARD]
    if not cands:
        return [], errors
    try:
        scores = sorted(
            (decider._score_discard(situation, a) for a in cands),  # noqa: SLF001
            key=lambda item: item.total, reverse=True)
    except Exception:  # noqa: BLE001
        errors["score"] += 1
        return [], errors
    top_sh = scores[0].shanten
    tied = [s for s in scores if s.shanten == top_sh]
    do_tie = len(tied) >= 2 and 0 <= top_sh <= UKEIRE_MAX_SHANTEN
    wait_aware = top_sh == 0  # v5 的 wait_aware_tenpai=True
    memo: dict = {}
    rows = []
    for s in scores:
        counts = list(situation.hand.counts)
        counts[s.tile] -= 1
        wait_copies = None
        ukeire_exact = None
        wait_kinds = None
        if do_tie and s in tied:
            if wait_aware:
                wc = _wait_copies(counts, situation.hand.meld_count, visible, s.tile)
                if wc is not None:
                    wait_copies = wc
                try:
                    wait_kinds = len(win_mod.winning_draws(counts, situation.hand.meld_count))
                except ValueError:
                    wait_kinds = 0
            else:
                try:
                    entries = shanten_mod.ukeire(
                        counts, situation.hand.meld_count, visible=visible, memo=memo)
                    ukeire_exact = sum(copy for _, copy in entries)
                except Exception:  # noqa: BLE001
                    errors["ukeire"] += 1
        rows.append({
            "tile": s.tile,
            "main_total": round(s.total, 6),
            "shanten": s.shanten,
            "is_bot": s.tile == bot_tile,
            "is_v5": s.tile == v5_tile,
            "wait_copies": wait_copies,
            "ukeire_exact": ukeire_exact,
            "wait_kinds": wait_kinds,
        })
    return rows, errors


def process_batch(batch, chunk_path, decider, targets):
    points = []
    rooms = 0
    errors = collections.Counter()
    for path in batch:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        seats_meta = doc.get("seats") or []
        ids = [str(s.get("user_id", "")) for s in seats_meta]
        if len(ids) != 4:
            continue
        # 哪些座位是目标 bot
        seat_of_uid = {}
        for si, s in enumerate(seats_meta):
            u = str(s.get("user_id", ""))
            if u in targets:
                seat_of_uid[si] = targets[u]
        if not seat_of_uid:
            continue
        rooms += 1
        room = Path(path).stem
        try:
            rounds = list(R.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            draws = [0, 0, 0, 0]
            for ev in events:
                et = ev.get("type")
                seat = ev.get("seat")
                if not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et == DRAWN:
                    draws[seat] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et != DISCARDED or seat not in seat_of_uid:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if not state.opened:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                bot_tile = R._tile_of(ev.get("tile"))  # noqa: SLF001
                if bot_tile is None:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                try:
                    sit = state.situation_for(seat, phase=PHASE_DRAW, drawn=None)
                    acts = legal_actions(sit)
                    chosen = decider.choose(sit, acts, budget_ms=600)
                except Exception:  # noqa: BLE001
                    errors["choose"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if chosen is None:
                    errors["no_choice"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                v5_tile = chosen.tile
                hand_counts = list(state.seats[seat].hand)
                meld_n = len(state.seats[seat].melds)
                try:
                    sh0 = shanten_mod.shanten_any(hand_counts, meld_n)
                except Exception:  # noqa: BLE001
                    errors["shanten"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if sh0 is None or sh0 < 0:
                    errors["shanten_neg"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                god_n = hand_counts[tiles.GOD]
                try:
                    feats = features_mod.extract(sit)
                except Exception:  # noqa: BLE001
                    errors["features"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                visible = shanten_mod.visible_counts(
                    sit.hand.counts,
                    [meld.tiles for meld in sit.all_melds],
                    sit.discards,
                )
                rows, cerr = candidate_rows(decider, sit, bot_tile, v5_tile, visible)
                errors.update(cerr)
                if not rows:
                    errors["no_candidates"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                points.append({
                    "schema": "stage_a_decision_point.v1",
                    "room": room,
                    "round_no": getattr(state, "round_no", None),
                    "turn_bucket": turn_bucket(draws[seat]),
                    "shanten": sh0,
                    "god_n": god_n,
                    "bot_name": seat_of_uid[seat],
                    "bot_seat": seat,
                    "bot_tile": bot_tile,
                    "v5_tile": v5_tile,
                    "agree": v5_tile == bot_tile,
                    "features": [round(float(x), 4) for x in feats],
                    "candidates": rows,
                })
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
    payload = {
        "rooms": rooms,
        "points": points,
        "errors": dict(errors),
    }
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms, len(points)


def merge_report(chunk_dir: Path) -> None:
    total_rooms = 0
    total_points = 0
    agree = 0
    err_all = collections.Counter()
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for p in d["points"]:
            total_points += 1
            if p["agree"]:
                agree += 1
        for k, v in d.get("errors", {}).items():
            err_all[k] += v
    print(f"\nrooms={total_rooms} points={total_points} "
          f"一致率={agree / total_points * 100 if total_points else 0:.1f}%（1b v2 参考 80.5%） "
          f"errors={dict(err_all)}")
    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage A′/B 数据底座导出（分块幂等续跑，可分片并行）")
    ap.add_argument("--rooms", type=int, default=2000)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0, help="本进程处理 ci %% shards == shard 的块")
    ap.add_argument("--shards", type=int, default=1, help="总分片数（每片一个进程，块不相交）")
    ap.add_argument("--merge-only", action="store_true", help="只跑合并（要求全部 chunk 已在盘）")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    # 与 1b v2 完全相同的采样（seed 42、random.sample 2000 房），保证数据底座与 1b v2 的
    # 80.5% 基线出自同一总体、可直接对照。
    if args.rooms and len(files) > args.rooms:
        random.seed(42)
        files = random.sample(files, args.rooms)
        print(f"采样: {args.rooms} 房（seed 42，与 1b v2 同批）", flush=True)
    files.sort()

    chunk_dir = ROOT / "agent" / "out" / "stage-a-dataset-chunks" / f"cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    print(f"事件流 {len(files)} 房 / {n_chunks} 块 -> {chunk_dir}", flush=True)

    if args.merge_only:
        merge_report(chunk_dir)
        return 0

    decider = versions.build("v6", Mode.QUALIFIER)
    print(f"决策器: {decider.name}", flush=True)

    # name -> uid（照 1b v2 的先扫一遍）
    name_to_uid = {}
    for fpath in files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception:  # noqa: BLE001
            pass
    targets = {name_to_uid[n]: n for n in TOP_BOTS if n in name_to_uid}
    print(f"目标 bot: {len(targets)}/{len(TOP_BOTS)} shard {args.shard}/{args.shards}", flush=True)

    for ci in range(n_chunks):
        if ci % args.shards != args.shard:
            continue
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms, npts = process_batch(batch, chunk_path, decider, targets)
        print(f"chunk {ci:04d}: {rooms} rooms, {npts} points", flush=True)

    # 只有单分片（或最后一个分片发现全部块已齐）才合并；多分片时请最后用 --merge-only 收尾
    done = sum(1 for _ in chunk_dir.glob("chunk-*.json"))
    if done == n_chunks:
        merge_report(chunk_dir)
    else:
        print(f"shard {args.shard} 完成；已落盘 {done}/{n_chunks} 块（全部齐后用 --merge-only 合并）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
