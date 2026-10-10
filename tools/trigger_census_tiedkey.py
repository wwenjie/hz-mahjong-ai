"""第四类触发点普查：**退化键定向放宽**（`v7` vs `v7-tiedfull`）在真机上的分歧点。

**触发定义**：我方弃牌时刻，`v7` 与 `v7-tiedfull` 选出的牌不同 ⇒ 这一次「放宽候选面」
真的改变了决策。这批点正好是 `tools/trigger_counterfactual.py` 定向臂该量的样本。

同时报两个剂量的**区别**（这条轴的关键读数）：
- **结构剂量**：`last_detail["tied_full"]` 出现率 = 「并列层最高形质相等者 > 3」的占比
  （B' 独立实测形质全并列 40.2%）；
- **行为剂量**：两档实选不同的占比。
两者相差越大 ⇒ 「被截断挡在比较之外的好牌」越少 ⇒ 该机制的**行为**价值越小。

产出 JSONL 与 `tools/trigger_census_tiebreak.py` 同构（供 `--mode discard --force-trigger`）。

用法::
    .venv/bin/python tools/trigger_census_tiedkey.py --rooms 400 --jobs 12 --out agent/out/trigger-points/tiedkey.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"


def scan_files(files: list[str], dec_base, dec_arm, limit_points: int) -> tuple[list[dict], dict]:
    stats: collections.Counter = collections.Counter()
    points: list[dict] = []
    for path in files:
        if limit_points and len(points) >= limit_points:
            break
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        stats["命中房"] += 1
        for block_no, (state, events) in enumerate(replay.iter_rounds(doc)):
            if limit_points and len(points) >= limit_points:
                break
            stats["局"] += 1
            for ev_index, event in enumerate(events):
                if (event.get("type") != "tile_discarded" or event.get("seat") != mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                stats["我方弃牌点"] += 1
                hand = list(state.seats[mine].hand)
                if sum(hand) % 3 != 2:
                    replay.apply_event(state, event)
                    continue
                try:
                    sit = state.situation_for(mine, phase=PHASE_DRAW, drawn=None)
                    actions = legal_actions(sit)
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if len([a for a in actions if a.kind == DISCARD]) < 2:
                    replay.apply_event(state, event)
                    continue
                try:
                    pick_base = dec_base.choose(sit, actions, budget_ms=2000)
                    pick_arm = dec_arm.choose(sit, actions, budget_ms=2000)
                    # **结构剂量必须读「处理臂」的 last_detail**：`tied_full` 是它写的键，
                    # 基线 `v7` 永远不写（读错对象会把结构剂量恒读成 0）。
                    detail = dict(getattr(dec_arm, "last_detail", {}) or {})
                except Exception:  # noqa: BLE001
                    replay.apply_event(state, event)
                    continue
                if pick_base is None or pick_arm is None or pick_base.kind != DISCARD:
                    replay.apply_event(state, event)
                    continue
                if detail.get("tied_full"):
                    stats["结构剂量(键退化)"] += 1
                    stats["结构剂量明细_并列数"] += int(detail["tied_full"][0])
                stats["可比点"] += 1
                if pick_arm.tile == pick_base.tile:
                    replay.apply_event(state, event)
                    continue
                stats["行为剂量(两档不同)"] += 1
                shanten = shanten_mod.shanten_any(hand, len(state.seats[mine].melds))
                klass = f"退化键|向听{min(shanten, 4)}"
                stats[klass] += 1
                points.append({
                    "room": doc.get("room_id"),
                    "file": str(Path(path)),
                    "round_no": state.round_no,
                    "block": block_no,
                    "ev_index": ev_index,
                    "seq": event.get("seq"),
                    "hand_counts": hand,
                    "melds": [list(m.tiles) for m in state.seats[mine].melds],
                    "god_n": int(hand[tiles.GOD]),
                    "chain": int(state.seats[mine].chain_count),
                    "piao": int(state.seats[mine].piao_count),
                    "current_shanten": shanten,
                    "v7_tile": pick_base.tile,
                    "arm_tile": pick_arm.tile,
                    "tied_full": detail.get("tied_full"),
                    "klass": klass,
                })
                replay.apply_event(state, event)
                if limit_points and len(points) >= limit_points:
                    break
    return points, dict(stats)


def _worker(payload):
    files, limit_points, arm = payload
    from majiang.cli import make_decider as build
    from majiang.strategy.policy import Mode as M

    dec_base = build("v7", M.QUALIFIER)
    dec_arm = build(arm, M.QUALIFIER)
    return scan_files(files, dec_base, dec_arm, limit_points)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="退化键定向放宽：真机分歧点普查")
    ap.add_argument("--rooms", type=int, default=400, help="抽样房数（0 = 全部）")
    ap.add_argument("--arm", default="v7-tiedfull")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit-points", type=int, default=0)
    ap.add_argument("--out", default="agent/out/trigger-points/tiedkey.jsonl")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunks = [files[index::args.jobs] for index in range(args.jobs)]
        stats: collections.Counter = collections.Counter()
        points: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for part, part_stats in pool.map(
                _worker, [(chunk, args.limit_points, args.arm) for chunk in chunks]
            ):
                points.extend(part)
                stats.update(part_stats)
    else:
        dec_base = make_decider("v7", Mode.QUALIFIER)
        dec_arm = make_decider(args.arm, Mode.QUALIFIER)
        points, raw = scan_files(files, dec_base, dec_arm, args.limit_points)
        stats = collections.Counter(raw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for point in points:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")

    rounds = max(1, stats["局"])
    comparable = max(1, stats["可比点"])
    print(f"扫描 {len(files)} 房 → 命中我方 {stats['命中房']} 房 / {stats['局']} 局")
    print(f"  我方弃牌点 {stats['我方弃牌点']}  可比点 {stats['可比点']}")
    print(f"  **结构剂量（并列层最高形质相等者 >3）{stats['结构剂量(键退化)']}"
          f" = {stats['结构剂量(键退化)'] / comparable:.1%}**")
    print(f"  **行为剂量（两档实选不同）{stats['行为剂量(两档不同)']}"
          f" = {stats['行为剂量(两档不同)'] / comparable:.1%}**")
    for key in sorted(k for k in stats if k.startswith("退化键|")):
        print(f"    {key} {stats[key]}")
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
