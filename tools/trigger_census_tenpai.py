"""触发点普查（第四类/第五类）：**与用户场景近似**的听牌取舍，按巡目分层。

**为什么重做**（用户 2026-10-09 11:06 质疑「构造的 case 是否跟当时情况类似」）：
上一版 `tools/trigger_census_early.py` 用 `shanten.ukeire` 算候选进张，但 **`ukeire` 在
「打完即听牌」时返回空元组** ⇒ 所有「打完就听牌」的候选被当成 `copies=0` 跳过，
**恰好把「早听（窄口）」这一类丢掉**；实测「打完即听牌」占决策 **24.1%**，其听口张数
最小 5 张、中位 ≈11 张（`/tmp/dose_tenpai_narrow.py`）。听牌档的正确口径是 `_wait_copies`
（可见听口张数）。麻雀理论还指出这条取舍**与巡目相关**（早巡好形优先、中晚巡速度优先；
[麻雀力向上委員会 第32回] 等），所以本次一并记录巡目。

两类触发点（同一份产出，供 `tools/trigger_counterfactual.py --mode discard --force-tile`）：
- **类 A「早听窄口 vs 晚一步宽进张」**：v5 打完即听牌、**听口 ≤6 张**，且存在「保持向听 ≥1
  但进张 ≥2×听口」的备选 ⇒ `arm_tile` = 那张宽进张牌。
- **类 B「听牌档内：张数优先 vs 种数优先」**：v5 打完即听牌，但**听口最多的那张不是种数最多的**
  ⇒ `arm_tile` = 种数最多的听牌牌（直接检验「听口种数」这条嫌疑，对应已量到的
  「持财神听口种数 4.53 vs bot 6.08」）。

用法::
    .venv/bin/python tools/trigger_census_tenpai.py --rooms 1500 --shard 0 --shards 14 \\
        --out agent/out/trigger-points/tenpai-shard0.jsonl
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
from majiang.rules import shanten as sh  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules import win as win_mod  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import Mode, _wait_copies  # noqa: E402

OUR = "u_a7f7c67bb14a"
NARROW_MAX = 6      # 类 A：听口 ≤ 该张数算「窄」
RATIO_MIN = 2.0     # 类 A：备选进张 ≥ 该倍数 × 听口张数


def turn_bucket(turn: int) -> str:
    return "1-6巡" if turn <= 6 else ("7-10巡" if turn <= 10 else "11巡+")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="听牌取舍触发点普查（类 A / 类 B）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--progress", type=int, default=25)
    ap.add_argument("--out", default="agent/out/trigger-points/tenpai.jsonl")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)

    dec = make_decider("v5", Mode.QUALIFIER)
    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        random.seed(args.seed)
        files = sorted(random.sample(files, min(args.rooms, len(files))))
    if args.shards > 1:
        files = files[args.shard::args.shards]

    stats: collections.Counter = collections.Counter()
    points: list[dict] = []
    examples: list[str] = []

    for index, path in enumerate(files):
        if args.progress and (index + 1) % args.progress == 0:
            print(f"  [进度] {index + 1}/{len(files)} 房，触发 {len(points)}", flush=True)
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        for block_no, (state, events) in enumerate(replay.iter_rounds(doc)):
            for ev_index, event in enumerate(events):
                if (event.get("type") != "tile_discarded" or event.get("seat") != mine
                        or (event.get("data") or {}).get("catch_play") or not state.opened):
                    replay.apply_event(state, event)
                    continue
                try:
                    situation = state.situation_for(mine)
                    actions = legal_actions(situation)
                    candidates = [a for a in actions if a.kind == DISCARD]
                    if len(candidates) < 2:
                        replay.apply_event(state, event)
                        continue
                    stats["决策"] += 1
                    scores = [dec._score_discard(situation, a) for a in candidates]  # noqa: SLF001
                    s0 = min(s.shanten for s in scores)
                    if s0 != 0:
                        replay.apply_event(state, event)
                        continue
                    stats["打完即听牌"] += 1
                    visible = sh.visible_counts(
                        situation.hand.counts,
                        [m.tiles for m in situation.all_melds],
                        situation.discards,
                    )
                    meld_count = situation.hand.meld_count

                    def after_discard(tile: int) -> list[int]:
                        counts = list(situation.hand.counts)
                        counts[tile] -= 1
                        return counts

                    def wait_stats(tile: int) -> tuple[int | None, int]:
                        after = after_discard(tile)
                        copies = _wait_copies(after, meld_count, visible, tile)
                        try:
                            kinds = len(win_mod.winning_draws(after, meld_count))
                        except ValueError:
                            kinds = 0
                        return copies, kinds

                    pick = dec.choose(situation, actions, budget_ms=2000)
                    if pick is None or pick.kind != DISCARD:
                        replay.apply_event(state, event)
                        continue
                    pick_copies, pick_kinds = wait_stats(pick.tile)
                    if not pick_copies:
                        replay.apply_event(state, event)
                        continue
                    turn = state.draws // 4
                    god_n = int(situation.hand.counts[tiles.GOD])

                    # 备选 1：保持向听 ≥1、进张最多的那张（类 A 的宽分支）
                    wide_tile, wide_copies, wide_sh = -1, -1, None
                    for score in scores:
                        if score.shanten < 1:
                            continue
                        after = after_discard(score.tile)
                        try:
                            copies = sum(c for _, c in sh.ukeire(after, meld_count, visible=visible))
                        except Exception:  # noqa: BLE001
                            continue
                        if copies > wide_copies:
                            wide_tile, wide_copies, wide_sh = score.tile, copies, score.shanten

                    # 备选 2：听牌档里**种数最多**的那张（类 B）
                    kind_best_tile, kind_best, kind_best_copies = -1, -1, 0
                    for score in scores:
                        if score.shanten != 0:
                            continue
                        copies, kinds = wait_stats(score.tile)
                        if copies is None:
                            continue
                        if kinds > kind_best:
                            kind_best_tile, kind_best, kind_best_copies = score.tile, kinds, copies

                    klass = None
                    arm_tile = -1
                    if pick_copies <= NARROW_MAX and wide_copies >= max(1, RATIO_MIN * pick_copies):
                        klass = (f"A早听窄|听口{pick_copies}|比{wide_copies / pick_copies:.1f}"
                                 f"|财神{god_n}|{turn_bucket(turn)}")
                        arm_tile = wide_tile
                        stats["类A"] += 1
                    elif kind_best_tile >= 0 and kind_best_tile != pick.tile and kind_best > pick_kinds:
                        klass = (f"B种数|张数{pick_copies}→{kind_best_copies}"
                                 f"|种数{pick_kinds}→{kind_best}|财神{god_n}|{turn_bucket(turn)}")
                        arm_tile = kind_best_tile
                        stats["类B"] += 1
                    if klass is None:
                        replay.apply_event(state, event)
                        continue
                    if len(examples) < args.examples:
                        hand = situation.hand.counts
                        hand_str = "".join(
                            tiles.to_codes([t for t in range(34) for _ in range(hand[t])])
                        )
                        examples.append(
                            f"    {Path(path).stem} 手={hand_str} 财神={god_n} 巡{turn} | v5="
                            f"{tiles.to_code(pick.tile)}(听口{pick_copies}张/{pick_kinds}种) → 强制="
                            f"{tiles.to_code(arm_tile)}"
                            f"{'' if klass.startswith('A') else f'(张数{kind_best_copies}/种数{kind_best})'}"
                        )
                    points.append({
                        "room": doc.get("room_id"),
                        "file": str(Path(path)),
                        "round_no": state.round_no,
                        "block": block_no,
                        "ev_index": ev_index,
                        "seq": event.get("seq"),
                        "offered": pick.tile,
                        "hand_counts": list(situation.hand.counts),
                        "melds": [list(m.tiles) for m in situation.all_melds],
                        "god_n": god_n,
                        "chain": int(state.seats[mine].chain_count),
                        "piao": int(state.seats[mine].piao_count),
                        "current_shanten": s0,
                        "turn": turn,
                        "turn_bucket": turn_bucket(turn),
                        "pair_tile": pick.tile,
                        "count_of_picked": int(situation.hand.counts[pick.tile]),
                        "v5_tile": pick.tile,
                        "arm_tile": arm_tile,
                        "narrow_copies": pick_copies,
                        "narrow_kinds": pick_kinds,
                        "wide_copies": wide_copies,
                        "wide_shanten": wide_sh,
                        "kind_best_copies": kind_best_copies,
                        "kind_best_kinds": kind_best,
                        "klass": klass,
                    })
                except Exception:  # noqa: BLE001
                    pass
                replay.apply_event(state, event)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for point in points:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")
    print(f"扫描 {len(files)} 房 → 决策 {stats['决策']}，打完即听牌 {stats['打完即听牌']}"
          f"（{stats['打完即听牌'] / max(1, stats['决策']):.1%}）")
    print(f"  **类A（早听窄口 ≤{NARROW_MAX} 张 + 宽备选 ≥{RATIO_MIN}×）= {stats['类A']}**"
          f"（{stats['类A'] / max(1, stats['决策']):.2%} of 决策）")
    print(f"  **类B（张数最优 ≠ 种数最优）= {stats['类B']}**"
          f"（{stats['类B'] / max(1, stats['决策']):.2%} of 决策）")
    for line in examples:
        print(line)
    print(f"数据集: {out}（{len(points)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
