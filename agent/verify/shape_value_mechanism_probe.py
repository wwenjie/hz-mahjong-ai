#!/usr/bin/env python
"""C 的机制仪器：`shape_value` 到底在起作用吗？（只读真机事件流，零平台请求）

A 2026-09-29 17:05 派、2026-09-30 00:45 **升级为承重件**（换档 v4 的依据是 A/B 而非机制，
所以这台仪器从「补验」变成「本档是否站得住」的判据之一）。口径照 A 原话：

  在真机决策点上，取「加 `shape_value` **之前** `total` 就并列」的候选组，报
  ① 该组占所有决策点的比例；
  ② `shape_value` 能把其中多少组分开；
  ③ 分开的方向对不对（听口更宽 / 形质更好的一方是否被选中）。

**唯一变量是 `PolicyConfig.shape_value`**：同一局面、同一候选集，两个只差该开关的
`HeuristicDecider`。主排序键 `total = −10×向听 + 形质 − 3×喂牌 − 财神罚`（默认走快路径，
不含 tie-break；见 `policy._score_discard`）。

指标（都在**我方出牌决策点**上算；分母逐个报）：
- ① 顶层 `total` 并列 ≥2 的占比。
- ② 在其中，`shape_value` 修正后并列**减少**的占比（+ 并列组平均大小变化）。
- ③ 方向：`shape_value` 改变**主键 argmax** 时，新选中的那张与旧那张比较
  **独立量**（`cheap_ukeire` 剩余张数）——更宽/更窄/相同。再按副露数分层。
- 端到端：两个 decider 走真实 `choose()` 路径，最终出牌不同的占比（含 tie-break 影响）。

用法：.venv/bin/python -u agent/verify/shape_value_mechanism_probe.py [--rooms N] [--cap-per-room M]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules.action import DISCARD, Action, legal_actions  # noqa: E402
from majiang.rules.tiles import TILE_KINDS  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.strategy.policy import (  # noqa: E402
    HeuristicDecider,
    PolicyConfig,
    cheap_ukeire,
)

OUR = "u_a7f7c67bb14a"
EPS = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=60)
    ap.add_argument("--cap-per-room", type=int, default=120)
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.rooms:
        files = files[: args.rooms]
    print(f"扫描事件流文件：{len(files)}（1 文件 = 1 场 = 8 局）", flush=True)

    dec3 = HeuristicDecider(PolicyConfig(shape_value=False))
    dec4 = HeuristicDecider(PolicyConfig(shape_value=True))

    n_dec = 0
    n_tied = 0
    n_separable = 0
    n_argmax_changed = 0
    dir_better = dir_worse = dir_eq = 0
    tie_before = tie_after = 0
    n_choose_diff = 0
    n_choose_cmp = 0
    layered: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0])

    for path in files:
        try:
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        taken = 0
        for event, state in replay.iter_before_each_event(doc):
            if event.get("type") != replay.DISCARDED or not state.opened:
                continue
            if event.get("seat") != mine:
                continue
            if taken >= args.cap_per_room:
                break
            try:
                sit = state.situation_for(mine, phase=replay.PHASE_DRAW)
            except Exception:  # noqa: BLE001
                continue
            counts = list(sit.hand.counts)
            avail = [t for t in range(TILE_KINDS) if counts[t] > 0]
            if len(avail) < 2:
                continue
            try:
                s3 = {t: dec3._score_discard(sit, Action(DISCARD, tile=t)).total for t in avail}
                s4 = {t: dec4._score_discard(sit, Action(DISCARD, tile=t)).total for t in avail}
            except Exception:  # noqa: BLE001
                continue
            taken += 1
            n_dec += 1
            m = sit.hand.meld_count
            layered[m][0] += 1

            top3 = max(s3.values())
            tied3 = [t for t in avail if abs(s3[t] - top3) <= EPS]
            p3 = min(tied3)  # 稳定排序默认 → 最小索引
            p4 = min(t for t in avail if abs(s4[t] - max(s4.values())) <= EPS)

            if len(tied3) >= 2:
                n_tied += 1
                layered[m][1] += 1
                tie_before += len(tied3)
                top4 = max(s4[t] for t in tied3)
                tied_a = [t for t in tied3 if abs(s4[t] - top4) <= EPS]
                tie_after += len(tied_a)
                if len(tied_a) < len(tied3):
                    n_separable += 1
                    layered[m][2] += 1

            if p4 != p3:
                n_argmax_changed += 1
                seen = shanten_mod.visible_counts(
                    counts, [x.tiles for x in sit.all_melds], sit.discards
                )

                def uke(t: int) -> int:
                    c = list(counts)
                    c[t] -= 1
                    return cheap_ukeire(c, seen)[1]

                u4, u3 = uke(p4), uke(p3)
                if u4 > u3:
                    dir_better += 1
                elif u4 < u3:
                    dir_worse += 1
                else:
                    dir_eq += 1

            # 端到端：真实 choose()
            acts = legal_actions(sit)
            if acts:
                try:
                    c3 = dec3.choose(sit, acts, budget_ms=0)
                    c4 = dec4.choose(sit, acts, budget_ms=0)
                except Exception:  # noqa: BLE001
                    c3 = c4 = None
                if c3 is not None and c4 is not None:
                    n_choose_cmp += 1
                    if (c3.kind, c3.tile) != (c4.kind, c4.tile):
                        n_choose_diff += 1

    def pct(a: int, b: int) -> str:
        return f"{a / b:.1%}" if b else "n/a"

    print("\n" + "=" * 78)
    print("① 顶层 total 并列 ≥2 的占比")
    print(f"   分母（我方出牌决策点）= {n_dec}")
    print(f"   分子 = {n_tied}  ⇒ ① = {pct(n_tied, n_dec)}")
    print("\n② shape_value 把并列组分开的比例")
    print(f"   分母（① 的并列组）= {n_tied}    分子 = {n_separable}  ⇒ ② = {pct(n_separable, n_tied)}")
    if n_tied:
        print(f"   佐证：并列组平均大小 {tie_before / n_tied:.2f} → {tie_after / n_tied:.2f}")
    print("\n③ 方向（主键 argmax 被 shape_value 改变时）")
    tot = dir_better + dir_worse + dir_eq
    print(f"   分母（主键 argmax 改变）= {n_argmax_changed}（占总决策 {pct(n_argmax_changed, n_dec)}）")
    print(f"   独立 ukeire：更宽 {dir_better} · 更窄 {dir_worse} · 相同 {dir_eq}  (分母 {tot})")
    if tot:
        print(f"   ⇒ '选到不更差' = {pct(dir_better + dir_eq, tot)}；'严格更宽' = {pct(dir_better, tot)}")
    print("\n端到端（真实 choose()）")
    print(f"   分母 = {n_choose_cmp}    最终出牌不同 = {n_choose_diff}  ⇒ {pct(n_choose_diff, n_choose_cmp)}")
    print("\n按副露数分层：m | 决策点 | 并列组 | 可分")
    for m in sorted(layered):
        d, t, s = layered[m]
        print(f"   m={m} | {d} | {t} | {s}  (① {pct(t, d)} / ② {pct(s, t)})" if d else "")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
