"""threat 层标定：模型给的「对手已听」概率 vs 实测（只读；离线允许用暗手作标签）。

**为什么这条关键**：真机实测中段出牌 **97.7%** 由「喂牌」项决定，而
`feed = visible_need(tile) × threat`：`visible_need` 是写死的牌种表，
`threat = Σ ready_probability` 是 `HeuristicReadyModel` 的**手写常数**
（`base + w_discard·(1−e^{−弃牌/6}) + w_meld·副露数 + 0.18·局进度`）。
`threat` 对所有候选是同一个标量 ⇒ 不改变排序方向，但**决定「少喂牌」与「好形质」的
相对权重**，也**同时**进 `lap_survival`（飘的阈值判定）。它此前从未被标定。

**方法**：逐座重建四家手牌。在我方每个出牌点记录每个对手的公开特征
（副露数、弃牌数、局进度）与模型给出的 ready；再以**该对手下一次出牌**（即下次摸牌后）
是否听牌（`shanten_any == 0`）作为标签。按 (副露数, 弃牌数) 分桶比较预测 vs 实测。

**实测结论（2026-09-29，n=7045）**：模型 **36.1%** vs 实测 **12.4%**，
差 **+23.7pp ≈ 2.9 倍**，且**每个分桶都高估**（0 副露桶高估 26~36pp、1 副露桶 19~26pp）。
两条连锁后果：
1. 喂牌惩罚被放大约 2.9 倍 ⇒ **过度避免喂牌**，代价是手牌形质；
2. `lap_survival` 被低估 ⇒ **过于保守不敢飘**（我们财飘/胡 0.26% vs 对手 0.8%）。

给出数据依据：`3.0 / 2.9 ≈ 1.03`（与 `shape-feed-low` 的 1.0 吻合）。

判据：系统性高估 ⇒ 喂牌项被放大、`feed_weight` 应下调；低估则相反。

用法::

    uv run python tools/calibrate_threat.py --limit 40
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"
# 与 `src/majiang/strategy/risk.py` 的常数**逐字对齐**（不凭记忆填）：
#   BASE_READY=0.06 / DISCARD_PROGRESS_WEIGHT=0.30 / MELD_READY_WEIGHT=0.16
#   READY_MAX=0.85 / READY_MIN=0.02 / 局进度系数 0.18（估计式里的硬编码项）
BASE = 0.06
W_DISCARD = 0.30
W_MELD = 0.16
PROGRESS_W = 0.18
READY_MAX = 0.85
READY_MIN = 0.02
DRAW_SPAN = 16.0


def predict(melds: int, discards: int, draws: int) -> float:
    """复刻 `HeuristicReadyModel.estimate` 的公式（只吃公开量）。"""
    progress = min(1.0, draws / DRAW_SPAN)
    ready = (
        BASE
        + W_DISCARD * (1.0 - math.exp(-discards / 6.0))
        + W_MELD * melds
        + PROGRESS_W * progress
    )
    return min(READY_MAX, max(READY_MIN, ready))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="threat 层标定")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args(argv)

    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))[: args.limit]
    rows: dict[tuple[int, int], list[tuple[float, int]]] = collections.defaultdict(list)
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        mine = ids.index(OUR)
        others = [s for s in range(4) if s != mine]
        for state, events in replay.iter_rounds(doc):
            pending: dict[int, tuple[float, int, int]] = {}
            for e in events:
                seat = e.get("seat")
                if (
                    e.get("type") == "tile_discarded"
                    and seat == mine
                    and state.opened
                    and not (e.get("data") or {}).get("catch_play")
                ):
                    for opp in others:
                        st = state.seats[opp]
                        key = (min(len(st.melds), 3), min(len(st.discards), 12))
                        pending[opp] = (
                            predict(len(st.melds), len(st.discards), state.draws),
                            key[0],
                            key[1],
                        )
                elif e.get("type") == "tile_discarded" and seat in others and state.opened:
                    info = pending.pop(seat, None)
                    tile = replay._tile_of(e.get("tile"))  # noqa: SLF001
                    if info is not None and tile is not None:
                        counts = list(state.seats[seat].hand)
                        if counts[tile] > 0:
                            counts[tile] -= 1
                        try:
                            value = sm.shanten_any(counts, len(state.seats[seat].melds))
                        except Exception:  # noqa: BLE001
                            value = -1
                        if value >= 0:
                            rows[(info[1], info[2])].append((info[0], 1 if value == 0 else 0))
                replay.apply_event(state, e)

    print(f"{'副露':>4s} {'弃牌':>4s} {'n':>6s} {'模型预测':>9s} {'实测听牌率':>11s} {'差':>8s}")
    tp = te = 0.0
    tn = 0
    for key in sorted(rows):
        data = rows[key]
        if len(data) < 120:  # 样本下限：少于此不报数
            continue
        pred = sum(p for p, _ in data) / len(data)
        emp = sum(l for _, l in data) / len(data)
        tp += sum(p for p, _ in data)
        te += sum(l for _, l in data)
        tn += len(data)
        print(f"{key[0]:>4d} {key[1]:>4d} {len(data):>6d} {pred:>9.1%} {emp:>11.1%} {pred - emp:>+8.1%}")
    if tn:
        print(f"\n合计 n={tn}  模型 {tp / tn:.1%} vs 实测 {te / tn:.1%}  差 {tp / tn - te / tn:+.1%}")
    print("\n判据：系统性高估 ⇒ 喂牌项被放大、feed_weight 应下调；低估则相反。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
