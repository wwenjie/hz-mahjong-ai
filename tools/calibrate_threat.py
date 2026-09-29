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

**实测结论（2026-09-29，**按房抽样 30 房 / n=58663**）**：

| 模型 | 预测 | 实测 | 偏差 |
|---|---|---|---|
| 手写 `HeuristicReadyModel`（**默认档在用**） | 42.0% | 28.2% | **+13.8pp（1.49×）** |
| GBDT `models/opponent_model.json` | **29.1%** | 28.2% | **+0.9pp（几乎完美）** |

两条连锁后果（对**手写**模型）：① 喂牌惩罚被放大约 1.5 倍 ⇒ 过度避免喂牌、代价是手形；
② 同一模型进 `lap_survival` ⇒ 低估生存概率 ⇒ 过于保守不敢飘（财飘/胡 0.26% vs 对手 0.8%）。
**校准后的 `feed_weight` ≈ 3.0 / 1.49 ≈ 2.0。**
⇒ **换用 GBDT 才是更彻底的修法**（它同时修好上面两条）。

**⚠ 一条我自己踩过的坑（这个工具上）**：第一版用 `--limit 40`——看着是 40 个文件、
其实只有 **4 个房**（每房 10 个文件），算出的过估是 2.9 倍；agent-c 全库复算给 1.4~1.6 倍。
**同一个「按文件抽样」坑我在别处反复警告，却在自己新写的工具上犯了。**
另一条旁证：**房间异质性极大**——4 房样本里实测听牌率 13.2%、30 房样本里 28.2%。
所以本类分析**必须按房抽样且房数足够**，并复报房数。

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


def rooms_first(paths: list[str], rooms: int) -> list[str]:
    """**按房抽样**，不按文件。

    `--limit 40` 这种写法看着覆盖 40 个文件、其实只有 **4 个房**（每房 10 个文件）。
    2026-09-29 我就在这个工具上踩了：只抽 4 个房时算出「手写模型高估 2.9 倍」，
    而 agent-c 全库复算给 1.4~1.6 倍——**同一个坑，我在别处警告别人却在这里自己犯**。
    """
    by_room: dict[str, list[str]] = {}
    for path in paths:
        by_room.setdefault(str(Path(path).parts[-3]), []).append(path)
    picked: list[str] = []
    for room in sorted(by_room)[:rooms]:
        picked.extend(sorted(by_room[room]))
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="threat 层标定")
    parser.add_argument("--rooms", type=int, default=20, help="**按房**抽样的房数（不是文件数）")
    parser.add_argument("--limit", type=int, default=0, help="兼容旧参数：最多几个文件（0=不限）")
    parser.add_argument(
        "--model",
        default="both",
        choices=("heuristic", "gbdt", "both"),
        help=(
            "标定哪个 ready 模型。**默认两个都标**——因为默认档用的是手写的 "
            "`HeuristicReadyModel`，而 `models/opponent_model.json`（GBDT）只挂在 `risk` 实验臂上、"
            "**从未被对拍过**。两者在**同一批位置**上比才有意义。"
        ),
    )
    parser.add_argument("--gbdt-path", default="models/opponent_model.json")
    args = parser.parse_args(argv)

    gbdt = None
    if args.model in ("gbdt", "both"):
        try:
            from majiang.strategy.opponent import load_or_none

            gbdt = load_or_none(args.gbdt_path)
            if not getattr(gbdt, "using_model", False):
                print("⚠ GBDT 模型不可用（回退手写），只标手写", file=__import__("sys").stderr)
                gbdt = None
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ 加载 GBDT 失败：{type(exc).__name__}", file=__import__("sys").stderr)
            gbdt = None

    files = rooms_first(sorted(glob.glob("data/auto_sessions/*/events/*.json")), args.rooms)
    if args.limit:
        files = files[: args.limit]
    print(f"按房抽样 {args.rooms} 房（{len(files)} 文件）", file=__import__("sys").stderr)
    rows: dict[tuple[int, int], list[tuple[float, float | None, int]]] = collections.defaultdict(list)
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
            pending: dict[int, tuple[float, int, int, float | None]] = {}
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
                        gbdt_ready = None
                        if gbdt is not None:
                            try:
                                situation = state.situation_for(mine)
                                gbdt_ready = float(gbdt.model.ready_probability(situation, opp))
                            except Exception:  # noqa: BLE001
                                gbdt_ready = None
                        pending[opp] = (
                            predict(len(st.melds), len(st.discards), state.draws),
                            key[0],
                            key[1],
                            gbdt_ready,
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
                            rows[(info[1], info[2])].append(
                                (info[0], info[3], 1 if value == 0 else 0)
                            )
                replay.apply_event(state, e)

    print(f"{'副露':>4s} {'弃牌':>4s} {'n':>6s} {'手写预测':>9s} {'GBDT预测':>9s} "
          f"{'实测听牌率':>11s} {'手写差':>8s} {'GBDT差':>8s}")
    tp = te = tn = 0.0
    gp = gn = 0.0
    for key in sorted(rows):
        data = rows[key]
        if len(data) < 120:  # 样本下限：少于此不报数
            continue
        pred = sum(p for p, _, _ in data) / len(data)
        emp = sum(l for _, _, l in data) / len(data)
        gvals = [g for _, g, _ in data if g is not None]
        gpred = sum(gvals) / len(gvals) if gvals else float("nan")
        tp += sum(p for p, _, _ in data)
        te += sum(l for _, _, l in data)
        tn += len(data)
        if gvals:
            gp += sum(gvals)
            gn += len(gvals)
        print(f"{key[0]:>4d} {key[1]:>4d} {len(data):>6d} {pred:>9.1%} {gpred:>9.1%} "
              f"{emp:>11.1%} {pred - emp:>+8.1%} {gpred - emp:>+8.1%}")
    if tn:
        print(f"\n手写模型合计 n={int(tn)}：{tp / tn:.1%} vs 实测 {te / tn:.1%}  差 {tp / tn - te / tn:+.1%}")
    if gn:
        print(f"GBDT 模型合计 n={int(gn)}：{gp / gn:.1%} vs 实测 {te / tn:.1%}  差 {gp / gn - te / tn:+.1%}")
    print("\n判据：系统性高估 ⇒ 喂牌项被放大、feed_weight 应下调；低估则相反。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
