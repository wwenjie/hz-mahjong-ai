"""D1 · threat 层标定的**独立复核**（agent-c，第二层口径；不 import A 的 tools/calibrate_threat.py）。

**要回答的问题（A 01:40 / 03:15 的读数复核）**：
`HeuristicReadyModel.estimate` 报的「对手已听」概率，相对实测是否系统性高估？
A 的读数：手写 36.1% vs 实测 12.4%（n=7045，≈2.9 倍）；受训 GBDT 21.3% vs 13.2%（≈1.6 倍）。

**口径与 A 的实现的**分岔点**（我逐条独立实现，不复用其代码）：
1. 抽样：A 用 `sorted(glob)[:limit]` ⇒ **按字母序取前 N 个房**（不是随机/分层）。
   我改用**按 seed 确定性随机抽房**，并报抽到的房列表指纹，避免「名字靠前的房」这一潜在偏置。
2. 单位/标签：与 A 同——在我方每个出牌点记每个对手的公开特征 + 模型 ready，
   以**该对手下一次出牌**时 `shanten_any == 0` 为标签。
   **注意这是可观测代理，不是「下次摸牌后是否听」的无偏标签**：听牌者若自摸即离场，
   会被从样本中去掉 ⇒ 实测率**偏低估**。所以 2.9 倍这个数**方向可信、量级是上界**。
   我把这条写进结论，不当作已证明的量级。
3. 统计：报每桶 n、点估计、二项 95% CI；合计报加权差与比值，并给 n。

**只读**：只读 `data/auto_sessions/**` 事件流与主仓 `src/majiang/**`；零平台请求；不写任何台账。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
import sys
from pathlib import Path

from majiang.rules import shanten as sm
from majiang.sim import replay

OUR = "u_a7f7c67bb14a"

# 与 src/majiang/strategy/risk.py 的常数逐字对齐（不凭记忆）
BASE = 0.06
W_DISCARD = 0.30
W_MELD = 0.16
PROGRESS_W = 0.18
READY_MAX = 0.85
READY_MIN = 0.02
DRAW_SPAN = 16.0
MIN_BUCKET = 120  # 与 A 一致：每桶至少这么多样本才报数


def predict(melds: int, discards: int, draws: int) -> float:
    progress = min(1.0, draws / DRAW_SPAN)
    ready = BASE + W_DISCARD * (1.0 - math.exp(-discards / 6.0)) + W_MELD * melds + PROGRESS_W * progress
    return min(READY_MAX, max(READY_MIN, ready))


def _room_era(path: str) -> str:
    """房目录 mtime 的 MM-DD（用于按 era 分层——A 的 limit 抽样实际只覆盖最早几房）。"""
    import time

    return time.strftime("%m-%d", time.localtime(Path(path).parent.parent.stat().st_mtime))


def binom_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wald 区间（够用；n≥百量级）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return (p - z * se, p + z * se)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="threat 层标定独立复核")
    ap.add_argument("--rooms", type=int, default=150, help="抽多少个房")
    ap.add_argument("--sample", default="random", choices=("random", "first"),
                    help="random=按 seed 随机抽（默认）；first=sorted(glob)[:N]，复刻 A 的抽样以便定位口径差")
    ap.add_argument("--era", default="", help="只取该日期的房（按房目录 mtime 的 MM-DD）；空=全部")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--model", default="both", choices=("heuristic", "gbdt", "both"))
    ap.add_argument("--out", default="agent/out/audit-threat.json", help="结果落盘路径")
    ap.add_argument("--gbdt-path", default="models/opponent_model.json")
    args = ap.parse_args(argv)
    out_path = args.out

    gbdt = None
    if args.model in ("gbdt", "both"):
        try:
            from majiang.strategy.opponent import load_or_none

            gbdt = load_or_none(args.gbdt_path)
            if not getattr(gbdt, "using_model", False):
                print("⚠ GBDT 不可用（回退手写），只标手写", file=sys.stderr)
                gbdt = None
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ 加载 GBDT 失败：{type(exc).__name__}", file=sys.stderr)
            gbdt = None

    allfiles = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    if args.era:
        allfiles = [f for f in allfiles if _room_era(f) == args.era]
        print(f"era={args.era}：过滤后 {len(allfiles)} 文件", flush=True)
    if args.sample == "first":
        picked = allfiles[: args.rooms]
        print(f"候选 {len(allfiles)} 房 · 取前 {len(picked)} 个（sorted 序，复刻 A）", flush=True)
    else:
        rng = random.Random(args.seed)
        picked = rng.sample(allfiles, min(args.rooms, len(allfiles)))
        print(f"候选 {len(allfiles)} 房 · 随机抽 {len(picked)} 房（seed={args.seed}）", flush=True)

    rows: dict[tuple[int, int], list[tuple[float, float | None, int]]] = collections.defaultdict(list)
    used = 0
    skipped = 0
    for i, path in enumerate(picked, 1):
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            skipped += 1
            continue
        used += 1
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
                        g_ready = None
                        if gbdt is not None:
                            try:
                                g_ready = float(gbdt.model.ready_probability(state.situation_for(mine), opp))
                            except Exception:  # noqa: BLE001
                                g_ready = None
                        pending[opp] = (
                            predict(len(st.melds), len(st.discards), state.draws),
                            min(len(st.melds), 3),
                            min(len(st.discards), 12),
                            g_ready,
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
                            rows[(info[1], info[2])].append((info[0], info[3], 1 if value == 0 else 0))
                replay.apply_event(state, e)
        if i % 25 == 0:
            n = sum(len(v) for v in rows.values())
            print(f"  …{i}/{len(picked)} 房，n={n}", flush=True)

    print(f"\n参与统计 {used} 房（跳过 {skipped}）")
    print(f"{'副露':>4s} {'弃牌':>4s} {'n':>7s} {'手写预测':>9s} {'GBDT预测':>9s} {'实测听牌率':>11s} {'手写差':>8s} {'GBDT差':>8s}")
    tp = te = tn = 0.0
    gp = gn = 0.0
    for key in sorted(rows):
        data = rows[key]
        if len(data) < MIN_BUCKET:
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
        print(f"{key[0]:>4d} {key[1]:>4d} {len(data):>7d} {pred:>9.1%} {gpred:>9.1%} {emp:>11.1%} "
              f"{pred - emp:>+8.1%} {gpred - emp:>+8.1%}")

    out: dict[str, object] = {"rooms_candidates": len(allfiles), "rooms_picked": len(picked),
                              "rooms_used": used, "seed": args.seed, "min_bucket": MIN_BUCKET,
                              "our_id": OUR}
    if tn:
        hp, ep = tp / tn, te / tn
        lo, hi = binom_ci(te, int(tn))
        ratio = hp / ep if ep else float("nan")
        print(f"\n手写合计 n={int(tn)}：{hp:.2%} vs 实测 {ep:.2%}  差 {hp - ep:+.2%}  比值 {ratio:.2f}×")
        print(f"  实测点估计 {ep:.2%}，二项95%CI [{lo:.2%}, {hi:.2%}]（区间只覆盖抽样误差，不含上述标签偏置）")
        out["heuristic"] = {"n": int(tn), "pred": hp, "emp": ep, "diff": hp - ep, "ratio": ratio,
                            "emp_ci95": [lo, hi]}
    if gn:
        gq, eq = gp / gn, te / tn
        print(f"GBDT 合计 n={int(gn)}：{gq:.2%} vs 实测 {eq:.2%}  差 {gq - eq:+.2%}  比值 "
              f"{(gq / eq if eq else float('nan')):.2f}×")
        out["gbdt"] = {"n": int(gn), "pred": gq, "emp": eq, "ratio": (gq / eq if eq else None)}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
