"""批量期望值层：对一类触发点**逐点**用蒙特卡洛算「期望得分 regret」（用户 2026-10-09 11:07 批准的方向）。

**为什么不是再跑一次整场 A/B**：用户与论文都指向同一结论——「速度 vs 宽度」是
**非线性的期望值权衡**，用固定线性权重（`-10×向听 + 两拍值`）混合必然测不出信号
（A 的 `cf-early` 真实续跑 t−1.70、`two-ply` 5 种子 t1.36 都是这个建模选择的预期后果）。
本工具换口径：**在单个触发点上，用 MC 直接估「打 A 的期望得分」与「打 B 的期望得分」**，
给出 regret 分布——这就是「精准量化权衡」。

**输入**：触发点 JSONL（如 `tools/trigger_census_early.py` 的产出，逐点带
`file`/`block`/`ev_index`/`v5_tile`/`arm_tile`）。
**方法**：对每个点用 `trigger_counterfactual.rebuild` 重建到决策点 → **只用公开信息**
重采样其余三家暗手与牌墙 N 次 → 在同一批确定化世界里分别强制打 `v5_tile` / `arm_tile`
并滚到底 → 配对差 `arm − v5`。
**合规**：采样只用公开信息；策略从不读真实对手手牌；每样本守恒 136 硬校验。

用法::
    .venv/bin/python tools/cf_ev_batch.py --points agent/out/trigger-points/early-sample200.jsonl \\
        --samples 40 --jobs 14 --out agent/out/trigger-points/ev-early-200.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from majiang.rules import tiles  # noqa: E402
from majiang.sim import round as R  # noqa: E402
import trigger_counterfactual as tcf  # noqa: E402
from cf_point_mc import public_counts  # noqa: E402

TOTAL = tiles.TILE_KINDS * tiles.COPIES_PER_KIND


def _localize_point_file(raw: str) -> str:
    """把触发点里**另一台机器**的绝对路径映射回本仓。

    本仓的 `trigger_counterfactual.resolve_point_file` 用 `path.exists()` 解析——
    但当路径指向**当前用户无权限 `stat` 的目录**（如远端的 `/root/...`）时会抛
    `PermissionError` 而非返回 False（2026-10-09 11:12 实测）。本工具不依赖那把尺，
    直接把 `/…/data/auto_sessions/…` 后缀挂回本仓 ROOT。
    """
    raw = str(raw)
    mark = "/data/auto_sessions/"
    idx = raw.find(mark)
    if idx != -1:
        local = ROOT / raw[idx + 1 :]
        if local.is_file():
            return str(local)
    return raw


def _deal(pool: list[int], sizes: list[int], mine: int, rng: random.Random):
    work = list(pool)
    rng.shuffle(work)
    hands = {}
    idx = 0
    for seat in range(4):
        if seat == mine:
            continue
        n = sizes[seat]
        hands[seat] = work[idx : idx + n]
        idx += n
    return hands, work[idx:]


def _one_world(real: R.RoundState, mine: int, cands: tuple[int, ...], unknown: list[int],
               sizes: list[int], seed: int, mode) -> dict:
    from majiang.cli import make_decider

    rng = random.Random(seed)
    hands, wall = _deal(unknown, sizes, mine, rng)
    seats = []
    for seat in range(4):
        if seat == mine:
            hand = list(real.seats[seat].hand)
        else:
            counts = [0] * tiles.TILE_KINDS
            for t in hands[seat]:
                counts[t] += 1
            hand = counts
        seats.append(R.Seat(
            hand=hand,
            melds=list(real.seats[seat].melds),
            discards=list(real.seats[seat].discards),
            chain_count=real.seats[seat].chain_count,
            piao_count=real.seats[seat].piao_count,
            god_count=real.seats[seat].god_count,
        ))
    state = R.RoundState(
        wall=list(wall), seats=seats, dealer=real.dealer, round_no=real.round_no,
        turn=mine, catch_play=real.catch_play, god_discarder=real.god_discarder,
        prior_scores=real.prior_scores, rounds_total=real.rounds_total,
    )
    cons = len(state.wall) + sum(
        sum(ss.hand) + len(ss.discards) + sum(len(m.tiles) for m in ss.melds)
        for ss in state.seats
    )
    if cons != TOTAL:
        return {}
    me = make_decider("v5", mode)
    opp = me
    pick = [me if s == mine else opp for s in range(4)]
    out = {}
    for tile in cands:
        st = R.RoundState(
            wall=list(state.wall),
            seats=[R.Seat(hand=list(ss.hand), melds=list(ss.melds), discards=list(ss.discards),
                          chain_count=ss.chain_count, piao_count=ss.piao_count,
                          god_count=ss.god_count) for ss in state.seats],
            dealer=state.dealer, round_no=state.round_no, turn=mine,
            catch_play=state.catch_play, god_discarder=state.god_discarder,
            prior_scores=state.prior_scores, rounds_total=state.rounds_total,
        )
        if st.seats[mine].hand[tile] <= 0:
            continue
        R.apply_discard(st, mine, tile, None)
        claim = R.resolve_responses(st, mine, tile, pick)
        nxt, need = ((mine + 1) % 4, True) if claim is None else claim
        try:
            outcome = R.play_round(st, pick, current=nxt, drawn=None, need_draw=need)
        except Exception:  # noqa: BLE001
            continue
        out[tile] = (outcome.scores[mine], outcome.winner == mine)
    return out


_MODE = None


def _init(mode_name: str):
    global _MODE
    from majiang.strategy.policy import Mode

    _MODE = {"qualifier": Mode.QUALIFIER, "final": Mode.FINAL}[mode_name]


def _worker(payload: tuple) -> dict:
    file, point, samples, seed0 = payload
    local = _localize_point_file(file)
    try:
        doc = json.loads(Path(local).read_text(encoding="utf-8"))
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if tcf.OUR not in ids:
            return {"ok": False, "why": "非我方房"}
        mine = ids.index(tcf.OUR)
        real, _events, _anom, _drawn = tcf.rebuild(doc, point)
        unknown, sizes = public_counts(real, mine)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "why": f"重建/池:{type(error).__name__}"}
    cands = tuple(int(t) for t in point["cands"])
    diffs = []
    for i in range(samples):
        try:
            out = _one_world(real, mine, cands, unknown, sizes, seed0 + i, _MODE)
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "why": f"世界:{type(error).__name__}"}
        if len(out) == len(cands):
            a = out[cands[0]][0]   # v5_tile（基准）
            b = out[cands[-1]][0]  # arm_tile（反事实）
            diffs.append(b - a)
    return {"ok": True, "room": point.get("room"), "file": local, "block": point.get("block"),
            "ev_index": point.get("ev_index"), "current_shanten": point.get("current_shanten"),
            "god_n": point.get("god_n"), "ratio": point.get("ratio"),
            "cands": [int(c) for c in cands], "n": len(diffs), "diffs": diffs,
            "mean": sum(diffs) / len(diffs) if diffs else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="批量期望值 regret（MC，配对）")
    ap.add_argument("--points", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--mode", default="qualifier", choices=("qualifier", "final"))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    rows = []
    for line in Path(args.points).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        v5 = p.get("v5_tile")
        arm = p.get("arm_tile")
        if v5 is None or arm is None or int(v5) == int(arm):
            continue
        p["cands"] = [int(v5), int(arm)]
        rows.append(p)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"触发点 {len(rows)}（v5_tile vs arm_tile）样本 {args.samples} 并行 {args.jobs}")

    payloads = [(p["file"], p, args.samples, args.seed * 100000 + i) for i, p in enumerate(rows)]
    results = []
    fails = collections.Counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                 initargs=(args.mode,)) as pool:
            for res in pool.map(_worker, payloads, chunksize=1):
                if res.get("ok"):
                    results.append(res)
                else:
                    fails[res.get("why")] += 1
    else:
        _init(args.mode)
        for p in payloads:
            res = _worker(p)
            if res.get("ok"):
                results.append(res)
            else:
                fails[res.get("why")] += 1

    all_diffs = [d for r in results for d in r["diffs"]]
    if all_diffs:
        mean = sum(all_diffs) / len(all_diffs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in all_diffs) / (len(all_diffs) - 1))
        se = sd / math.sqrt(len(all_diffs))
        print(f"汇总（pooled，{len(results)} 点 / {len(all_diffs)} 世界）："
              f"regret(arm−v5) = {mean:+.3f}  se {se:.3f}  t {mean/se if se else float('nan'):+.2f}")
        print(f"  <0 {sum(1 for x in all_diffs if x < 0)} / =0 {sum(1 for x in all_diffs if x == 0)}"
              f" / >0 {sum(1 for x in all_diffs if x > 0)}")
        pt = [r["mean"] for r in results if r["mean"] is not None]
        if pt:
            pm = sum(pt) / len(pt)
            ps = math.sqrt(sum((x - pm) ** 2 for x in pt) / (len(pt) - 1)) if len(pt) > 1 else 0.0
            print(f"  逐点均值（n={len(pt)}）：{pm:+.3f}  se {ps/math.sqrt(len(pt)):.3f}"
                  f"  （点为正 {sum(1 for x in pt if x > 0)} / 为负 {sum(1 for x in pt if x < 0)}）")
    if fails:
        print(f"  失败: {dict(fails)}")
    if args.out and results:
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")
        print(f"  逐点: {args.out}（{len(results)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
