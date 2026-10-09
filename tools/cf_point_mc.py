"""单点蒙特卡洛「构造场景 → 算胜率」工具（回答用户 2026-10-09 10:55：条件对拍=构造特定场景计算胜率）。

**是什么**：给定一份网页报障快照（`webapp/reports/*.json`，含**有序牌墙**），
对指定的弃牌候选，做 N 次「确定化 + 滚出」：用**公开信息**重新采样其余三家的暗手与
牌墙顺序，再让 `v5`（或指定档）把本局滚到底，统计**我方胜率**与**期望得分**。

**与 `tools/trigger_counterfactual.py` 的区别**（两者互补，不要混用）：
- `trigger_counterfactual.py`：**真实续跑**——用录制牌墙，一次一个确定结果（单样本，
  沿用真机未来）。适合「这批点整体上 A 与 B 谁更好」的均值检验。
- 本工具：**蒙特卡洛**——同一个局面重采样 N 次，给的是**该点的胜率/期望分估计**。
  适合「这一手，打 A 还是打 B」的**单点期望值**比较（用户要的正是这个）。

**合规**：采样只用公开信息（自己暗手 + 四家副露 + 四家弃牌 + 各家暗手张数 + 牌墙剩余）。
策略侧**从不**读真实对手手牌；确定化出来的手牌是**采样**，不是观测。

**正确性护栏**（若任一不成立即报错，不出结论）：
- 未知池张数 = 136 −（我方暗手 + 全部副露 + 全部弃牌）；
- 各家暗手张数按公开规则（轮到自己 = 14−3×副露，其余 = 13−3×副露）；
- 每样本重分配后，`Σ(四家暗手)+副露+弃牌+牌墙 = 136` 必须守恒。

用法::
    .venv/bin/python tools/cf_point_mc.py --report webapp/reports/report_..._seq71.json \\
        --candidates 2t,7w --samples 300 --jobs 8
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

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, Action, legal_actions  # noqa: E402
from majiang.rules.melds import parse_melds  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import round as R  # noqa: E402

TOTAL = tiles.TILE_KINDS * tiles.COPIES_PER_KIND


def _pool() -> list[int]:
    return [t for t in range(tiles.TILE_KINDS) for _ in range(tiles.COPIES_PER_KIND)]


def _parse_melds_full(raw: list[dict]) -> tuple:
    """full_state 的副露格式 = [{"kind": "peng", "tiles": ["西","西","西"]}, ...]。"""
    return tuple(parse_melds(raw or ()))


def _build_from_full(fs: dict) -> R.RoundState:
    seats = []
    for s in fs["seats"]:
        seats.append(
            R.Seat(
                hand=list(s["hand"]),
                melds=list(_parse_melds_full(s.get("melds") or [])),
                discards=list(s.get("discards") or []),
                chain_count=int(s.get("chain_count", 0)),
                piao_count=int(s.get("piao_count", 0)),
                god_count=int(s.get("god_count", 0)),
            )
        )
    return R.RoundState(
        wall=list(fs["wall"]),
        seats=seats,
        dealer=int(fs["dealer"]),
        round_no=int(fs["round_no"]),
        turn=int(fs["turn"]),
        catch_play=bool(fs.get("catch_play", False)),
        god_discarder=int(fs.get("god_discarder", -1)),
        prior_scores=tuple(fs.get("prior_scores") or ()),
        rounds_total=int(fs.get("rounds_total", 0) or 0),
    )


def _public_counts(doc: dict, mine: int, real: R.RoundState) -> tuple[list[int], list[int]]:
    """返回（未知池, 四家暗手张数）。未知池 = 136 − 我方暗手 − 全部副露 − 全部弃牌。

    暗手张数：轮到自己 = 14 − 3×副露，其余 = 13 − 3×副露（公开可数）。
    """
    pool = _pool()
    known = collections.Counter()
    # 我方暗手 + 全部副露 + 全部弃牌都是公开信息
    for t, n in enumerate(real.seats[mine].hand):
        known[t] += n
    for ss in real.seats:
        for m in ss.melds:
            for t in m.tiles:
                known[t] += 1
        for t in ss.discards:
            known[t] += 1
    remaining = []
    for t, n in known.items():
        for _ in range(n):
            pool.remove(t)
    remaining = pool
    sizes = []
    for seat in range(4):
        melds = len(real.seats[seat].melds)
        base = 13 - 3 * melds
        if seat == mine:
            base += 1  # 已摸牌、待弃
        sizes.append(base)
    return remaining, sizes


def _deal(unknown: list[int], sizes: list[int], mine: int, rng: random.Random):
    """重采样：把未知池切成各家暗手与牌墙（顺序随机）。返回（hands_by_seat, wall）。"""
    pool = list(unknown)
    rng.shuffle(pool)
    hands = {}
    idx = 0
    for seat in range(4):
        if seat == mine:
            continue
        n = sizes[seat]
        hands[seat] = pool[idx : idx + n]
        idx += n
    wall = pool[idx:]
    return hands, wall


def _rollout_one(doc: dict, mine: int, tile: int, seed: int, baseline: str, opponents: str,
                 _cache: dict | None = None) -> dict:
    """一次确定化 + 滚出：我方强制打 `tile`，其余按 baseline 决策，滚到局末。"""
    fs = doc["full_state"]
    real = _build_from_full(fs)
    rng = random.Random(seed)
    unknown, sizes = _public_counts(doc, mine, real)
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
        seats.append(
            R.Seat(
                hand=hand,
                melds=list(real.seats[seat].melds),
                discards=list(real.seats[seat].discards),
                chain_count=real.seats[seat].chain_count,
                piao_count=real.seats[seat].piao_count,
                god_count=real.seats[seat].god_count,
            )
        )
    state = R.RoundState(
        wall=list(wall), seats=seats, dealer=real.dealer, round_no=real.round_no,
        turn=mine, catch_play=real.catch_play, god_discarder=real.god_discarder,
        prior_scores=real.prior_scores, rounds_total=real.rounds_total,
    )
    cons = len(state.wall)
    for ss in state.seats:
        cons += sum(ss.hand) + len(ss.discards) + sum(len(m.tiles) for m in ss.melds)
    if cons != TOTAL:
        return {"ok": False, "why": f"守恒 {cons}", "win": None, "score": None}

    me = make_decider(baseline, _MODE)
    opp = make_decider(opponents, _MODE)
    pick = [me if s == mine else opp for s in range(4)]

    sit = R.situation_for(state, mine, PHASE_DRAW, drawn=None)
    legal = tuple(legal_actions(sit))
    chosen = next((a for a in legal if a.kind == DISCARD and a.tile == tile), None)
    if chosen is None:
        return {"ok": False, "why": "候选不在合法内", "win": None, "score": None}
    R.apply_discard(state, mine, tile, None)
    claim = R.resolve_responses(state, mine, tile, pick)
    nxt, need = ((mine + 1) % 4, True) if claim is None else claim
    try:
        outcome = R.play_round(state, pick, current=nxt, drawn=None, need_draw=need)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "why": f"{type(error).__name__}", "win": None, "score": None}
    return {"ok": True, "win": outcome.winner == mine, "score": outcome.scores[mine],
            "flow": outcome.is_flow}


_MODE = None


def _init(mode_name: str, baseline: str, opponents: str):
    global _MODE
    from majiang.strategy.policy import Mode

    _MODE = {"qualifier": Mode.QUALIFIER, "final": Mode.FINAL}[mode_name]


_DOC: dict = {}


def _worker(payload: tuple) -> dict:
    """同一确定化世界下评所有候选（seed 决定世界 ⇒ 候选间配对）。"""
    report, mine, cands, seed, baseline, opponents = payload
    if report not in _DOC:
        _DOC[report] = json.loads(Path(report).read_text(encoding="utf-8"))
    doc = _DOC[report]
    return {int(t): _rollout_one(doc, mine, int(t), seed, baseline, opponents) for t in cands}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="单点蒙特卡洛 胜率/期望分")
    ap.add_argument("--report", required=True)
    ap.add_argument("--candidates", required=True, help="逗号分隔牌码，如 2t,7w；或 all")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--baseline", default="v5")
    ap.add_argument("--opponents", default="v5")
    ap.add_argument("--mode", default="qualifier", choices=("qualifier", "final"))
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.report).read_text(encoding="utf-8"))
    ids = [str(s.get("user_id", "")) for s in (doc.get("full_state", {}) or {}).get("seats", [])]
    # 我方 = 报障会话的人类座位
    mine = int(doc["session"]["human_seat"])

    # 候选
    if args.candidates == "all":
        from replay_report import _situation_from_view  # noqa

        sit = _situation_from_view(doc["decision"]["situation"])
        cands = sorted({a.tile for a in legal_actions(sit) if a.kind == DISCARD})
    else:
        cands = [tiles.parse(c.strip()) for c in args.candidates.split(",") if c.strip()]

    print(f"报告 {Path(args.report).name}  我方座位={mine}  候选={[tiles.to_code(c) for c in cands]}"
          f"  样本={args.samples}  对手={args.opponents}")

    codes = [tiles.to_code(c) for c in cands]
    payloads = [(args.report, mine, tuple(cands), args.seed * 100000 + i, args.baseline, args.opponents)
                for i in range(args.samples)]
    scores_by: dict[int, list[float]] = {int(c): [] for c in cands}
    wins_by: dict[int, int] = {int(c): 0 for c in cands}
    fails: collections.Counter = collections.Counter()
    paired: list[dict] = []  # 每个世界：{cand: score}
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                 initargs=(args.mode, args.baseline, args.opponents)) as pool:
            for world in pool.map(_worker, payloads, chunksize=4):
                row = {}
                for t, res in world.items():
                    if res["ok"]:
                        scores_by[t].append(res["score"])
                        wins_by[t] += 1 if res["win"] else 0
                        row[t] = res["score"]
                    else:
                        fails[res["why"]] += 1
                if len(row) == len(cands):
                    paired.append(row)
    else:
        _init(args.mode, args.baseline, args.opponents)
        for p in payloads:
            world = _worker(p)
            row = {}
            for t, res in world.items():
                if res["ok"]:
                    scores_by[t].append(res["score"])
                    wins_by[t] += 1 if res["win"] else 0
                    row[t] = res["score"]
                else:
                    fails[res["why"]] += 1
            if len(row) == len(cands):
                paired.append(row)

    for t in cands:
        vals = scores_by[int(t)]
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n else float("nan")
        print(f"  打 {tiles.to_code(t):>3}: 有效 {n:>4}  胜率 {wins_by[int(t)]/n if n else float('nan'):.3f}"
              f"  期望得分 {mean:+.3f} (se {se:.3f})")
    # 配对标：以第一个候选为基准
    base_t = int(cands[0])
    print(f"  配对差值（vs 打 {tiles.to_code(base_t)}，同确定化世界，n={len(paired)}）：")
    for t in cands[1:]:
        diffs = [row[int(t)] - row[base_t] for row in paired]
        n = len(diffs)
        mean = sum(diffs) / n if n else float("nan")
        sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n else float("nan")
        print(f"    打 {tiles.to_code(t)} − 打 {tiles.to_code(base_t)}: {mean:+.3f} (se {se:.3f}"
              f", t {mean/se if se else float('nan'):+.2f})")
    if fails:
        print(f"  失败明细: {dict(fails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
