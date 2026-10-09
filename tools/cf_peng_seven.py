"""第三类触发点条件对拍：**响应窗口「碰毁七对」**（回答用户第③问）。

对拍口径与 ``tools/trigger_counterfactual.py --mode response`` 一致：
两支各重建局面，**我方座位**分别放 baseline（v5）与 treatment（七对保护档），
其余三家固定 v5，续跑到局末，比较我方本局净分。

``treatment`` = 本工具内置的 ``_PairGuard``：**只在**「响应窗口、可碰、且碰会毁掉
七对（`seven_pairs_shanten <= sp-max`）」时，把 v5 的 PENG 改成 PASS；其余完全跟随 v5。
这样测的就是「给碰加一道七对保护」在触发点上的净效应——与①②同一把尺。

**不触碰 `src/**`**：guard 是评测用包装器，只存在于本工具里；若结论为正，
再按 A 的流程在 `src/` 里落成默认关的开关。

用法::
    .venv/bin/python tools/cf_peng_seven.py --points agent/out/trigger-points/peng.jsonl --limit 200 --jobs 6
    .venv/bin/python tools/cf_peng_seven.py --points ... --klass 碰毁七对\\(v5会碰\\) --limit 0
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import PENG, PASS, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_RESPONSE_PENG  # noqa: E402
from majiang.sim import round as R  # noqa: E402
import trigger_counterfactual as tcf  # noqa: E402

OUR = "u_a7f7c67bb14a"
TOTAL_TILES = 136


class PairGuard:
    """v5 包装器：把「会毁掉七对的碰」改成 PASS。评测用，不进 `src/`。"""

    def __init__(self, base, sp_max: int = 2, pair_route_pairs: int = 5,
                 mode: str = "guard") -> None:
        self._base = base
        self._sp_max = sp_max
        self._prp = pair_route_pairs
        self._mode = mode  # guard=碰→过 ；force=过→碰（测用户第③问的反方向）
        self.name = ("pairguard" if mode == "guard" else "forcepeng") + "(" + getattr(base, "name", "v5") + ")"
        self.last_reason = ""

    def choose(self, situation, actions, *, budget_ms: int = 0):
        chosen = self._base.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = str(getattr(self._base, "last_reason", "") or "")
        if situation.phase != PHASE_RESPONSE_PENG or chosen is None:
            return chosen
        counts = situation.hand.counts
        sp = shanten_mod.seven_pairs_shanten(counts)
        pairs = sum(c // 2 for c in counts)
        if not (sp <= self._sp_max and situation.hand.meld_count == 0 and pairs < self._prp):
            return chosen
        code = tiles.to_code(situation.offered_tile) if situation.offered_tile is not None else "?"
        peng = next((a for a in actions if a.kind == PENG), None)
        if self._mode == "guard" and chosen.kind == PENG:
            fallback = next((a for a in actions if a.kind == PASS), None)
            if fallback is not None:
                self.last_reason = f"七对保护(评测)：七对{sp} 放弃碰 {code}"
                return fallback
        elif self._mode == "force" and chosen.kind != PENG and peng is not None:
            self.last_reason = f"强制碰(评测)：七对{sp} 对数{pairs} 碰 {code}"
            return peng
        return chosen


def _mine_of(file: str) -> int:
    doc = json.loads(Path(file).read_text(encoding="utf-8"))
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    return ids.index(OUR)


def run_one(doc: dict, point: dict, deciders: dict, mine: int) -> dict:
    """两支各重建、续跑到局末；分支差只在「我方座位用哪个决策器」。"""
    offered = point["offered"]
    discarder = int(point["discarder"])
    out: dict = {}
    for branch in ("baseline", "treatment"):
        try:
            state, _events, _anomalies, drawn = tcf.rebuild(doc, point)
            ours = deciders[branch]
            pick = [ours if s == mine else deciders["opponents"] for s in range(4)]
            R.apply_discard(state, discarder, offered, drawn)
            claim = R.resolve_responses(state, discarder, offered, pick)
            if claim is None:
                nxt, need_draw = (discarder + 1) % 4, True
            else:
                nxt, need_draw = claim
            outcome = R.play_round(state, pick, current=nxt, drawn=None, need_draw=need_draw)
            out[branch] = {
                "score": outcome.scores[mine],
                "win": outcome.winner == mine,
                "flow": outcome.is_flow,
                "conservation_ok": tcf.conservation(state) == TOTAL_TILES,
            }
        except tcf.RebuildError as error:
            return {"ok": False, "why": f"不可重建: {error}"}
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "why": f"异常:{type(error).__name__}:{error}"}
    base = out["baseline"]
    treat = out["treatment"]
    return {
        "ok": True,
        "diff": treat["score"] - base["score"],
        "baseline_score": base["score"],
        "treatment_score": treat["score"],
        "baseline_win": base["win"],
        "treatment_win": treat["win"],
        "flow": base["flow"] and treat["flow"],
        "conservation_ok": base["conservation_ok"] and treat["conservation_ok"],
    }


_WORKER: dict = {}


def _init_worker(baseline: str, treatment: str, opponents: str, sp_max: int, prp: int,
                 mode: str) -> None:
    _WORKER["deciders"] = {
        "baseline": make_decider(baseline, tcf.Mode.QUALIFIER),
        "treatment": PairGuard(make_decider(treatment, tcf.Mode.QUALIFIER), sp_max, prp, mode),
        "opponents": make_decider(opponents, tcf.Mode.QUALIFIER),
    }


def _run_point(payload: tuple[str, dict]) -> dict:
    file, point = payload
    doc = json.loads(Path(file).read_text(encoding="utf-8"))
    return run_one(doc, point, _WORKER["deciders"], _mine_of(file))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="碰毁七对 条件对拍")
    ap.add_argument("--points", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--klass", default="", help="只对这类触发点对拍（空 = 全部）")
    ap.add_argument("--baseline", default="v5")
    ap.add_argument("--treatment", default="v5", help="guard 的内层档（默认 v5）")
    ap.add_argument("--opponents", default="v5")
    ap.add_argument("--sp-max", type=int, default=2)
    ap.add_argument("--prp", type=int, default=5, help="七对保护门槛（对数 < 此值才保护）")
    ap.add_argument("--mode", default="guard", choices=("guard", "force"),
                    help="guard=碰→过（测'该少碰'）；force=过→碰（测用户第③问'该多碰'）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    points = [json.loads(x) for x in Path(args.points).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.klass:
        points = [p for p in points if p.get("klass") == args.klass]
    print(f"数据集 {len(points)} 个触发点（klass={args.klass or '全部'}）")
    if args.limit > 0:
        points = points[: args.limit]

    payloads = [(p["file"], p) for p in points]
    diffs: list[float] = []
    stats: collections.Counter = collections.Counter()
    rows: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(args.baseline, args.treatment, args.opponents, args.sp_max, args.prp, args.mode),
        ) as pool:
            results = pool.map(_run_point, payloads, chunksize=4)
            for point, out in zip(points, results):
                _acc(out, point, stats, diffs, rows)
    else:
        _init_worker(args.baseline, args.treatment, args.opponents, args.sp_max, args.prp, args.mode)
        for point, payload in zip(points, payloads):
            _acc(_run_point(payload), point, stats, diffs, rows)

    if diffs:
        mean = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
        se = sd / math.sqrt(len(diffs)) if diffs else 0.0
        t = mean / se if se else 0.0
        print(f"有效点 {len(diffs)}；失败 {stats['失败']}")
        print(f"我方本局净分差（treatment−baseline）= {mean:+.3f}  se {se:.3f}  t {t:+.2f}")
        if len(diffs) > 1:
            print(f"  95%CI [{mean - 1.96 * se:+.3f}, {mean + 1.96 * se:+.3f}]")
        print(f"  分布：<0 {sum(1 for d in diffs if d < 0)} / =0 {sum(1 for d in diffs if d == 0)} / >0 {sum(1 for d in diffs if d > 0)}")
        print(f"  conservation 全 OK? {stats['守恒OK']}/{len(diffs)}")
    else:
        print(f"无可对拍点（失败 {stats['失败']}）")
    if args.out and rows:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"逐点结果: {args.out}（{len(rows)} 行）")
    return 0


def _acc(out: dict, point: dict, stats, diffs, rows) -> None:
    if not out.get("ok"):
        stats["失败"] += 1
        return
    diffs.append(out["diff"])
    if out.get("conservation_ok"):
        stats["守恒OK"] += 1
    rows.append({**{k: point.get(k) for k in ("room", "round_no", "block", "ev_index", "klass")}, **out})


if __name__ == "__main__":
    raise SystemExit(main())
