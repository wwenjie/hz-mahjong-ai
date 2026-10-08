"""出牌层「孤张保护 / 不覆盖主分」条件对拍（回答用户 #7/#9）。

背景（`tools/trigger_census_override.py` 实测 200 房）：v5 的 `_choose_discard` 先按
主分 ``total = -10×向听 + 形质 - 3×喂牌 - 财神罚`` 排序，再由 `_break_ties_by_ukeire`
在**同向听**候选里**只按精确进张**重排并取优 ⇒ **24.6% 的出牌决策主分被覆盖**；
其中主分赢家是**孤张**的占 5.7%（字牌 2.6% / 数牌 3.1%）。这正是用户两次报的
「明显孤牌为啥不打」（#7 打「发」、#9 打孤张）。

本工具与 `cf_peng_seven.py` 同把尺：**不触碰 `src/**`**，guard 只是评测包装器。
- ``baseline`` = v5（含覆盖）。
- ``treatment`` = `IsoGuard(v5, mode)`：
  - ``isoguard``：**仅当主分赢家是孤张**且 v5 选了别的，改回主分赢家（最贴用户原则）。
  - ``nooverride``：只要 treatment 选的那张**主分严格更低**（非真并列），一律改回主分赢家。

用法::
    .venv/bin/python tools/cf_isolated.py --points agent/out/trigger-points/override.jsonl \\
        --mode isoguard --isolated-only --jobs 4
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
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, Action  # noqa: E402
import trigger_counterfactual as tcf  # noqa: E402


def _isolated(counts, tile: int) -> bool:
    if tile == tiles.GOD:
        return False
    if tile >= 27:
        return counts[tile] == 1
    base = (tile // 9) * 9
    for d in (-2, -1, 1, 2):
        u = tile + d
        if base <= u < base + 9 and counts[u] > 0:
            return False
    return True


class IsoGuard:
    """v5 包装器：按 ``mode`` 决定是否把「被破平层覆盖掉的主分赢家」改回来。"""

    def __init__(self, base, mode: str = "isoguard", eps: float = 0.15) -> None:
        self._base = base
        self._mode = mode
        self._eps = eps
        self.name = f"isoguard[{mode}]({getattr(base, 'name', 'v5')})"
        self.last_reason = ""
        self.reverted = 0
        self.seen = 0

    def choose(self, situation, actions, *, budget_ms: int = 0):
        chosen = self._base.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = str(getattr(self._base, "last_reason", "") or "")
        if chosen is None or chosen.kind != DISCARD:
            return chosen
        cands = [a for a in actions if a.kind == DISCARD]
        if len(cands) < 2:
            return chosen
        scored = {a.tile: self._base._score_discard(situation, a) for a in cands}  # noqa: SLF001
        main = max(cands, key=lambda a: scored[a.tile].total)
        if main.tile == chosen.tile:
            return chosen
        if self._mode == "isoguard":
            if not _isolated(situation.hand.counts, main.tile):
                return chosen
        elif self._mode == "nooverride":
            if abs(scored[main.tile].total - scored[chosen.tile].total) <= self._eps:
                return chosen
        self.reverted += 1
        self.last_reason = (
            f"{self._mode}：主分改选 {tiles.to_code(main.tile)}"
            f"（原选 {tiles.to_code(chosen.tile)}）"
        )
        return Action(DISCARD, tile=main.tile)


def _mine_of(file: str) -> int:
    doc = json.loads(Path(file).read_text(encoding="utf-8"))
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    return ids.index(tcf.OUR)


_WORKER: dict = {}


def _init_worker(baseline: str, treatment: str, opponents: str, mode: str, eps: float) -> None:
    _WORKER["deciders"] = {
        "baseline": make_decider(baseline, tcf.Mode.QUALIFIER),
        "treatment": IsoGuard(make_decider(treatment, tcf.Mode.QUALIFIER), mode, eps),
        "opponents": make_decider(opponents, tcf.Mode.QUALIFIER),
    }


def _run_point(payload: tuple[str, dict]) -> dict:
    file, point = payload
    doc = json.loads(Path(file).read_text(encoding="utf-8"))
    out = tcf.run_one(doc, point, _WORKER["deciders"], _mine_of(file), mode="discard")
    guard = _WORKER["deciders"]["treatment"]
    out["reverted_by_guard"] = guard.reverted > 0
    guard.reverted = 0
    return out


def _load_points(path: str, klass: str, isolated_only: bool) -> list[dict]:
    points = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        if klass and klass not in str(p.get("klass", "")):
            continue
        if isolated_only and not _iso_point(p):
            continue
        points.append(p)
    return points


def _iso_point(p: dict) -> bool:
    hc = p.get("hand_counts")
    mt = p.get("main_tile")
    if hc is None or mt is None:
        return False
    return _isolated(hc, int(mt))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="孤张保护 / 不覆盖主分 条件对拍")
    ap.add_argument("--points", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--mode", default="isoguard", choices=("isoguard", "nooverride"))
    ap.add_argument("--isolated-only", action="store_true", help="只对「主分赢家是孤张」的点对拍")
    ap.add_argument("--check", type=int, default=0, help=">0 只跑保真校验")
    ap.add_argument("--baseline", default="v5")
    ap.add_argument("--treatment", default="v5")
    ap.add_argument("--opponents", default="v5")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    points = _load_points(args.points, "", args.isolated_only)
    print(f"数据集 {len(points)} 个触发点（isolated_only={args.isolated_only}）")
    if args.limit > 0:
        points = points[: args.limit]
    if not points:
        print("无触发点")
        return 1

    payloads = [(p["file"], p) for p in points]
    diffs: list[float] = []
    stats: collections.Counter = collections.Counter()
    rows: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(args.baseline, args.treatment, args.opponents, args.mode, args.eps),
        ) as pool:
            for point, out in zip(points, pool.map(_run_point, payloads, chunksize=4)):
                _acc(out, point, stats, diffs, rows)
    else:
        _init_worker(args.baseline, args.treatment, args.opponents, args.mode, args.eps)
        for payload, point in zip(payloads, points):
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
        n_rev = stats["guard生效"]
        print(f"  guard 实际改判 {n_rev}/{len(diffs)}（{n_rev / len(diffs):.1%}）")
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
    if out.get("conserved"):
        stats["守恒OK"] += 1
    if out.get("reverted_by_guard"):
        stats["guard生效"] += 1
    rows.append({**{k: point.get(k) for k in ("room", "round_no", "block", "ev_index")}, **out})


if __name__ == "__main__":
    raise SystemExit(main())
