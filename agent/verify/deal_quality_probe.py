"""C28：发牌质量对照——检验 C27 的「慢到听」是不是「牌更差」造成的。

只读、零平台请求。对每个座位-局：
1. 发牌后（本人尚未出牌）的**初始向听数**（庄 14 张 → 枚举打出一张取最小）；
2. 本局该座位**出牌次数**（= 摸牌节奏）；
3. 本局总摸数（round 长度）。

若我方初始向听分布与对手一致 ⇒ 到听慢不是牌差，而是出牌/留牌决策。

用法：``.venv/bin/python agent/verify/deal_quality_probe.py [--rooms N]``
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from math import erf, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
DISCARDED = "tile_discarded"


def _phat(p: float) -> float:
    return 0.5 * (1.0 + erf(p / sqrt(2.0)))


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    se = sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    if not se:
        return float("nan"), float("nan")
    t = (st.mean(a) - st.mean(b)) / se
    return t, 2 * (1 - _phat(abs(t)))


def initial_shanten(counts: list[int], memo: dict) -> int | None:
    n = sum(counts)
    if n == 13:
        try:
            return shanten_mod.shanten(counts, 0, memo=memo)
        except Exception:  # noqa: BLE001
            return None
    if n == 14:  # 庄家：枚举打出一张后取最小
        best = None
        for t, c in enumerate(counts):
            if c <= 0:
                continue
            counts[t] -= 1
            try:
                sh = shanten_mod.shanten(counts, 0, memo=memo)
            except Exception:  # noqa: BLE001
                sh = None
            counts[t] += 1
            if sh is not None and (best is None or sh < best):
                best = sh
        return best
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="发牌质量对照")
    ap.add_argument("--rooms", type=int, default=0)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    memo: dict = {}
    data = {"our": {"ish": [], "discards": []}, "opp": {"ish": [], "discards": []}}
    round_draws: list[int] = []
    rooms = 0
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4:
            continue
        rooms += 1
        try:
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            # 初始向听（发牌后、任何事件之前）
            ish: list[int | None] = []
            for s in range(4):
                ish.append(initial_shanten(list(state.seats[s].hand), memo))
            disc = [0, 0, 0, 0]
            for ev in events:
                if ev.get("type") == DISCARDED and isinstance(ev.get("seat"), int):
                    s = ev["seat"]
                    if 0 <= s < 4:
                        disc[s] += 1
            round_draws.append(sum(disc))
            for s in range(4):
                grp = "our" if ids[s] == OUR else "opp"
                if ish[s] is not None:
                    data[grp]["ish"].append(ish[s])
                data[grp]["discards"].append(disc[s])

    print(f"房={rooms}")
    for grp, name in (("our", "我方"), ("opp", "对手")):
        v = data[grp]["ish"]
        d = data[grp]["discards"]
        if not v:
            continue
        print(
            f"\n[{name}] n={len(v)} 初始向听 均={st.mean(v):.3f} 中位={st.median(v):.1f} "
            f"| 分布={{" + ", ".join(f"{k}:{v.count(k)}" for k in sorted(set(v))) + "}"
        )
        print(f"       出牌次数/局 均={st.mean(d):.3f} 中位={st.median(d):.1f}")
    for key, lab in (("ish", "初始向听"), ("discards", "出牌次数")):
        t, p = welch(data["our"][key], data["opp"][key])
        print(f"\n{lab}: 我方均={st.mean(data['our'][key]):.3f} 对手均={st.mean(data['opp'][key]):.3f} "
              f"差={st.mean(data['our'][key]) - st.mean(data['opp'][key]):+.3f} (t={t:+.2f}, p={p:.4f})")
    print(f"\n本局总出牌数 均={st.mean(round_draws):.2f}（作为轮的节奏参照）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
