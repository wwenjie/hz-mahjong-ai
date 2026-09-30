"""C29：出牌决策 vs oracle——我们「少胡」是决策次优，还是打得对也输？

只读、零平台请求。对每个座位-局、每一次**简单出牌**（仅一张牌从手牌减少）：
- 重建出牌前手牌（含刚摸到的牌）；
- 对着每个合法候选（手中有该牌）算出牌后向听，取 min 为 oracle 最优；
- 记实际出牌后的向听 − oracle 最优 = **向听损失 gap**，以及实际是否属于 argmin 集合。

口径：
- 只在 `sum(hand) == HAND_SIZE + 1 − 3×副露数` 时评估（即「摸牌后出牌」的整点）；
  吃/碰/杠后的补切不满足该式，自动跳过。
- 财神按 `shanten` 内部规则处理（已含百搭）。
- 不做任何对手手牌推断（合规）。

用法：``.venv/bin/python agent/verify/discard_oracle_probe.py [--rooms N]``
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


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return ((k2 / n2) - (k1 / n1)) / se if se else float("nan")


def eval_discard(hand: list[int], meld_n: int, tile: int, memo: dict) -> tuple[int, int] | None:
    """返回 (实际出牌后向听, oracle 最优向听)；无法评估则 None。"""
    if hand[tile] <= 0:
        return None
    # 实际
    hand[tile] -= 1
    try:
        actual = shanten_mod.shanten(hand, meld_n, memo=memo)
    except Exception:  # noqa: BLE001
        hand[tile] += 1
        return None
    hand[tile] += 1
    # oracle：枚举所有有牌的候选
    best = None
    for t, c in enumerate(hand):
        if c <= 0:
            continue
        hand[t] -= 1
        try:
            sh = shanten_mod.shanten(hand, meld_n, memo=memo)
        except Exception:  # noqa: BLE001
            sh = None
        hand[t] += 1
        if sh is not None and (best is None or sh < best):
            best = sh
    if best is None:
        return None
    return actual, best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="出牌 vs oracle")
    ap.add_argument("--rooms", type=int, default=0)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    memo: dict = {}
    n_eval = {"our": 0, "opp": 0}
    loss_sum = {"our": 0, "opp": 0}
    loss_max = {"our": 0, "opp": 0}
    optimal = {"our": 0, "opp": 0}   # gap==0 的次数
    gaps = {"our": [], "opp": []}     # 逐次 gap（用于检验）
    # 首次 gap>0 之前的连续最优手数（描述稳定性，可选）
    rooms_seen = 0
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4:
            continue
        rooms_seen += 1
        try:
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            for ev in events:
                if ev.get("type") != DISCARDED:
                    replay.apply_event(state, ev)
                    continue
                seat = ev.get("seat")
                if not isinstance(seat, int) or not (0 <= seat < 4):
                    replay.apply_event(state, ev)
                    continue
                before = list(state.seats[seat].hand)
                replay.apply_event(state, ev)   # 施加后手牌少一张
                after = state.seats[seat].hand
                diff = [t for t in range(tiles.TILE_KINDS) if before[t] - after[t] == 1]
                if len(diff) != 1:
                    continue  # 非简单出牌（理论上不该发生）
                meld_n = len(state.seats[seat].melds)
                expected = tiles.HAND_SIZE + 1 - tiles.MELD_SLOTS * meld_n
                if sum(before) != expected:
                    continue
                res = eval_discard(before, meld_n, diff[0], memo)
                if res is None:
                    continue
                actual, best = res
                grp = "our" if ids[seat] == OUR else "opp"
                n_eval[grp] += 1
                loss_sum[grp] += actual - best
                loss_max[grp] = max(loss_max[grp], actual - best)
                if actual - best == 0:
                    optimal[grp] += 1
                gaps[grp].append(actual - best)

    print(f"房={rooms_seen}")
    for grp, name in (("our", "我方"), ("opp", "对手")):
        n = n_eval[grp]
        if not n:
            continue
        print(
            f"\n[{name}] 可评估出牌={n}"
            f" | 达到 oracle 最优={optimal[grp]} ({optimal[grp] / n:.1%})"
            f" | 平均向听损失={loss_sum[grp] / n:.4f}"
            f" | 最大损失={loss_max[grp]}"
        )
    z = two_prop_z(optimal["our"], n_eval["our"], optimal["opp"], n_eval["opp"])
    print(f"\n「达到最优」率差 z = {z:+.2f}（对手 − 我方）")
    t, p = welch(gaps["our"], gaps["opp"])
    print(f"平均向听损失差 t = {t:+.2f} (p={p:.4f})（我方 − 对手）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
