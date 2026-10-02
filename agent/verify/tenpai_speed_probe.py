"""C27：听牌速度探针——考验 C26 假设①「摸到 1 张财神时我们到听更慢」。

只读、零平台请求。从事件流重建每个座位的暗手，在每个「已出牌」的整点算向听数，
取「首次听牌发生在自己的第几摸」（按座位自己的出牌次数计）。

用法：``.venv/bin/python agent/verify/tenpai_speed_probe.py [--rooms N]``
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
GOD = tiles.GOD
DRAWN = "tile_drawn"
DISCARDED = "tile_discarded"


def _phat(p: float) -> float:
    return 0.5 * (1.0 + erf(p / sqrt(2.0)))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p2 - p1) / se if se else float("nan")


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = sqrt(va / len(a) + vb / len(b))
    if not se:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    return t, 2 * (1 - _phat(abs(t)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="听牌速度探针")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--arm-map", default=None,
                    help="game_id→臂映射 JSON（build_arm_map.py 产物）；给了就按臂过滤")
    ap.add_argument("--arm", default=None, help="只统计指定臂（如 v5/v6）；与 --arm-map 联用")
    args = ap.parse_args(argv)

    arm_of = {}
    if args.arm_map:
        with open(args.arm_map, encoding="utf-8") as f:
            arm_of = json.load(f).get("map", {})

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    if arm_of:
        keep = []
        for p in files:
            gid = Path(p).stem
            arm = arm_of.get(gid)
            if arm is None or arm == "mixed":
                continue
            if args.arm and arm != args.arm:
                continue
            keep.append(p)
        files = keep
    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    memo = shanten_mod.Memo() if hasattr(shanten_mod, "Memo") else {}

    # 桶：(did_draw_god 0/1) × 组(our/opp) → 首次听牌摸序列表 + 未听牌数 + 局数
    buckets: dict[tuple[int, str], dict[str, list]] = {
        (g, grp): {"tenpai_at": [], "never": 0, "n": 0}
        for g in (0, 1)
        for grp in ("our", "opp")
    }
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
            # 注意：iter_rounds 产出的 state **已含发牌**（庄 14 张、其余 13 张），
            # 必须就地 apply_event，绝不可自建空 state（会丢掉整副发牌）。
            gods = [0, 0, 0, 0]
            first = [None, None, None, None]
            turn_no = [0, 0, 0, 0]
            for ev in events:
                kind = ev.get("type")
                seat = ev.get("seat")
                if kind == DRAWN and isinstance(seat, int) and 0 <= seat < 4:
                    if ev.get("tile") == "白" or _is_god_code(ev.get("tile")):
                        gods[seat] += 1
                replay.apply_event(state, ev)
                if kind == DISCARDED and isinstance(seat, int) and 0 <= seat < 4:
                    turn_no[seat] += 1
                    if first[seat] is not None:
                        continue
                    counts = state.seats[seat].hand
                    meld_n = len(state.seats[seat].melds)
                    expected = tiles.HAND_SIZE - tiles.MELD_SLOTS * meld_n
                    if sum(counts) != expected:
                        continue
                    try:
                        sh = shanten_mod.shanten(counts, meld_n, memo=memo)
                    except Exception:  # noqa: BLE001
                        continue
                    if sh <= 0:
                        first[seat] = turn_no[seat]
            for s in range(4):
                g = 1 if gods[s] >= 1 else 0
                grp = "our" if ids[s] == OUR else "opp"
                b = buckets[(g, grp)]
                b["n"] += 1
                if first[s] is None:
                    b["never"] += 1
                else:
                    b["tenpai_at"].append(first[s])

    print(f"房={rooms}")
    for g, lab in ((0, "无财神"), (1, "有财神(≥1)")):
        print(f"\n=== {lab} ===")
        for grp, name in (("our", "我方"), ("opp", "对手")):
            b = buckets[(g, grp)]
            n = b["n"]
            if not n:
                continue
            ever = len(b["tenpai_at"])
            mean_at = st.mean(b["tenpai_at"]) if b["tenpai_at"] else float("nan")
            med = st.median(b["tenpai_at"]) if b["tenpai_at"] else float("nan")
            print(
                f"  [{name}] 座位-局={n} | 曾听牌={ever} ({ever / n:.1%}) "
                f"| 首次听牌 均={mean_at:.2f} 中位={med:.1f}（自己第几摸）"
            )
        # 显著性：曾听牌率 + 首次听牌摸序
        a, o = buckets[(g, "our")], buckets[(g, "opp")]
        z = two_prop_z(len(a["tenpai_at"]), a["n"], len(o["tenpai_at"]), o["n"])
        print(f"  曾听牌率差 z = {z:+.2f}（对手 − 我方）")
        if a["tenpai_at"] and o["tenpai_at"]:
            t, pv = welch(a["tenpai_at"], o["tenpai_at"])
            print(f"  首次听牌摸序差 t = {t:+.2f} (p={pv:.4f})（对手 − 我方；正=我们更晚）")
    return 0


def _is_god_code(raw: object) -> bool:
    """事件流的 tile 若是数字码，判断是否财神。"""
    if isinstance(raw, int):
        return raw == GOD
    return False


if __name__ == "__main__":
    raise SystemExit(main())
