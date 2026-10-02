#!/usr/bin/env python
"""缺口表行：副露率 + 中张弃牌占比（v5/v6 时代分臂，我方 vs 同期对手）。

只读真机事件流，零平台请求。两个口径都简单、放一起省一次全量扫描：

1. **副露率**：每座位每局的副露次数（吃+碰+明杠；暗杠不算副露——不亮给对手）。
   聚合 我方 vs 对手 的「人均副露次数/局」与「有副露局占比」。
   历史参照（跨时代）：我方 ~0.6 vs 对手 ~1.3（A 22:00 提到的结构性差距之一）。

2. **中张弃牌占比**：我方弃牌里中张（2-8 万/条/筒）的比例 vs 对手。
   口径：弃牌事件（tile_discarded）里，花色牌 2..8 算中张，1/9/字牌算边张。

分时代：--arm-map + --arm（照 convert_by_remaining_probe 模式，mixed/未知剔除）。

用法：`.venv/bin/python agent/verify/meld_middle_probe.py [--rooms N] --arm-map agent/out/arm_map.json --arm v5`
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import statistics as st
import sys
from math import erf, sqrt

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
MELD_EV = {"chi", "peng", "minggang"}  # 事件名按事件流实际取值，跑时校验


def _phat(p: float) -> float:
    return 0.5 * (1.0 + erf(p / sqrt(2.0)))


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = sqrt(va / len(a) + vb / len(b))
    if not se:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    return t, 2 * (1 - _phat(abs(t)))


def is_middle(code: str) -> bool | None:
    """中张=花色 2..8；边张=1/9/字。返回 None 表示无法解析。"""
    if not code or code in ("白", "发", "中", "东", "南", "西", "北"):
        return False
    if len(code) == 2 and code[0].isdigit() and code[1] in "wtb":
        return 2 <= int(code[0]) <= 8
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--arm-map", default=None)
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()

    arm_of = {}
    if args.arm_map:
        with open(args.arm_map, encoding="utf-8") as f:
            arm_of = json.load(f).get("map", {})

    files = sorted(glob.glob(str(REPO / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if arm_of:
        keep = []
        for p in files:
            gid = pathlib.Path(p).stem
            arm = arm_of.get(gid)
            if arm is None or arm == "mixed":
                continue
            if args.arm and arm != args.arm:
                continue
            keep.append(p)
        files = keep
    if args.rooms:
        files = files[:: max(1, len(files) // args.rooms)][: args.rooms]

    # 副露率：{grp: [每局副露次数...]}；中张：{grp: [中张, 总数]}
    meld_per_round = {"our": [], "opp": []}
    middle = {"our": [0, 0], "opp": [0, 0]}
    event_kinds = collections.Counter()
    rooms = rounds_seen = 0

    for path in files:
        try:
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        me = ids.index(OUR)
        rooms += 1
        try:
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for _state, events in rounds:
            rounds_seen += 1
            melds = [0, 0, 0, 0]
            for ev in events:
                kind = str(ev.get("type", ""))
                event_kinds[kind] += 1
                seat = ev.get("seat")
                if not isinstance(seat, int) or not 0 <= seat < 4:
                    continue
                kl = kind.lower()
                if kl in MELD_EV or "chi" in kl or "peng" in kl or ("gang" in kl and "an" not in kl):
                    melds[seat] += 1
                if kind == replay.DISCARDED:
                    mid = is_middle(str(ev.get("tile", "")))
                    if mid is not None:
                        grp = "our" if seat == me else "opp"
                        middle[grp][1] += 1
                        if mid:
                            middle[grp][0] += 1
            for s in range(4):
                grp = "our" if s == me else "opp"
                meld_per_round[grp].append(melds[s])

    print(f"房={rooms} 局={rounds_seen}")
    print("\n① 副露率（每局每座位副露次数；含吃/碰/明杠）")
    for grp, name in (("our", "我方"), ("opp", "对手")):
        v = meld_per_round[grp]
        nz = sum(1 for x in v if x > 0)
        print(f"  {name}: n={len(v)} 人均={st.mean(v):.3f}/局 有副露局占比={nz/len(v):.1%}")
    t, p = welch(meld_per_round["our"], meld_per_round["opp"])
    print(f"  差（对手−我方）={st.mean(meld_per_round['opp'])-st.mean(meld_per_round['our']):+.3f} (t={t:+.2f}, p={p:.4f})")

    print("\n② 中张弃牌占比（花色 2-8 / 全部可解析弃牌）")
    for grp, name in (("our", "我方"), ("opp", "对手")):
        k, n = middle[grp]
        if n:
            print(f"  {name}: {k}/{n} = {k/n:.1%}")
    # 二比例 z
    k1, n1 = middle["our"]
    k2, n2 = middle["opp"]
    if n1 and n2:
        pp = (k1 + k2) / (n1 + n2)
        se = sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        z = ((k2 / n2) - (k1 / n1)) / se if se else float("nan")
        print(f"  差（对手−我方）z = {z:+.2f}")

    if rounds_seen and not event_kinds:
        print("⚠ 未捕到任何事件（事件流 schema 变了？）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
