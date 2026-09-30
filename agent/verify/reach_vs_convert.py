#!/usr/bin/env python
"""C18 胜率缺口分解：**「没到听」还是「到了听没胡」？**（只读真机事件流，零平台请求）

**动机**：C16 测出病根是**少胡**（胡率 21.1% vs 对手 26.1%，补胜率可回补 ≈+1.12 分/局）。
再往下拆一步：胡率 = P(到听) × P(到听后胡 | 到听)。C 侧独立实现，**不 import A 的
`analyze_hand_progress.py` / `analyze_tenpai_timing.py`**（口径独立才有复核价值）。

**口径**：
- 对每个「该座位摸牌后」的局面，用**精确** `shanten.shanten(counts, meld_count)` 算向听；
- 「到听」= 该局该座位**至少有一次**向听 == 0；
- 「到听后胡」= 该局该座位最终是 `winner`（且到听过）；
- 分**出牌序号 n** 分层（同机会比较：被早胡掉的短局四家都摸得少，按局算会失真——
  这是 A 的 `analyze_tenpai_timing.py` docstring 里记过的坑，此处沿用同一修正）。

输出：我们 vs 对手（三家合并）的
① 到听率；② 到听后胡率；③ 分段（早/中/晚）到听率；④ 两个因子的乘积分解。

用法：`.venv/bin/python agent/verify/reach_vs_convert.py [--rooms N]`
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.rooms:
        files = files[:: max(1, len(files) // args.rooms)][: args.rooms]

    # 按座位聚合：到听局数 / 总局数；到听后胡局数 / 到听局数
    stat = {"ours": [0, 0, 0, 0], "theirs": [0, 0, 0, 0]}  # [总局, 到听局, 到听后胡局, 有向听样本]
    tenpai_by_n = {"ours": collections.defaultdict(lambda: [0, 0]), "theirs": collections.defaultdict(lambda: [0, 0])}
    n_rounds = 0

    for path in files:
        try:
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        me = ids.index(OUR)
        # `rounds[]`（文档级）带 `winner` / `is_draw`；而 `round_ended` 事件会覆盖
        # `state.result` 成只带 `draw` 的 data 字典（无 winner）⇒ 必须用文档级这份。
        round_meta = {int(r.get("round_no", 0) or 0): r for r in (doc.get("rounds") or [])}
        try:
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for state, events in rounds:
            n_rounds += 1
            reached = [False] * 4
            draw_idx = [0] * 4
            seen_tenpai_n: list[set[int]] = [set() for _ in range(4)]
            for ev in events:
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                if ev.get("type") != replay.DRAWN:
                    continue
                seat = ev.get("seat")
                if not isinstance(seat, int) or not 0 <= seat < 4:
                    continue
                rs = state.seats[seat]
                total = sum(rs.hand)
                if total <= 0:
                    continue
                try:
                    # 摸牌后是「多一张」状态（14−3×副露），必须用 shanten_any；
                    # 直接用 shanten 会因张数不符抛 ShantenError（这正是它 docstring 里
                    # 记过的坑——会把已听牌的牌算错）。
                    sh = shanten_mod.shanten_any(rs.hand, len(rs.melds))
                except Exception:  # noqa: BLE001
                    continue
                draw_idx[seat] += 1
                n = draw_idx[seat]
                grp = "ours" if seat == me else "theirs"
                tenpai_by_n[grp][n][0] += 1
                if sh <= 0:
                    reached[seat] = True
                    seen_tenpai_n[seat].add(n)
                    tenpai_by_n[grp][n][1] += 1
            result = round_meta.get(int(state.round_no), {}) or {}
            winner = result.get("winner")
            is_draw = bool(result.get("is_draw"))
            for seat in range(4):
                grp = "ours" if seat == me else "theirs"
                stat[grp][0] += 1
                if reached[seat]:
                    stat[grp][1] += 1
                    if (not is_draw) and winner == seat:
                        stat[grp][2] += 1

    def rate(a: int, b: int) -> str:
        return f"{a / b:.1%}" if b else "n/a"

    print(f"扫描 {len(files)} 文件 · 局数 {n_rounds}\n")
    print("① 到听率 与 到听后胡率（按座位聚合）")
    print(f"   {'组':>6} {'总座位-局':>10} {'到听':>10} {'到听率':>8} {'到听后胡局':>10} {'到听后胡率':>10}")
    for grp, label in (("ours", "我们"), ("theirs", "对手")):
        tot, tp, win = stat[grp][0], stat[grp][1], stat[grp][2]
        print(f"   {label:>6} {tot:>10} {tp:>10} {rate(tp, tot):>8} {win:>10} {rate(win, tp):>10}")

    print("\n② 分段到听率（按该座位第 n 次摸牌；分母=该 n 的摸牌样本）")
    print(f"   {'n':>4} {'我们到听率':>10} {'对手到听率':>10}  样本(我们/对手)")
    for n in sorted(set(tenpai_by_n["ours"]) | set(tenpai_by_n["theirs"])):
        o = tenpai_by_n["ours"].get(n)
        t = tenpai_by_n["theirs"].get(n)
        if not o or o[0] < 30:
            continue
        ot = rate(o[1], o[0])
        tt = rate(t[1], t[0]) if t and t[0] else "n/a"
        print(f"   {n:>4} {ot:>10} {tt:>10}  {o[0]}/{t[0] if t else 0}")

    print("\n③ 分解：胡率 = 到听率 × 到听后胡率")
    for grp, label in (("ours", "我们"), ("theirs", "对手")):
        tot, tp, win = stat[grp][0], stat[grp][1], stat[grp][2]
        p_reach = tp / tot if tot else 0
        p_conv = win / tp if tp else 0
        print(f"   {label}: 到听 {p_reach:.1%} × 转换 {p_conv:.1%} = 胡率 {p_reach * p_conv:.1%}")
    o_r = stat["ours"][1] / stat["ours"][0] if stat["ours"][0] else 0
    t_r = stat["theirs"][1] / stat["theirs"][0] if stat["theirs"][0] else 0
    o_c = stat["ours"][2] / stat["ours"][1] if stat["ours"][1] else 0
    t_c = stat["theirs"][2] / stat["theirs"][1] if stat["theirs"][1] else 0
    print(f"\n   到听率缺口（对手−我们）= {t_r - o_r:+.1%}")
    print(f"   转换率缺口（对手−我们）= {t_c - o_c:+.1%}")
    print("   ⇒ 缺口主要落在哪一项，就是下一档该改的方向。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
