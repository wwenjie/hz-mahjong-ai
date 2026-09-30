#!/usr/bin/env python
"""C16 我方**每局得分分解**：那 −1.09 分/局到底丢在哪？（只读真机事件流，零平台请求）

**动机**：C10 测出我方单局均分 **−1.09**、对手 **+0.36**，胡局占比 **21.1%**（真机首名率
18.2%）。「在输分」是根本症结，但**丢在哪一层**没量化过。本仪器把每局净分解成可归因的分项：

1. **胡局率**（我们 vs 同场对手）——是不是「少胡」；
2. **赢时的番/分大小**——是不是「胡了但小」；
3. **流局率与流局时的分**——是不是「该流不流/该冲不冲」；
4. **庄闲效应**——我在庄/在闲时各丢多少；
5. **座位效应**——是否某个座位系统性差（四座位旋转的对照组）；
6. **被自摸的暴露**——我们放给对手的自摸次数（本平台唯一失分来源是别人自摸）。

数据源：每场事件流 `rounds[]`（`dealer`/`is_draw`/`multiplier`/`scores`/`winner`）。
`winner` 是胡牌座位（流局时无意义）；`scores` 是四家净分（零和）。

用法：.venv/bin/python agent/verify/round_score_decomposition.py [--rooms N]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import statistics

OUR = "u_a7f7c67bb14a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(__file__).resolve().parents[2] / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.rooms:
        files = files[:: max(1, len(files) // args.rooms)][: args.rooms]

    n_rounds = 0
    our_hu = opp_hu = draws = 0
    our_scores: list[int] = []
    opp_scores: list[int] = []
    our_win_scores: list[int] = []      # 我们胡那局，我们拿的净分
    opp_win_scores: list[int] = []      # 对手胡那局，赢家净分
    draw_scores_ours: list[int] = []    # 流局时我们净分
    dealer_ours: list[int] = []         # 我们坐庄时的净分
    nondealer_ours: list[int] = []      # 我们坐闲时的净分
    seat_scores: dict[int, list[int]] = collections.defaultdict(list)  # 座位号->我方净分
    # 我们放给对手的自摸：我们非赢家的局里，赢家非我们
    our_pay_when_opp_wins: list[int] = []

    for path in files:
        try:
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            continue
        me = ids.index(OUR)
        for rd in doc.get("rounds") or []:
            sc = [int(x) for x in (rd.get("scores") or [])]
            if len(sc) != 4:
                continue
            n_rounds += 1
            our = sc[me]
            our_scores.append(our)
            for j in range(4):
                if j != me:
                    opp_scores.append(sc[j])
            seat_scores[me].append(our)  # 座位号其实随房变化，这里只留我方
            is_draw = bool(rd.get("is_draw"))
            winner = rd.get("winner")
            if is_draw:
                draws += 1
                draw_scores_ours.append(our)
            elif winner == me:
                our_hu += 1
                our_win_scores.append(our)
            else:
                opp_hu += 1
                our_pay_when_opp_wins.append(our)
                if isinstance(winner, int) and 0 <= winner < 4:
                    opp_win_scores.append(sc[winner])
            if rd.get("dealer") == me:
                dealer_ours.append(our)
            else:
                nondealer_ours.append(our)

    def m(xs):
        return statistics.fmean(xs) if xs else float("nan")

    print(f"扫描 {len(files)} 文件（我方在场）· 局数 {n_rounds}\n")
    print("① 胡局结构")
    print(f"   我们胡局 = {our_hu:6d}  ({our_hu / n_rounds:.1%})")
    print(f"   对手胡局 = {opp_hu:6d}  ({opp_hu / n_rounds:.1%})  ⇒ 三家合计 {(opp_hu / n_rounds) / 3:.1%}/家")
    print(f"   流局     = {draws:6d}  ({draws / n_rounds:.1%})")

    print("\n② 每局净分（零和校验）")
    print(f"   我方均分 = {m(our_scores):+.3f}   对手均分 = {m(opp_scores):+.3f}")
    print(f"   核验零和：{m(our_scores) + 3 * m(opp_scores):+.4f} ≈ 0")

    print("\n③ 赢时的「大小」")
    print(f"   我们胡时净分均值 = {m(our_win_scores):+.3f}（中位 {statistics.median(our_win_scores) if our_win_scores else float('nan'):.0f}）")
    print(f"   对手胡时赢家净分均值 = {m(opp_win_scores):+.3f}（中位 {statistics.median(opp_win_scores) if opp_win_scores else float('nan'):.0f}）")
    print(f"   我们非赢家局（被自摸）净分均值 = {m(our_pay_when_opp_wins):+.3f}")

    print("\n④ 庄闲效应（我方）")
    print(f"   我们坐庄 n={len(dealer_ours):6d} 均分 {m(dealer_ours):+.3f}")
    print(f"   我们坐闲 n={len(nondealer_ours):6d} 均分 {m(nondealer_ours):+.3f}")
    print(f"   流局时我们均分 = {m(draw_scores_ours):+.3f}（n={len(draw_scores_ours)}）")

    print("\n⑤ 归因小结（每 8 局一场的量级）")
    exp_per_round = m(our_scores)
    hu_gap = (opp_hu / 3 / n_rounds) - (our_hu / n_rounds)
    print(f"   我方 {exp_per_round:+.3f}/局 ⇒ 每场约 {exp_per_round * 8:+.2f} 分")
    print(f"   胡局率差（对面单家 − 我们）= {hu_gap:+.1%}；赢时大小差 = {m(our_win_scores) - m(opp_win_scores):+.2f} 分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
