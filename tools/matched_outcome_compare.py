#!/usr/bin/env python3
"""匹配后果对拍（推广版）——A 01:43 交办，B' 01:50 认领。

问题：在匹配过的局面上（同巡目 × 同向听 × 同财神数 × 同并列），
「跟 bot 走」vs「跟我们自己的键走」两条分支的**结局**差多少？

数据：
- 决策点：C 的 stage-a 底座（agent/out/stage-a-dataset-chunks/cs200-n2000/chunk-*.json）
  字段：room(=game_id) / round_no / turn_bucket / god_n / shanten / bot_tile / v5_tile / agree / candidates
- 结局：原始事件文件 data/auto_sessions/{room_id}/events/{game_id}.json
  里 blocks[].events 的 round_ended 的 scores（按 seat）

方法（观测数据，非反事实——口径写清）：
- 分歧点：bot_tile != v5_tile 的决策点（n ~ 18.6%）
- 对每个分歧点，定位该局该座（bot_seat）的结局分（round_ended.scores[bot_seat]）
- 匹配：按 (turn_bucket, shanten, god_n, 是否并列) 分桶
- ★ 口径核心：同一分歧点只有一个实际结局（我们当时实际选了 v5 的方向或 bot 的方向之一——
  实际上是 v5 在打，所以实际结局 = 「跟我们键走」分支）。
  「跟 bot 走」分支的结局 = 用**匹配桶内「v5 恰好选了 bot 同向」的点**（即 agree=1 的点）
  的结局作代理对照组。
  ⇒ 比较：桶内「分歧点（agree=0）的结局均值」 vs 「一致点（agree=1）的结局均值」
  这就是「跟 bot 走 vs 跟我们键走」在匹配口径下的结局差。
  局限：agree=1 的点本身有选择偏差（v5 和 bot 一致的点可能本来就简单）——如实标注。

预登记判据（A 01:43③）：
- 跟 bot 走结局显著更好（胡率 +≥3pp 或番值 +≥10%）⇒ 批 pairwise ranking
- 差距 <1pp ⇒ SL 路线整体暂停

用法：
    uv run python tools/matched_outcome_compare.py [--out agent/out/matched-outcome.txt]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

DATA = "agent/out/stage-a-dataset-chunks/cs200-n2000"
EVENTS = "data/auto_sessions"


def game_id_to_event_path(game_id: str) -> str | None:
    """a_00acc6dcfa5c_r1_b1_t0 → data/auto_sessions/a_00acc6dcfa5c/events/a_00acc6dcfa5c_r1_b1_t0.json"""
    parts = game_id.split("_r")
    if len(parts) != 2:
        return None
    room_id = parts[0]
    p = f"{EVENTS}/{room_id}/events/{game_id}.json"
    return p if os.path.exists(p) else None


def load_outcome_cache() -> dict:
    """读所有相关事件文件，抽取 {(game_id, round_no): {seat: score}} 及胡牌信息。"""
    return {}


def extract_round_outcomes(event_doc: dict) -> dict:
    """从事件文档抽取 {round_no: {"scores": [...], "winner": int, "fan": ...}}。

    blocks[].events 里找 type=round_ended。
    """
    out = {}
    for blk in event_doc.get("blocks", []):
        for ev in blk.get("events", []):
            if ev.get("type") == "round_ended":
                data = ev.get("data") or {}
                rn = data.get("round_no")
                if rn is None:
                    # 有些格式可能没 round_no，用计数器——先记 None 跳过
                    continue
                out[rn] = {
                    "scores": data.get("scores"),
                    "winner": ev.get("seat"),
                    "dealer": data.get("dealer"),
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="agent/out/matched-outcome.txt")
    args = ap.parse_args()

    # 1. 加载决策点
    files = sorted(glob.glob(f"{args.data}/chunk-*.json"))
    if not files:
        print(f"no chunk under {args.data}", flush=True)
        return 1
    points = []
    for f in files:
        d = json.load(open(f))
        points.extend(d.get("points", []))
    print(f"决策点 {len(points)}", flush=True)

    # 2. 定位需要的事件文件（按 game_id 去重）
    game_ids = sorted({p["room"] for p in points})
    print(f"涉及 game_id {len(game_ids)}", flush=True)

    # 3. 逐个事件文件抽取结局（round_ended.scores）
    #    game_id 里含 b{round_no}？看样本 a_..._r1_b1_t0——b1 可能就是 round。
    #    但 point.round_no 单独有。先试：一个 game_id 文件里通常有多少 round_ended
    outcomes = {}  # (game_id, round_no) -> {"scores": [...], "winner": seat}
    n_files = 0
    n_events = 0
    missing = 0
    for gid in game_ids:
        path = game_id_to_event_path(gid)
        if not path:
            missing += 1
            continue
        try:
            doc = json.load(open(path))
        except Exception:
            missing += 1
            continue
        n_files += 1
        ro = extract_round_outcomes(doc)
        n_events += len(ro)
        for rn, info in ro.items():
            outcomes[(gid, rn)] = info
    print(f"事件文件 {n_files}（缺失 {missing}），抽出 round_ended {n_events}", flush=True)

    # 4. 探针口径检查：point.round_no 是否能对上 outcomes
    n_join = 0
    n_no_join = 0
    for p in points[:2000]:
        key = (p["room"], p["round_no"])
        if key in outcomes:
            n_join += 1
        else:
            n_no_join += 1
    print(f"join 检查（前 2000 点）：对上 {n_join} / 对不上 {n_no_join}", flush=True)

    if n_join == 0:
        # round_no 对不上——试 b{batch} 口径
        print("round_no 对不上，检查 game_id 后缀与 round_no 的关系", flush=True)
        # 打印样本帮助调试
        for p in points[:5]:
            print(f"  room={p['room']} round_no={p['round_no']}", flush=True)
        for k in list(outcomes.keys())[:5]:
            print(f"  outcome key: {k}", flush=True)

    # 5. 聚合：匹配桶 (turn_bucket, shanten, god_n, is_tie)
    #    桶内比较 agree=0（分歧，实际走了 v5）vs agree=1（一致，走了 bot 方向）的结局
    buckets = defaultdict(lambda: {"div": [], "agr": []})
    for p in points:
        key = (p["room"], p["round_no"])
        oc = outcomes.get(key)
        if not oc or not oc.get("scores"):
            continue
        seat = p.get("bot_seat")
        if seat is None or seat >= len(oc["scores"]):
            continue
        my_score = oc["scores"][seat]
        won = (oc.get("winner") == seat)
        # 匹配键
        cands = p.get("candidates") or []
        mt = [c["main_total"] for c in cands] if cands else [0]
        mx = max(mt)
        is_tie = sum(1 for v in mt if abs(v - mx) < 1e-9) > 1
        bkey = (p.get("turn_bucket"), p.get("shanten"), p.get("god_n"), is_tie)
        rec = {"score": my_score, "won": won}
        if p.get("agree"):
            buckets[bkey]["agr"].append(rec)
        else:
            buckets[bkey]["div"].append(rec)

    # 6. 报告
    lines = [
        "=" * 72,
        "匹配后果对拍（推广版）——A 01:43③ 预登记判据",
        "口径：桶内「分歧点（实际走了 v5 键）」vs「一致点（走了 bot 方向）」的结局差",
        "⚠ 局限：观测数据非反事实，一致点有选择偏差（v5 和 bot 一致的点可能更简单）",
        "=" * 72,
    ]
    tot_div_sc = tot_agr_sc = 0.0
    tot_div_n = tot_agr_n = 0
    tot_div_win = tot_agr_win = 0
    rows = []
    for bkey in sorted(buckets, key=str):
        b = buckets[bkey]
        if len(b["div"]) < 30 or len(b["agr"]) < 30:
            continue  # 样本太少不报
        div_sc = sum(r["score"] for r in b["div"]) / len(b["div"])
        agr_sc = sum(r["score"] for r in b["agr"]) / len(b["agr"])
        div_win = sum(r["won"] for r in b["div"]) / len(b["div"])
        agr_win = sum(r["won"] for r in b["agr"]) / len(b["agr"])
        rows.append((bkey, len(b["div"]), len(b["agr"]), div_sc, agr_sc, div_win, agr_win))
        tot_div_sc += sum(r["score"] for r in b["div"])
        tot_agr_sc += sum(r["score"] for r in b["agr"])
        tot_div_n += len(b["div"])
        tot_agr_n += len(b["agr"])
        tot_div_win += sum(r["won"] for r in b["div"])
        tot_agr_win += sum(r["won"] for r in b["agr"])

    lines.append(f"\n{'桶(巡目,向听,财神,并列)':<28}{'n分歧':>7}{'n一致':>7}{'分/分歧':>9}{'分/一致':>9}{'胡率分':>8}{'胡率一':>8}")
    for r in rows:
        bkey, dn, an, dsc, asc, dw, aw = r
        lines.append(f"{str(bkey):<28}{dn:>7}{an:>7}{dsc:>9.2f}{asc:>9.2f}{dw:>8.3f}{aw:>8.3f}")

    if tot_div_n and tot_agr_n:
        g_dsc = tot_div_sc / tot_div_n
        g_asc = tot_agr_sc / tot_agr_n
        g_dw = tot_div_win / tot_div_n
        g_aw = tot_agr_win / tot_agr_n
        lines.append(f"\n总合（n分歧={tot_div_n}, n一致={tot_agr_n}）：")
        lines.append(f"  每手分：分歧 {g_dsc:+.2f} vs 一致 {g_asc:+.2f}，差 {g_asc-g_dsc:+.2f}")
        lines.append(f"  胡率：分歧 {g_dw:.3f} vs 一致 {g_aw:.3f}，差 {(g_aw-g_dw)*100:+.1f}pp")
        lines.append(f"\n判据（A 01:43③）：胡率差 {abs(g_aw-g_dw)*100:.1f}pp "
                     f"{'≥3pp ⇒ 更一致=更强成立' if abs(g_aw-g_dw)>=0.03 else '<3pp'}；"
                     f"每手分差 {abs(g_asc-g_dsc):.2f}")
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
