#!/usr/bin/env python3
"""分歧点 total 分量拆解（A 2026-10-06 15:27②-1 交办，B' 认领）。

问题：在强 bot 分歧点上，「进张更高的牌」在我们的 `total` 里是哪一项输掉的？
（shanten? pair_value? meld_value? route_value? feed_cost? god_penalty?）

方法：复用 C 的 v2 fidelity 通路（R.iter_rounds + apply_event + situation_for）重建局面，
对每个分歧点跑 `_score_discard` 拿全 8 分量，对比 bot 选的候选 vs v5 选的候选。

输出：哪个分量在分歧点上让 bot 的候选输掉（差值分布 + 主导分量统计）。
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "agent/verify")
sys.path.insert(0, "tools")

from majiang.sim import replay as R  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"

STRONG = {"三杯猫","我胡汉三又回来了","白虎-0211","中国人能飞","大杂烩-G119",
          "别炸我庄求你了","白映","菜菜子","康陶应雀","Kimi-K4.1","腾蛇-0638","铳一色14","走马"}

COMPONENTS = ["shanten", "blocks", "pair_value", "meld_value", "route_value", "feed_cost", "god_penalty", "total"]


def build_name_to_uid(files):
    m = {}
    for fpath in files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in m:
                    m[n] = u
        except Exception:
            pass
    return m


def main() -> int:
    # 1. 从数据集拿强 bot 分歧点坐标（room, round_no, bot_seat, bot_tile, v5_tile）
    chunks = sorted(glob.glob("agent/out/stage-a-dataset-chunks/cs200-n2000/chunk-*.json"))
    points = []
    for f in chunks:
        points.extend(json.load(open(f))["points"])
    div = [p for p in points if p["bot_tile"] != p["v5_tile"] and p["bot_name"] in STRONG]
    print(f"vs 强 bot 分歧点：{len(div)}", flush=True)

    # 只取「2 向听 + 早巡」核心子集（远分歧最集中处）
    core = [p for p in div if p["shanten"] == 2 and p["turn_bucket"] == "1-5"]
    print(f"「2 向听 + 早巡」核心子集：{len(core)}", flush=True)

    # 按 room 分组，只处理有核心分歧点的房
    by_room = {}
    for p in core:
        by_room.setdefault(p["room"], []).append(p)
    print(f"涉及房间：{len(by_room)}", flush=True)

    name_to_uid = build_name_to_uid(glob.glob("data/auto_sessions/*/events/*.json")[:3000])
    strong_uids = {name_to_uid[n] for n in STRONG if n in name_to_uid}

    decider = versions.build("v6", Mode.QUALIFIER)

    # 2. 逐房 replay，重建分歧点局面，算全分量
    stats = Counter()
    comp_diffs = {c: [] for c in COMPONENTS}  # bot - v5（各分量差）
    dominant = Counter()  # 哪个分量是「让 bot 候选输掉」的主导项
    n_done = 0

    for room, room_points in sorted(by_room.items()):
        room_dir = room.split("_r")[0]
        ev_path = f"data/auto_sessions/{room_dir}/events/{room}.json"
        if not Path(ev_path).exists():
            stats["no_file"] += len(room_points)
            continue
        try:
            doc = json.loads(Path(ev_path).read_text(encoding="utf-8"))
        except Exception:
            stats["bad_json"] += len(room_points)
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4 or OUR not in ids:
            stats["no_our_seat"] += len(room_points)
            continue

        # 本房的目标分歧点按 round_no 索引
        target_by_round = {}
        for p in room_points:
            target_by_round.setdefault(p["round_no"], []).append(p)

        try:
            rounds = list(R.iter_rounds(doc))
        except Exception:
            stats["iter_rounds"] += len(room_points)
            continue

        for state, events in rounds:
            cur_round = getattr(state, "round_no", None)
            if cur_round not in target_by_round:
                # 仍要 apply 事件推进状态
                for ev in events:
                    try:
                        R.apply_event(state, ev)
                    except Exception:
                        break
                continue
            for ev in events:
                etype = ev.get("type")
                seat = ev.get("seat")
                if etype != "tile_drawn":
                    try:
                        R.apply_event(state, ev)
                    except Exception:
                        break
                    continue
                tile = R._tile_of(ev.get("tile"))  # noqa: SLF001
                try:
                    R.apply_event(state, ev)
                except Exception:
                    break
                # 这个 tile_drawn 之后，该 seat 面临出牌决策——检查是否命中目标分歧点
                pending = target_by_round.get(cur_round, [])
                if not pending:
                    continue
                for p in list(pending):
                    if p["bot_seat"] != seat:
                        continue
                    # 重建 situation
                    try:
                        sit = state.situation_for(seat, phase=PHASE_DRAW, drawn=tile)
                    except Exception:
                        stats["situation"] += 1
                        continue
                    # 算所有候选的全分量
                    from majiang.rules.action import DISCARD, legal_actions
                    cands = [a for a in legal_actions(sit) if a.kind == DISCARD]
                    scores = {}
                    for a in cands:
                        ds = decider._score_discard(sit, a)
                        scores[ds.tile] = ds
                    bot_tile = p["bot_tile"]
                    v5_tile = p["v5_tile"]
                    if bot_tile not in scores or v5_tile not in scores:
                        stats["tile_missing"] += 1
                        continue
                    bs, vs = scores[bot_tile], scores[v5_tile]
                    # ★ 校验对点正确性：重算的 v5 候选 main_total 应与数据集存盘值一致（容差 6e-5，存储舍入）
                    stored_v5 = [c for c in p["candidates"] if c.get("is_v5")]
                    if stored_v5:
                        stored_mt = stored_v5[0].get("main_total")
                        if stored_mt is not None and abs(float(stored_mt) - float(vs.total)) > 6e-5:
                            stats["point_mismatch"] += 1
                            continue
                    # 分量差（bot - v5）。注意方向：
                    #   shanten 越小越好 ⇒ diff>0 = bot 更差
                    #   其他分量越大越好 ⇒ diff<0 = bot 更差
                    diffs = {}
                    for c in COMPONENTS:
                        bv = getattr(bs, c)
                        vv = getattr(vs, c)
                        diffs[c] = float(bv) - float(vv)
                        comp_diffs[c].append(diffs[c])
                    # 主导分量：让 bot 候选「变差」最多的分量。
                    # 统一换算成「bot 的损失」：shanten 取 +diff（bot 向听更高=损失），其余取 -diff（bot 值更低=损失）
                    loss = {c: (diffs[c] if c == "shanten" else -diffs[c]) for c in COMPONENTS if c != "total"}
                    # 只统计真正有损失的（排除平局 loss=0）
                    pos_loss = {c: v for c, v in loss.items() if v > 1e-9}
                    if pos_loss:
                        dom = max(pos_loss, key=lambda c: pos_loss[c])
                        dominant[dom] += 1
                    else:
                        dominant["(平局/无损失)"] += 1
                    stats["done"] += 1
                    n_done += 1
                    pending.remove(p)
        if n_done >= 400:  # 采样上限
            break

    print(f"\n成功拆解 {stats['done']} 个分歧点", flush=True)
    print(f"失败明细：{dict(stats)}", flush=True)

    print("\n=== 分量差（bot − v5）的均值 / 中位（注意：shanten 越小越好，diff>0=bot 更差；其余越大越好，diff<0=bot 更差）===")
    for c in COMPONENTS:
        xs = sorted(comp_diffs[c])
        if not xs:
            continue
        mean = sum(xs) / len(xs)
        med = xs[len(xs) // 2]
        if c == "shanten":
            worse = sum(1 for x in xs if x > 0) / len(xs) * 100
        else:
            worse = sum(1 for x in xs if x < 0) / len(xs) * 100
        print(f"  {c:<14} mean {mean:+8.3f}  med {med:+8.3f}  bot 更差占比 {worse:.1f}%")

    print("\n=== 主导分量（让 bot 候选输掉的最负分量） ===")
    for c, n in dominant.most_common():
        print(f"  {c:<14} {n} ({n/max(stats['done'],1)*100:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
