#!/usr/bin/env python3
"""P3 M1：真机对手行为克隆——数据集构建（离线，不打平台）。

从事件流重建**对手座位**每次自由出牌时刻的完整信息集：
  特征（全公开量 + 该座位手牌，与决策者当时可见信息一致）：
    [0:34]   手牌计数
    [34:68]  刚摸到的牌 one-hot
    [68:72]  四座位副露组数
    [72:106] 桌面已见弃牌计数（四家合计）
    [106]    牌墙剩余（136-53-全场摸牌数）
    [107]    该座位第几摸
    [108]    round_no
    [109]    该座位是否庄家
    [110]    该座位手留白板数
    [111]    该座位爆头态（win.is_baotou）
    [112:116] 本场前序局累计分（相对该座位视角的原始 4 维）
  标签：实际打出的牌（0-33）

过滤（不是自由决策，不教模型）：
  - catch_play=true 的出牌（抓打圈强制打刚摸的牌）
  - 紧邻其后的 timeout(kind=discard) 自动出牌（超时强制）
  - 我们自己座位的出牌（只克隆对手）

输出：verify/out/clone_samples.npz（X float32 / y int8 / meta）+ 统计打印。
用法：nice -n 15 uv run python verify/clone_dataset.py [--limit N]
"""
import argparse
import collections
import glob
import json

import numpy as np

from majiang.rules import tiles, win

OUR_UID = "u_a7f7c67bb14a"
N_FEAT = 116


def split_rounds(doc):
    rounds, cur = [], None
    for b in doc["blocks"]:
        rn = b.get("round_no")
        if cur is None or rn != cur["round_no"]:
            cur = {"round_no": rn, "hands": None, "events": []}
            rounds.append(cur)
            sh = b.get("start_hands")
            if sh and sh[0] is not None:
                cur["hands"] = sh
        cur["events"].extend(b.get("events") or [])
    return rounds


def scan_room(path, room_id, out_x, out_y, out_meta, stats):
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("status") != "finished":
        stats["skip_not_finished"] += 1
        return
    seats = doc.get("seats") or []
    our = next((i for i, s in enumerate(seats) if s.get("user_id") == OUR_UID), None)
    if our is None:
        stats["skip_no_us"] += 1
        return

    match_scores = [0, 0, 0, 0]  # 本场前序局累计分
    for rnd in split_rounds(doc):
        if rnd["hands"] is None:
            continue
        hand = [collections.Counter(h) for h in rnd["hands"]]
        melds = [0, 0, 0, 0]
        table_discards = collections.Counter()
        wall_draws = 0
        draw_no = [0, 0, 0, 0]
        last_drawn = [None] * 4
        dealer = rnd.get("dealer")

        events = rnd["events"]
        for idx, e in enumerate(events):
            et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
            if et == "tile_drawn" and s in (0, 1, 2, 3):
                hand[s][tile] += 1
                draw_no[s] += 1
                wall_draws += 1
                last_drawn[s] = tile
            elif et == "tile_discarded" and s in (0, 1, 2, 3):
                catch_play = bool(data.get("catch_play"))
                nxt = events[idx + 1] if idx + 1 < len(events) else {}
                timeout_auto = (nxt.get("type") == "timeout" and nxt.get("seat") == s
                                and (nxt.get("data") or {}).get("kind") == "discard")
                if s == our:
                    stats["skip_our_seat"] += 1
                elif catch_play:
                    stats["skip_catch_play"] += 1
                elif timeout_auto:
                    stats["skip_timeout_auto"] += 1
                else:
                    # 决策时点特征：手牌含刚摸的牌（14 张），桌面不含本次出牌
                    counts = [0] * tiles.TILE_KINDS
                    for code, c in hand[s].items():
                        counts[tiles.parse(code)] = c
                    feat = np.zeros(N_FEAT, dtype=np.float32)
                    feat[0:34] = counts
                    if last_drawn[s] is not None:
                        feat[34 + tiles.parse(last_drawn[s])] = 1.0
                    feat[68:72] = melds
                    for code, c in table_discards.items():
                        feat[72 + tiles.parse(code)] = c
                    feat[106] = 136 - 53 - wall_draws
                    feat[107] = draw_no[s]
                    feat[108] = rnd["round_no"]
                    feat[109] = 1.0 if s == dealer else 0.0
                    feat[110] = counts[tiles.GOD]
                    try:
                        feat[111] = 1.0 if win.is_baotou(counts, melds[s]) else 0.0
                    except Exception:
                        feat[111] = 0.0
                    feat[112:116] = match_scores
                    out_x.append(feat)
                    out_y.append(tiles.parse(tile))
                    out_meta.append((room_id, s, rnd["round_no"], draw_no[s]))
                    stats["samples"] += 1
                hand[s][tile] -= 1
                table_discards[tile] += 1
            elif et == "chi" and s in (0, 1, 2, 3):
                need = collections.Counter(data.get("tiles") or [])
                need[tile] -= 1
                if need[tile] == 0:
                    del need[tile]
                for t, c in need.items():
                    hand[s][t] -= c
                melds[s] += 1
            elif et == "peng" and s in (0, 1, 2, 3):
                hand[s][tile] -= 2
                melds[s] += 1
            elif et == "gang" and s in (0, 1, 2, 3):
                kind = data.get("kind")
                hand[s][tile] -= {"an": 4, "bu": 1, "ming": 3}.get(kind, 0)
                if kind != "bu":
                    melds[s] += 1
            elif et == "round_ended":
                sc = (e.get("data") or {}).get("scores")
                if sc and len(sc) == 4:
                    for i2 in range(4):
                        match_scores[i2] += sc[i2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（调试用）")
    ap.add_argument("--glob", default="data/auto_sessions/*/events/*.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if args.limit:
        files = files[: args.limit]
    out_x, out_y, out_meta = [], [], []
    stats = collections.Counter()
    for i, p in enumerate(files):
        room_id = p.split("/")[-3]
        try:
            scan_room(p, room_id, out_x, out_y, out_meta, stats)
        except Exception as exc:  # noqa: BLE001
            stats["file_errors"] += 1
            if stats["file_errors"] <= 5:
                print(f"文件错误 {p}: {type(exc).__name__}: {exc}")
        if (i + 1) % 200 == 0:
            print(f"  进度 {i + 1}/{len(files)} 样本 {stats['samples']}")

    x = np.stack(out_x) if out_x else np.zeros((0, N_FEAT), dtype=np.float32)
    y = np.array(out_y, dtype=np.int8)
    rooms = sorted({m[0] for m in out_meta})
    room_idx = {r: i for i, r in enumerate(rooms)}
    meta = np.array([(room_idx[m[0]], m[1], m[2], m[3]) for m in out_meta], dtype=np.int32)

    import os
    os.makedirs("verify/out", exist_ok=True)
    np.savez_compressed("verify/out/clone_samples.npz", x=x, y=y, meta=meta,
                        rooms=np.array(rooms))
    print(f"\n样本={len(y)} 特征维={x.shape[1]} 房数={len(rooms)}")
    print("过滤统计:", dict(stats))
    if len(y):
        print("标签熵（基线参考）: 最多牌类占比 %.1f%%" % (100 * np.bincount(y).max() / len(y)))
    print("-> verify/out/clone_samples.npz")


if __name__ == "__main__":
    main()
