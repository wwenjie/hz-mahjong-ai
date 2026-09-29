#!/usr/bin/env python
"""B6：从原始事件流独立复算比赛指标（不 import A 的 replay/measure、不 import verify/**）。

**自建解析**（只读 data/），逐层验证：

第 1 层 · 结构性不变量
  (i)  每局 `scores` 四家和为 0。
  (ii) 每局 `winner` 的分数 = 四家最大分。
  (iii) 顶层 `rounds[k].scores` 与 `round_ended` 事件的 `data.scores` 两处独立记录一致。
  (iv) 每局各家 `start_hands` 均为 13 张；**四种牌型全场 ≤4**（跨四家起手合计）。
  (v)  我方（OUR）手牌全量重放：起手 + 我方摸牌 − 我方弃牌 − 我方副露，
       **任一时刻各牌种计数 ∈ [0,4]**，且**每次弃出的牌当时确在我方手上**。
  (vi) 任一家「弃牌 + 副露」中同一牌种累计 ≤ 4（必要条件，不追踪墙序）。

第 2 层 · 指标（口径与 A/B 独立）
  - 我方总得分、名次分（+3/+1/−1/−3）、胡局数、放铳（本变体应恒 0，因无点炮）。

第 3 层 · 小样本手算对照（可选打印，人工可核）

用法：.venv/bin/python agent/verify/recompute_metrics_probe.py [--rooms N]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUR = "u_a7f7c67bb14a"
CPK = 4
SEATS = 4
RANK = {0: +3, 1: +1, 2: -1, 3: -3}


def load(path: str):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.rooms:
        files = files[: args.rooms]
    print(f"扫描事件流文件：{len(files)} 个（1 文件 = 1 场 = 8 局）", flush=True)

    bad: list[str] = []
    n_rounds = 0
    our_total = 0
    our_rank = 0
    our_wins = 0
    our_rounds = 0
    sample_printed = 0

    for path in files:
        fname = pathlib.Path(path).name
        try:
            doc = load(path)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{path}: 无法解析 {exc}")
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != SEATS or OUR not in ids:
            continue
        mine = ids.index(OUR)
        rounds = doc.get("rounds") or []

        # 按 round_no 归并 blocks，取首个带 start_hands 的 block
        blocks_by_round: dict[int, list[dict]] = collections.defaultdict(list)
        starts: dict[int, list[list[str]]] = {}
        for b in doc.get("blocks") or []:
            rn = b.get("round_no")
            blocks_by_round[rn].append(b)
            sh = b.get("start_hands")
            if rn not in starts and sh:
                starts[rn] = sh

        # 顶层 rounds 与 round_ended 事件对读
        ended = {}
        for b in doc.get("blocks") or []:
            for e in b.get("events") or []:
                if e.get("type") == "round_ended":
                    d = e.get("data") or {}
                    ended[d.get("round_no")] = d

        for r in rounds:
            rn = r.get("round_no")
            n_rounds += 1
            sc = r.get("scores") or []
            # (i) 和为 0
            if len(sc) != SEATS or sum(sc) != 0:
                bad.append(f"{fname} r{rn}: scores 不守恒 {sc}")
            # (ii) winner = 最大分（流局 winner=-1 不适用，跳过）
            if r.get("winner") != -1:
                if r.get("winner") not in range(SEATS) or sc[r["winner"]] != max(sc):
                    bad.append(f"{fname} r{rn}: winner={r.get('winner')} 非最大分 {sc}")
            # (iii) 与 round_ended 一致
            e = ended.get(rn)
            if e is not None and e.get("scores") != sc:
                bad.append(f"{fname} r{rn}: rounds.scores {sc} != round_ended {e.get('scores')}")
            # (iv) start_hands 张数与 ≤4
            sh = starts.get(rn)
            if sh and all(h is not None for h in sh):
                lens = [len(h) for h in sh]
                # 实测约定：庄家（本局首个弃牌者）起手 14 张，其余三家 13 张，合计 53。
                # `start_hands` 只为每局的**首个** block 填充；续块为 [None]*4（已过滤）。
                if len(sh) != SEATS or sorted(lens) != [13, 13, 13, 14]:
                    bad.append(f"{fname} r{rn}: start_hands 张数异常 {lens}")
                else:
                    dealer_seat = lens.index(14)
                    first_disc = next((ev.get("seat") for b in blocks_by_round.get(rn, [])
                                       for ev in (b.get("events") or [])
                                       if ev.get("type") == "tile_discarded"), None)
                    if first_disc is not None and first_disc != dealer_seat:
                        bad.append(f"{fname} r{rn}: 14 张座 {dealer_seat} != 首个弃牌座 {first_disc}")
                cnt = collections.Counter(t for h in sh for t in h)
                over = {k: v for k, v in cnt.items() if v > CPK}
                if over:
                    bad.append(f"{fname} r{rn}: 起手跨家有牌种 >4 {over}")
                if sum(lens) != 53:
                    bad.append(f"{fname} r{rn}: 起手总数 {sum(lens)} != 53")
            # (v)(vi) 事件重放
            if sh and all(h is not None for h in sh):
                mine_hand = collections.Counter(sh[mine])
                per_seat_meld = [collections.Counter() for _ in range(SEATS)]
                for b in blocks_by_round.get(rn, []):
                    for ev in b.get("events") or []:
                        et, seat, tile = ev.get("type"), ev.get("seat"), ev.get("tile")
                        data = ev.get("data") or {}
                        if et == "tile_drawn" and seat == mine and tile:
                            mine_hand[tile] += 1
                            if mine_hand[tile] > CPK:
                                bad.append(f"{fname} r{rn}: 我方摸牌后 {tile} >4")
                        elif et == "tile_discarded":
                            if seat == mine:
                                if tile not in mine_hand or mine_hand[tile] <= 0:
                                    bad.append(f"{fname} r{rn}: 我方弃出 {tile} 当时不在手")
                                else:
                                    mine_hand[tile] -= 1
                            if isinstance(seat, int) and 0 <= seat < SEATS:
                                per_seat_meld[seat][tile] += 1
                        elif et in ("peng", "chi", "gang"):
                            # 副露事件形状（实测）：
                            #   chi  : data.tiles = [3 张]，含被吃的那张；从手牌只移 2 张
                            #   peng : tile = 单张码，共 3 张；从手牌移 2 张
                            #   gang : kind=ming(3+1)/an(4)/bu(碰升级,只加 1)；data.tiles 不提供
                            kind = data.get("kind")
                            if et == "chi":
                                meld_tiles = list(data.get("tiles") or [])
                                # 被吃的那张来自放牌者，其余 2 张来自手牌
                                from_hand = list(meld_tiles)
                                if from_hand and tile in from_hand:
                                    from_hand.remove(tile)
                                hand_tiles = from_hand
                            elif et == "peng":
                                meld_tiles = [tile, tile, tile]
                                hand_tiles = [tile, tile]
                            else:  # gang
                                if kind == "bu":
                                    meld_tiles = [tile]
                                    hand_tiles = [tile]
                                elif kind == "an":
                                    meld_tiles = [tile, tile, tile, tile]
                                    hand_tiles = [tile, tile, tile, tile]
                                else:  # ming
                                    meld_tiles = [tile, tile, tile, tile]
                                    hand_tiles = [tile, tile, tile]
                            if isinstance(seat, int) and 0 <= seat < SEATS:
                                for t in meld_tiles:
                                    per_seat_meld[seat][t] += 1
                            if seat == mine:
                                for t in hand_tiles:
                                    if t in mine_hand and mine_hand[t] > 0:
                                        mine_hand[t] -= 1
                                    else:
                                        bad.append(f"{fname} r{rn}: 我方副露({et})消耗 {t} 时不在手")
                for i, c in enumerate(per_seat_meld):
                    over = {k: v for k, v in c.items() if v > CPK}
                    if over:
                        bad.append(f"{fname} r{rn}: 座{i} 弃牌+副露 牌种 >4 {over}")

            # 第 2 层指标
            our_rounds += 1
            our_total += sc[mine]
            our_wins += 1 if r.get("winner") == mine else 0
            order = sorted(range(SEATS), key=lambda i: (-sc[i], i))
            our_rank += RANK[order.index(mine)]

            if sample_printed < 3 and sc:
                sample_printed += 1
                print(f"  [手算对照] {fname} r{rn}: scores={sc} winner={r.get('winner')} "
                      f"→ 我方(mine={mine}) 得分 {sc[mine]} 名次 {order.index(mine)+1}")

    print("\n" + "=" * 74)
    print(f"局数={n_rounds}  我方局数={our_rounds}")
    print(f"我方总得分={our_total}  名次分={our_rank}  胡局数={our_wins}  "
          f"胡率={our_wins/our_rounds:.1%}" if our_rounds else "n/a")
    print(f"不变量违规数={len(bad)}")
    for b in bad[:20]:
        print(f"  - {b}")
    if len(bad) > 20:
        print(f"  … 另有 {len(bad)-20} 条")
    print("=" * 74)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
