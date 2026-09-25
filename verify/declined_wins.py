#!/usr/bin/env python3
"""独立复核「弃胡求爆头（chase_baotou）」反事实的结构数字（A 报：2098 可胡摸牌 /
311 弃胡 / 265 自补，264/265 番数翻倍，净赚 +3.8 分/局）。

口径（自己从事件流重建，番数用共享规则层 compute_fan 计算）：
- 可胡摸牌时刻：我方 tile_drawn 后，compute_fan(摸前手, 摸牌, 副露数).hu 为真
- 弃胡：可胡但我方下一张事件是 tile_discarded（而不是 round_ended  winner=我方）
- 自补：弃胡后同一局最终仍由我方胡
- 番数翻倍：round_ended 的实际 fan ≥ 弃胡时点的 baseline fan × 2
  （baseline 用 chain_count=0 近似——链状态对翻倍结论的影响另行列出）

用法：uv run python verify/declined_wins.py
"""
import collections
import json

from majiang.rules import tiles
from majiang.rules.fan import compute_fan

OUR_UID = "u_a7f7c67bb14a"


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


def scan_round2(our_seat, rnd, out):
    """更直白的状态机：弃胡标记挂起，round_ended 统一结算。"""
    if rnd["hands"] is None:
        return
    hand = [0] * tiles.TILE_KINDS
    for code in rnd["hands"][our_seat]:
        hand[tiles.parse(code)] += 1
    meld_sets = 0
    open_decline = None  # 弃胡时点的 baseline fan
    round_decided = False

    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if et == "tile_drawn" and s == our_seat:
            t = tiles.parse(tile)
            res = compute_fan(hand, t, meld_sets)
            if res.hu:
                out["winnable_draws"] += 1
                if open_decline is None:
                    open_decline = ("winnable", res.fan)
            hand[t] += 1
        elif et == "tile_discarded" and s == our_seat:
            if open_decline and open_decline[0] == "winnable":
                # 可胡却出牌 → 弃胡成立
                out["declined"] += 1
                open_decline = ("declined", open_decline[1])
            hand[tiles.parse(tile)] -= 1
        elif et in ("chi", "peng", "gang") and s == our_seat:
            if et == "chi":
                need = collections.Counter(data.get("tiles") or [])
                need[tile] -= 1
                if need[tile] == 0:
                    del need[tile]
                for code, c in need.items():
                    hand[tiles.parse(code)] -= c
                meld_sets += 1
            elif et == "peng":
                hand[tiles.parse(tile)] -= 2
                meld_sets += 1
            else:
                kind = data.get("kind")
                hand[tiles.parse(tile)] -= {"an": 4, "bu": 1, "ming": 3}.get(kind, 0)
                if kind != "bu":
                    meld_sets += 1
            open_decline = None
        elif et == "round_ended":
            win_by_us = not data.get("draw") and s == our_seat
            if open_decline is not None:
                if open_decline[0] == "winnable":
                    out["took_immediately"] += 1  # 可胡且本局胡（未垫出牌）
                else:
                    base_fan = open_decline[1]
                    if win_by_us:
                        out["self_win_later"] += 1
                        actual = data.get("fan") or 0
                        if actual >= 2 * base_fan:
                            out["fan_doubled"] += 1
                        out["fan_pairs"].append((base_fan, actual))
                    else:
                        out["decline_lost"] += 1
            round_decided = True
    return round_decided


def main():
    files = [l.strip() for l in open("notes/manifest-20260926.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    out = collections.Counter()
    out["fan_pairs"] = []
    rounds = 0
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        seats = doc.get("seats") or []
        our = next((i for i, s in enumerate(seats) if s.get("user_id") == OUR_UID), None)
        if our is None:
            continue
        for rnd in split_rounds(doc):
            rounds += 1
            scan_round2(our, rnd, out)

    pairs = out["fan_pairs"]
    print(f"局数={rounds}")
    print(f"可胡摸牌时刻={out['winnable_draws']} 立即胡={out['took_immediately']} "
          f"弃胡={out['declined']}（占可胡 {out['declined'] / max(1, out['winnable_draws']):.1%}）")
    print(f"弃胡后自补={out['self_win_later']} 被截/流局={out['decline_lost']} "
          f"番数翻倍={out['fan_doubled']}/{out['self_win_later']}")
    if pairs:
        dist = collections.Counter(pairs)
        print("弃胡fan→实际fan 分布（前10）:", dist.most_common(10))


if __name__ == "__main__":
    main()
