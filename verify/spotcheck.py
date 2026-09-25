#!/usr/bin/env python3
"""手算小样本对照（第三层验收）：随机抽 10 局，打印可供肉眼核对原始 JSON 的明细。

用法：uv run python verify/spotcheck.py [--seed 42] [--n 10]
输出：每局的起手（按座位）、逐事件紧凑轨迹、终态（手牌张数/副露/牌墙剩余/赢家/番）。
核对方法：拿一局，用 `uv run python -m json.tool <file>` 或编辑器对照原始事件逐条走。
"""
import argparse
import collections
import json
import random

ALL_TILES = {f"{n}{s}" for s in "wtb" for n in range(1, 10)} | set("东南西北中发白")


def split_rounds(doc):
    rounds, cur = [], None
    for b in doc["blocks"]:
        rn = b.get("round_no")
        if cur is None or rn != cur["round_no"]:
            cur = {"round_no": rn, "dealer": b.get("dealer"), "hands": None, "events": []}
            rounds.append(cur)
            sh = b.get("start_hands")
            if sh and sh[0] is not None:
                cur["hands"] = sh
        cur["events"].extend(b.get("events") or [])
    return rounds


def render(path, rnd):
    hand = [collections.Counter(h) for h in rnd["hands"]]
    melds = [[] for _ in range(4)]
    draws = 0
    winner, fan, detail, scores, is_draw = None, None, None, None, False
    trace = []
    for e in rnd["events"]:
        et, s, tile, data = e["type"], e.get("seat"), e.get("tile") or "", e.get("data") or {}
        if et == "tile_drawn":
            hand[s][tile] += 1
            draws += 1
            trace.append(f"{e['seq']}:s{s}+摸{tile}")
        elif et == "tile_discarded":
            hand[s][tile] -= 1
            trace.append(f"{e['seq']}:s{s}-打{tile}")
        elif et == "chi":
            need = collections.Counter(data.get("tiles") or [])
            need[tile] -= 1
            if need[tile] == 0:
                del need[tile]
            for t, c in need.items():
                hand[s][t] -= c
            melds[s].append(f"吃{sorted(data.get('tiles') or [])}")
            trace.append(f"{e['seq']}:s{s}吃{tile}用{sorted(need.elements())}")
        elif et == "peng":
            hand[s][tile] -= 2
            melds[s].append(f"碰{tile}")
            trace.append(f"{e['seq']}:s{s}碰{tile}")
        elif et == "gang":
            k = data.get("kind")
            hand[s][tile] -= {"an": 4, "bu": 1, "ming": 3}[k]
            melds[s].append(f"杠({k}){tile}")
            trace.append(f"{e['seq']}:s{s}杠{k}:{tile}")
        elif et == "round_ended":
            is_draw = bool(data.get("draw"))
            winner = None if is_draw else s
            fan, detail, scores = data.get("fan"), data.get("detail"), data.get("scores")
            trace.append(f"{e['seq']}:终 draw={is_draw} winner={winner} fan={fan}")
    wall = 136 - 53 - draws
    print(f"\n=== {path.split('/')[-1]} round_no={rnd['round_no']} dealer={rnd['dealer']}")
    for s in range(4):
        print(f"  s{s} 起手{sum(collections.Counter(rnd['hands'][s]).values())}张: "
              f"{sorted(rnd['hands'][s])}")
    print("  轨迹:", " | ".join(trace))
    for s in range(4):
        n = sum(hand[s].values())
        print(f"  s{s} 终态手牌{n}张={sorted(hand[s].elements())} 副露{len(melds[s])}组:{melds[s]}")
    print(f"  牌墙剩余={wall} 摸牌数={draws} 赢家={winner} 番={fan} 细节={detail} 分差={scores}")
    return {"wall": wall, "draws": draws, "winner": winner, "fan": fan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--manifest", default="notes/manifest-20260926.txt")
    args = ap.parse_args()

    files = [l.strip() for l in open(args.manifest, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    pool = []
    for p in files:
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("status") != "finished":
            continue
        for rnd in split_rounds(doc):
            if rnd["hands"]:
                pool.append((p, rnd))
    random.seed(args.seed)
    sample = random.sample(pool, args.n)
    print(f"池子={len(pool)}局 抽{len(sample)}局 seed={args.seed}")
    for p, rnd in sample:
        render(p, rnd)


if __name__ == "__main__":
    main()
