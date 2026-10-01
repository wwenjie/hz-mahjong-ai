"""抓同向听层高 gap 决策点的具体牌例，供 A 直接判读。

只读离线事件流，零平台请求。流式扫描，抓够目标数即停（不跑全量）。

目标桶（C31 判读定位的最大缺口）：2 向听 + 财神在手 + 摸序≥5，我方，gap≥阈值。
输出：markdown，每例含牌面、各候选 ukeire 对比、实际选择、最优选择。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay  # noqa: E402

OUR = "u_a7f7c67bb14a"
DISCARDED = "tile_discarded"
DRAWN = "tile_drawn"
COPIES = 4


def tile_name(t: int) -> str:
    return tiles.to_code(t)


def hand_str(hand: list[int]) -> str:
    parts = []
    for t, c in enumerate(hand):
        if c > 0:
            parts.append(f"{tile_name(t)}x{c}" if c > 1 else tile_name(t))
    return " ".join(parts)


def visible_counts(state) -> list[int]:
    vis = [0] * tiles.TILE_KINDS
    for seat in state.seats:
        for t in seat.discards:
            vis[t] += 1
        for m in seat.melds:
            for t in m.tiles:
                vis[t] += 1
    return vis


def ukeire(after: list[int], meld_n: int, s: int, vis: list[int], memo: dict) -> int:
    total = 0
    for t in range(tiles.TILE_KINDS):
        remaining = COPIES - vis[t] - after[t]
        if remaining <= 0:
            continue
        after[t] += 1
        try:
            sh = shanten_mod.shanten_any(after, meld_n, memo=memo)
        except Exception:  # noqa: BLE001
            sh = None
        after[t] -= 1
        if sh is not None and sh < s:
            total += remaining
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=10, help="抓多少例")
    ap.add_argument("--min-gap", type=int, default=4)
    ap.add_argument("--shanten", type=int, default=2, help="目标向听层")
    ap.add_argument("--min-draw", type=int, default=5, help="摸序下限(中盘)")
    ap.add_argument("--max-files", type=int, default=80, help="最多扫多少房")
    ap.add_argument("--out", default="agent/out/c31-examples.md")
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    files = files[: args.max_files]

    out = ROOT / args.out
    state_path = out.with_suffix(".state.json")
    # 幂等续跑：读已扫描位置与已抓实例
    start_idx = 0
    examples: list[dict] = []
    if state_path.exists():
        st_ = json.loads(state_path.read_text(encoding="utf-8"))
        if st_.get("want") == args.want and st_.get("min_gap") == args.min_gap:
            start_idx = st_.get("next_file_idx", 0)
            examples = st_.get("examples", [])
            print(f"[resume] 从文件 #{start_idx} 续，已有 {len(examples)} 例", file=sys.stderr, flush=True)

    memo: dict = {}
    scanned = start_idx
    for fi, path in enumerate(files):
        if fi < start_idx:
            continue
        if len(examples) >= args.want:
            break
        scanned = fi + 1
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        if len(ids) != 4:
            continue
        scanned += 1
        room = Path(path).stem
        try:
            rounds = list(replay.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            continue
        for ri, (state, events) in enumerate(rounds):
            draw_idx = [0, 0, 0, 0]
            for ev in events:
                if len(examples) >= args.want:
                    break
                etype = ev.get("type")
                seat = ev.get("seat")
                if etype == DRAWN and isinstance(seat, int) and 0 <= seat < 4:
                    draw_idx[seat] += 1
                    replay.apply_event(state, ev)
                    continue
                if etype != DISCARDED or not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if ids[seat] != OUR:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                before = list(state.seats[seat].hand)
                n = draw_idx[seat]
                if n < args.min_draw or before[tiles.GOD] <= 0:
                    try:
                        replay.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                after_actual = state.seats[seat].hand
                diff = [t for t in range(tiles.TILE_KINDS) if before[t] - after_actual[t] == 1]
                if len(diff) != 1:
                    continue
                meld_n = len(state.seats[seat].melds)
                if sum(before) != tiles.HAND_SIZE + 1 - tiles.MELD_SLOTS * meld_n:
                    continue
                try:
                    s_actual = shanten_mod.shanten(list(after_actual), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    continue
                if s_actual != args.shanten:
                    continue
                vis = visible_counts(state)
                vis[diff[0]] -= 1
                u_actual = ukeire(list(after_actual), meld_n, s_actual, vis, memo)
                cands = []
                best = u_actual
                best_t = diff[0]
                for t, c in enumerate(before):
                    if c <= 0 or t == diff[0]:
                        continue
                    cand = list(before)
                    cand[t] -= 1
                    try:
                        s_c = shanten_mod.shanten(cand, meld_n, memo=memo)
                    except Exception:  # noqa: BLE001
                        continue
                    if s_c != s_actual:
                        continue
                    u_c = ukeire(cand, meld_n, s_actual, vis, memo)
                    cands.append((t, u_c))
                    if u_c > best:
                        best = u_c
                        best_t = t
                gap = best - u_actual
                if gap >= args.min_gap:
                    cands.sort(key=lambda x: -x[1])
                    examples.append({
                        "room": room, "round": ri, "draw_n": n,
                        "hand": hand_str(before), "god_n": before[tiles.GOD],
                        "actual": tile_name(diff[0]), "u_actual": u_actual,
                        "best": tile_name(best_t), "u_best": best, "gap": gap,
                        "cands": [(tile_name(t), u) for t, u in cands],
                    })
        if scanned % 1 == 0:
            # 增量落盘：每扫完一房保存进度
            state_path.write_text(json.dumps({
                "want": args.want, "min_gap": args.min_gap,
                "next_file_idx": scanned, "examples": examples,
            }), encoding="utf-8")
        print(f"[scan {scanned}/{len(files)}] examples={len(examples)}/{args.want}", file=sys.stderr, flush=True)

    # 终稿（含可能为空的情况）
    lines = [
        "# C31 高 gap 牌例（2 向听 + 财神在手 + 摸序≥{}，我方，gap≥{}）".format(args.min_draw, args.min_gap),
        "",
        f"扫描 {scanned} 房，抓到 {len(examples)} 例。每例：手牌 → 实际切牌(进张) vs 最优切牌(进张)。",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"## 例 {i} — {ex['room']} 第{ex['round']}局 摸序{ex['draw_n']} 财神x{ex['god_n']}")
        lines.append(f"- 手牌：{ex['hand']}")
        lines.append(f"- **实际切 {ex['actual']} → ukeire {ex['u_actual']}**；最优切 {ex['best']} → ukeire {ex['u_best']}（**gap {ex['gap']}**）")
        top = "、".join(f"{t}({u})" for t, u in ex["cands"][:6])
        lines.append(f"- 其他候选：{top}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"落盘 {out}，{len(examples)} 例 / 扫 {scanned} 房")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
