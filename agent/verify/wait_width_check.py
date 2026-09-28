#!/usr/bin/env python3
"""独立复算「听牌时听口可见张数」按财神数分层（只读、离线、零平台请求）。

目的：A 的 `tools/analyze_wait_ceiling.py` / `analyze_god_conversion.py` 是**同一作者**
自建的仪器，而「听口宽度」是他当前 P0 的唯一依据。我用自己的重实现独立复算同一张表，
**不 import 他的工具、不 import 他的测量代码**（AGENTS.md §6「仪表要对拍」）。

我的口径（显式写出，与 A 的可能有细节差）：
  - 时点：**我们的出牌之后**，且该时点我方 `best_shanten(...)==0`（听牌）。
  - 可见张数 = Σ_{胡牌张 t} (4 − seen[t])，其中
    seen[t] = 我方暗手张数 + **四家副露吃碰杠的牌面** + **四家已打出牌**（含我方刚打的那张）。
  - 财神分层 = 该时点我方暗手里的财神数。
  - 剔除抓打圈强制的出牌（无法识别时不下结论——本脚本不识别，故统计全量并在结论标注）。

复用 `agent/verify/baotou_decompose.py` 的已对拍重建器（22,833 局自检 100%）。
用法: nice -n 19 uv run python agent/verify/wait_width_check.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent" / "verify"))

from majiang.rules import tiles  # noqa: E402
from majiang.rules import shanten as shantenmod  # noqa: E402
from majiang.rules import win as winmod  # noqa: E402

import baotou_decompose as B  # noqa: E402

GOD = B.GOD


def b_width_bucket(w):
    """B 的桶边（`verify/wait_width_table.py` width_bucket，预登记写死，勿改）。"""
    if w <= 4:
        return "1-4"
    if w <= 8:
        return "5-8"
    if w <= 12:
        return "9-12"
    if w <= 20:
        return "13-20"
    return "21+"


def replay_wait_widths(start_hands, events, our):
    """返回我方每个听牌出牌点的 (财神数, 可见张数, 是否抓打圈强制出牌)。

    第三项 `forced` 用于 B 口径对照：B 的 `wait_width_table.py` 会剔除抓打圈强制的出牌
    （`data.catch_play`），而本脚本原有输出统计全量。两套数字都要能出，故标记而非丢弃。
    """
    hands = {i: [0] * tiles.TILE_KINDS for i in range(4)}
    meld_tiles = {i: [0] * tiles.TILE_KINDS for i in range(4)}
    melds = {0: 0, 1: 0, 2: 0, 3: 0}
    discards = [0] * tiles.TILE_KINDS
    for i, codes in enumerate(start_hands or []):
        for c in codes:
            hands[i][tiles.parse(c)] += 1
    out = []
    for e in events:
        t = e.get("type")
        s = e.get("seat")
        if s is None or not (0 <= s < 4):
            continue
        if t == "tile_drawn":
            hands[s][tiles.parse(e["tile"])] += 1
        elif t == "tile_discarded":
            data = e.get("data") or {}
            tile = tiles.parse(e["tile"])
            hands[s][tile] -= 1
            discards[tile] += 1
            if s == our:
                # 出牌后时点：手牌是 13-3m 的等待型 → 用 shanten（不是 best_shanten，
                # 后者期望 14-3m 的摸牌后手牌，在此处会越界报错——我踩过这个坑）。
                try:
                    if shantenmod.shanten(hands[s], melds[s]) == 0:
                        seen = [0] * tiles.TILE_KINDS
                        for k in range(tiles.TILE_KINDS):
                            seen[k] = (hands[s][k] + meld_tiles[0][k] + meld_tiles[1][k]
                                       + meld_tiles[2][k] + meld_tiles[3][k] + discards[k])
                        wins = winmod.winning_draws(hands[s], melds[s])
                        vis = sum(max(0, tiles.COPIES_PER_KIND - seen[k]) for k in wins)
                        out.append((hands[s][GOD], vis, bool(data.get("catch_play"))))
                except Exception:
                    pass
        elif t == "peng":
            tile = tiles.parse(e["tile"])
            hands[s][tile] -= 2
            meld_tiles[s][tile] += 3
            melds[s] += 1
        elif t == "gang":
            kind = str((e.get("data") or {}).get("kind") or "ming")
            tile = tiles.parse(e["tile"])
            if kind == "an":
                hands[s][tile] -= 4
                meld_tiles[s][tile] += 4
                melds[s] += 1
            elif kind == "bu":
                hands[s][tile] -= 1
                meld_tiles[s][tile] += 1
            else:
                hands[s][tile] -= 3
                meld_tiles[s][tile] += 3
                melds[s] += 1
        elif t == "chi":
            data = e.get("data") or {}
            called = tiles.parse(e["tile"]) if e.get("tile") else None
            for c in (data.get("tiles") or []):
                cc = tiles.parse(c)
                meld_tiles[s][cc] += 1
                if cc == called:
                    continue
                hands[s][cc] -= 1
            melds[s] += 1
    return out


def room_of(path: str) -> str:
    return Path(path).parts[-3]


def decider_by_room() -> dict:
    """房间 → 当时跑的策略名（读 `sessions.jsonl` 台账）。

    **为什么必须过滤**：09-27 默认档从 v1（`tiebreak=blocks`，完全不看进张）切到 v2，
    09-28 又新增 v3 并上线真机。整份清单上的「我们」是多个策略的**混合**，
    既不代表 v1 也不代表 v2/v3。同一条警告适用于任何「我们 vs 对手」对照。
    我自己实现（不 import A 的 `tools/`），保持独立。
    """
    ledger = ROOT / "data" / "auto_sessions" / "sessions.jsonl"
    mapping = {}
    if not ledger.exists():
        return mapping
    for line in ledger.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("room_id"):
            mapping[e["room_id"]] = e.get("decider")
    return mapping


def filter_era(paths, eras):
    if not eras:
        return paths, 0
    mapping = decider_by_room()
    kept, unknown = [], 0
    for p in paths:
        d = mapping.get(room_of(p))
        if d is None:
            unknown += 1
            continue
        if any(d.startswith(x) for x in eras):
            kept.append(p)
    return kept, unknown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--era", default="",
                    help="逗号分隔的档位前缀（如 v2 / v3）；不给则不过滤（会跨时代混口径）")
    args = ap.parse_args()
    paths = sorted(glob.glob(str(ROOT / "data" / "auto_sessions" / "*" / "events" / "*.json")))
    if args.limit:
        paths = paths[:args.limit]
    eras = [x.strip() for x in args.era.split(",") if x.strip()]
    if eras:
        paths, unknown = filter_era(paths, eras)
        if unknown:
            print(f"  （{unknown} 个文件房间不在台账里，已剔除）")
        print(f"era 过滤 {eras} → {len(paths)} 个文件")
    else:
        print("⚠ 未做 era 过滤：整份清单是多个策略时代的混合，数字只代表「历史平均」")
    print(f"扫描 {len(paths)} 个文件（每文件 8 局）")

    bucket = {0: [], 1: [], 2: []}  # god -> [可见张数]（全量，原口径）
    b_widths = []                  # B 口径：剔抓打圈强制出牌后的可见张数
    n_points = 0
    for p in paths:
        try:
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        seats = doc.get("seats") or []
        our = next((i for i, s in enumerate(seats) if s.get("user_id") == B.OUR_UID), None)
        if our is None:
            continue
        for rn, r in B.group_rounds(doc).items():
            if not r["start_hands"]:
                continue
            try:
                for g, vis, forced in replay_wait_widths(r["start_hands"], r["events"], our):
                    bucket[min(g, 2)].append(vis)
                    n_points += 1
                    if not forced:
                        b_widths.append(vis)
            except Exception:
                continue

    print(f"我方听牌出牌点 {n_points}")
    print("\n=== 我方听牌时的听口可见张数（按财神数分层）===")
    labels = {0: "0 张", 1: "1 张", 2: ">=2 张"}
    for g in (0, 1, 2):
        v = bucket[g]
        if not v:
            print(f"  {labels[g]}: 无样本")
            continue
        mean = sum(v) / len(v)
        sv = sorted(v)
        p25 = sv[int(0.25 * len(sv))]
        print(f"  {labels[g]:>6}: n={len(v):5d}  均值 {mean:6.2f}  中位 {sv[len(sv)//2]:5.1f}  p25 {p25}")
    print("\n对照 A 的仪器（analyze_god_conversion.py，400 文件）：")
    print("  0 张 7.64 / 1 张 11.73 / >=2 张 23.09（我方）")

    # —— B 口径对照块（新增，不改上面任何输出）——
    # 用 B 的桶边 + B 的抓打圈过滤，输出 B 预登记要的两个量：均宽、窄桶(1-4)占比。
    if b_widths:
        dist = collections.Counter(b_width_bucket(w) for w in b_widths)
        n = len(b_widths)
        mean = sum(b_widths) / n
        print("\n=== B 口径对照（宽度分桶 + 均宽 + 窄桶占比；已剔抓打圈强制出牌）===")
        print(f"  听牌出牌点（剔抓打圈后）n={n}")
        print(f"  均宽 = {mean:.2f}")
        print("  分布: " + "  ".join(
            f"{b}={dist.get(b, 0)/n:.1%}" for b in ("1-4", "5-8", "9-12", "13-20", "21+")))
        print(f"  窄桶 1-4 占比 = {dist.get('1-4', 0)/n:.1%}")
        print("  （B 预登记：v2→v3 自对弈 均宽 12.33→13.37、窄桶 1-4 4.3%→0.5%；"
              "真机预期 窄桶占比下降、均宽 +0.5~1.0 张）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
