#!/usr/bin/env python
"""独立复算：从**决策轨迹**（`logs/*.jsonl` 的 `decision.made`）复核「同向听并列集退化 → 合计由喂牌单键决定」。

**为什么这是第三来源**：
- A 用 `probe_midgame_objective`（自建重建器）；
- C 用 `agent/verify/probe_candidate_truncation.py`（自建重建器）；
- researcher 用事件流 `data/auto_sessions/**` 全量重放；
- **本脚本既不复用 A 的工具、也不重放事件流**，只吃**进程内已经算好并落盘的决策轨迹**——它是
  「策略实际跑的那一刻」的快照，能独立回答「当时的候选与打分是怎样的」。

**口径**：只取 `phase=="draw"`、`choice` 为弃牌、且冠军档
（decider 含 `wait-aware-tenpai=True`）的记录。`detail.discards` 是策略按 `合计` 降序取的**前 4 名**
（`policy.py::_choose_discard` → `scores[:4]`），故本脚本只能在这 4 名内做交叉验证——
这**足以**检验「argmax(合计) 是否 = argmin(喂牌)」这条主命题（前 4 名里同名次者都可比）。

用法::

    uv run python verify-b/read_decision_traces.py                 # 全部日志
    uv run python verify-b/read_decision_traces.py --era v3        # 只冠军档
    uv run python verify-b/read_decision_traces.py --out verify-b/out/traces.json

退出码：0 正常；非 0 表示数据不足。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter

# 与 policy.py::DiscardScore.describe / DiscardScore 字段一一对应
LINE_RE = re.compile(
    r"^(?P<tile>\S+) 向听=(?P<shanten>-?\d+) 路线=(?P<route>\S+) "
    r"七对值=(?P<pair>[\d.]+) 副露值=(?P<meld>[\d.]+) "
    r"喂牌=(?P<feed>[\d.]+) 财神=(?P<god>[\d.]+) 合计=(?P<total>-?[\d.]+)$"
)
CHAMPION_MARKER = "wait-aware-tenpai=True"
# v3 只冻结 wait_aware_tenpai；喂牌权重沿用 `PolicyConfig` 默认（shape-feed-low 才改 3.0→1.0）。
# 用于把 total 的组内极差分解为 block + feed_weight×feed 两部分。
DEFAULT_FEED_WEIGHT = 3.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def parse_line(text: str) -> dict | None:
    match = LINE_RE.match(text.strip())
    if match is None:
        return None
    group = match.groupdict()
    return {
        "tile": group["tile"],
        "shanten": int(group["shanten"]),
        "route": group["route"],
        "pair": float(group["pair"]),
        "meld": float(group["meld"]),
        "feed": float(group["feed"]),
        "god": float(group["god"]),
        "total": float(group["total"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs/*.jsonl")
    parser.add_argument("--era", choices=["all", "v3"], default="v3",
                        help="v3=只取冠军档（decider 含 wait-aware-tenpai=True）")
    parser.add_argument("--out", default="verify-b/out/decision-traces.json")
    args = parser.parse_args()

    n_lines = 0
    n_decisions = 0
    n_detail = 0
    n_parsed = 0
    n_parse_fail = 0
    tie_sizes: Counter[int] = Counter()
    winner_is_top = 0            # 展示里第 0 名是否即 choice
    winner_feed_is_min_same_s = 0   # 同名次内，winner 喂牌是否最小
    total_argmax_is_feedmin = 0     # 前4名整体：argmax(合计) 是否 = argmin(喂牌)
    identical_feed_set = 0          # 前4名喂牌全相等
    feed_spread_sum = 0.0
    total_spread_sum = 0.0
    feed_spread_n = 0
    feed_spreads: list[float] = []
    total_spreads: list[float] = []
    shanten_hist: Counter[int] = Counter()

    for path in sorted(glob.glob(args.logs)):
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                n_lines += 1
                if '"event": "decision.made"' not in raw:
                    continue
                n_decisions += 1
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if rec.get("phase") != "draw":
                    continue
                choice = rec.get("choice") or ""
                if not choice.startswith("discard:"):
                    continue
                if args.era == "v3" and CHAMPION_MARKER not in (rec.get("decider") or ""):
                    continue
                detail = rec.get("detail") or {}
                rows = detail.get("discards") or []
                if not rows:
                    continue
                n_detail += 1
                parsed = []
                for row in rows:
                    item = parse_line(row)
                    if item is None:
                        n_parse_fail += 1
                        continue
                    parsed.append(item)
                if not parsed:
                    continue
                n_parsed += 1

                winner_tile = choice.split(":", 1)[1]
                if parsed[0]["tile"] == winner_tile:
                    winner_is_top += 1

                # 同名次（与展示第一名的向听相同）子集
                top_s = parsed[0]["shanten"]
                shanten_hist[top_s] += 1
                same = [item for item in parsed if item["shanten"] == top_s]
                tie_sizes[len(same)] += 1
                min_feed = min(item["feed"] for item in same)
                chosen = next(
                    (item for item in parsed if item["tile"] == winner_tile), parsed[0]
                )
                if abs(chosen["feed"] - min_feed) < 1e-9:
                    winner_feed_is_min_same_s += 1
                if len(same) >= 2:
                    fs = max(item["feed"] for item in same) - min_feed
                    ts = max(item["total"] for item in same) - min(
                        item["total"] for item in same
                    )
                    feed_spread_sum += fs
                    total_spread_sum += ts
                    feed_spreads.append(fs)
                    total_spreads.append(ts)
                    feed_spread_n += 1

                # 前 4 名整体：argmax(合计) 的喂牌是否即全局最小喂牌
                by_total = max(parsed, key=lambda item: item["total"])
                feeds = [item["feed"] for item in parsed]
                if len(set(round(f, 9) for f in feeds)) == 1:
                    identical_feed_set += 1
                elif abs(by_total["feed"] - min(feeds)) < 1e-9:
                    total_argmax_is_feedmin += 1

    summary = {
        "era": args.era,
        "n_decision_lines": n_decisions,
        "n_with_discard_detail": n_detail,
        "n_parsed": n_parsed,
        "n_parse_fail": n_parse_fail,
        "winner_is_displayed_top": winner_is_top,
        "winner_is_displayed_top_pct": round(100.0 * winner_is_top / max(1, n_parsed), 2),
        "same_shanten_winner_feed_is_min": winner_feed_is_min_same_s,
        "same_shanten_winner_feed_is_min_pct": round(
            100.0 * winner_feed_is_min_same_s / max(1, n_parsed), 2
        ),
        "top4_argmax_total_equals_feedmin": total_argmax_is_feedmin,
        "top4_argmax_total_equals_feedmin_pct": round(
            100.0 * total_argmax_is_feedmin / max(1, n_parsed), 2
        ),
        "top4_all_feed_equal": identical_feed_set,
        "top4_all_feed_equal_pct": round(100.0 * identical_feed_set / max(1, n_parsed), 2),
        "mean_feed_spread_same_shanten": round(feed_spread_sum / max(1, feed_spread_n), 4),
        "mean_total_spread_same_shanten": round(total_spread_sum / max(1, feed_spread_n), 4),
        "median_feed_spread_same_shanten": round(_median(feed_spreads), 4),
        "median_total_spread_same_shanten": round(_median(total_spreads), 4),
        # total = block - feed_weight*feed（同向听内偏移量相同，取组内极差即可分离）
        # ⇒ block 的组内极差 ≈ total 极差 - feed_weight × feed 极差。
        "implied_block_spread_mean": round(
            (total_spread_sum - DEFAULT_FEED_WEIGHT * feed_spread_sum) / max(1, feed_spread_n), 4
        ),
        "feed_weight_used": DEFAULT_FEED_WEIGHT,
        "feed_share_of_total_spread_mean": round(
            DEFAULT_FEED_WEIGHT * feed_spread_sum / max(1e-9, total_spread_sum), 4
        ),
        "shanten_hist": dict(sorted(shanten_hist.items())),
        "tie_size_hist": dict(sorted(tie_sizes.items())),
        "note": (
            "来源=进程内决策轨迹（非事件流重放、非重建器）；"
            "detail.discards 只含按合计降序的前 4 名"
        ),
    }
    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if n_parsed > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
