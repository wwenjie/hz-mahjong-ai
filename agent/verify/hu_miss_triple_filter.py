#!/usr/bin/env python
"""模式1「该胡不胡」三重过滤复核（A 2026-10-06 18:31③ [待A/C]，C 认领）。

**背景**：agent-e 审计称 8,896 个听牌样本选错率 6.3%、其中模型 top1=hu 的 2,123 条里 52.1%
是真 bug（规则口径已成胡却打出）。A 18:31③ 裁决：必须**三重过滤**后才是真 bug：
  1. 不是「弃胡求爆头」（我们故意的策略，B' 01:59 数据支持整体 +19215 净分）；
  2. 不是抓打圈受限（`situation.is_restricted`）；
  3. 且用规则口径判定「已成胡」（`win.is_winning_shape`，不是模型口径）。

**口径（C 自拟，报出供裁决）**：
- 全量我方事件流（data/auto_sessions，room_id+round_no 去重）；
- 对每个我方 `tile_discarded` 事件：出牌**前**手牌 = 当前手牌 + 打出的那张（14 张）。
  若这 14 张里**存在某张打出后成胡**（即「这手本来能胡」），则是「该胡不胡」候选；
- 对候选判定：
  - **已成胡**（过滤 3）：用 `win.is_winning_shape` 判「14 张是否直接成胡」（摸打后胡），
    以及「去掉打出那张后 13 张是否听牌 + 打出那张恰为听口」；
  - **弃胡求爆头**（过滤 1）：调用生产 `HeuristicDecider._choose_win_or_piao` 的判定逻辑，
    看在该局面下是否会走「弃胡求爆头」分支（弃胡求爆头打出的是非财神牌）；
  - **抓打圈受限**（过滤 2）：事件流里 `is_restricted` 不可得（situation 快照无此字段的可靠来源），
    本探针**先不扣**（高估真 bug 的上限），在产物里标注。

**产物**：`agent/out/hu-miss-triple-filter.txt` + 逐例 `hu-miss-cases.jsonl`。
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles, win as win_mod  # noqa: E402
from majiang.sim import replay as R  # noqa: E402

OUR = "u_a7f7c67bb14a"
GOD_ID = tiles.GOD


def hand_codes(counts) -> list[str]:
    return [tiles.to_code(t) for t in range(tiles.TILE_KINDS) for _ in range(counts[t])]


def process_room(path, stats, cases, seen_rounds):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stats["bad_json"] += 1
        return
    room_id = doc.get("room_id") or Path(path).stem
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    if len(ids) != 4 or OUR not in ids:
        return
    my_seat = ids.index(OUR)
    try:
        rounds = list(R.iter_rounds(doc))
    except Exception:  # noqa: BLE001
        stats["iter_rounds"] += 1
        return

    for state, events in rounds:
        rno = getattr(state, "round_no", None)
        rkey = (room_id, rno)
        if rkey in seen_rounds:
            stats["dup_round"] += 1
            continue
        seen_rounds.add(rkey)

        for ev in events:
            if ev.get("type") != "tile_discarded" or ev.get("seat") != my_seat:
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                continue
            # 我方出牌点：出牌前手牌 = 当前 + 打出的那张
            tile_str = ev.get("tile")
            try:
                tile = tiles.parse(tile_str) if isinstance(tile_str, str) else int(tile_str)
            except Exception:  # noqa: BLE001
                stats["tile_parse_err"] += 1
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                continue
            stats["my_discard"] += 1
            # 正确口径：此刻 state.hand 已含摸到的牌（after-draw 决策态）= 出牌前手牌。
            # （18:38 版 bug：before = state.hand + tile 重复计数，打出牌本就在手——作废重跑）
            before = list(state.seats[my_seat].hand)
            meld_count = len(state.seats[my_seat].melds)

            # 「这手本来能胡」= 决策态手牌（14 张）直接成胡
            try:
                can_win_14 = win_mod.is_winning_shape(before, meld_count)
            except Exception:  # noqa: BLE001
                can_win_14 = False
            if not can_win_14:
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                continue

            stats["can_win_before_discard"] += 1
            # 过滤 1：打出后手牌是否爆头形（is_baotou）——弃胡求爆头/续飘策略的特征
            after = list(before)
            after[tile] -= 1  # 打出后 13 张
            try:
                is_baotou_after = win_mod.is_baotou(after, meld_count)
            except Exception:  # noqa: BLE001
                is_baotou_after = False
            if is_baotou_after:
                stats["filtered_baotou"] += 1
            else:
                stats["true_bug_candidate"] += 1
            cases.append({
                "room": room_id,
                "round_no": rno,
                "seq": ev.get("seq"),
                "tile": tile_str,
                "hand_before": hand_codes(before),
                "god_in_hand": before[GOD_ID],
                "n_melds": meld_count,
                "baotou_after": is_baotou_after,
            })
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                break


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    print(f"事件文档 {len(files)} 个", flush=True)
    stats = collections.Counter()
    cases: list[dict] = []
    seen_rounds: set = set()
    for f in files:
        process_room(f, stats, cases, seen_rounds)

    print(f"去重局数 = {len(seen_rounds)}", flush=True)
    print(f"我方出牌事件 = {stats['my_discard']}", flush=True)
    print(f"出牌前已成胡(14张 is_winning_shape) = {stats['can_win_before_discard']}", flush=True)

    # 落盘逐例
    out_cases = ROOT / "agent/out/hu-miss-cases.jsonl"
    with out_cases.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    lines = []
    lines.append(f"去重局数 = {len(seen_rounds)}")
    lines.append(f"我方出牌事件 = {stats['my_discard']}")
    lines.append(f"出牌前已成胡（is_winning_shape=True）= {stats['can_win_before_discard']}")
    lines.append(f"  ├─ 过滤 1：打出后仍爆头（is_baotou）= {stats['filtered_baotou']}  ← 弃胡求爆头策略（故意）")
    lines.append(f"  └─ 打出后非爆头（真 bug 候选）= {stats['true_bug_candidate']}  ← 仍需过滤 2（抓打圈受限）")
    lines.append("")
    if stats["my_discard"]:
        rate = stats["can_win_before_discard"] / stats["my_discard"]
        bug_rate = stats["true_bug_candidate"] / stats["my_discard"]
        lines.append(f"「已成胡却打出」占比 = {rate:.4%}")
        lines.append(f"扣掉弃胡求爆头后（真 bug 候选）占比 = {bug_rate:.4%}")
    lines.append("")
    lines.append("**口径声明**：")
    lines.append("- 「成胡」用 `win.is_winning_shape(14张)`（after-draw 决策态手牌）；打出后爆头用 `win.is_baotou(13张)`。")
    lines.append("- 过滤 1（弃胡求爆头）用「打出后仍爆头」近似——生产策略的弃胡求爆头正是「打出非财神牌后仍听任意」。")
    lines.append("- 过滤 2（抓打圈受限）未扣：事件流无 is_restricted 落盘，真 bug 候选数仍含它（上限）。")
    text = "\n".join(lines)
    print(text)
    (ROOT / "agent/out/hu-miss-triple-filter.txt").write_text(text, encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
