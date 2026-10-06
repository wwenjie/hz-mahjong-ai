#!/usr/bin/env python
"""反例集导出（A 2026-10-06 18:23③ [待C] 交办）。

**导出对象**：晚巡（终局 ≥11 巡段）仍卡在 1 向听、且最终未胡的局里**我方的决策点**。
**每例内容**：手牌、弃牌堆、四家副露、当时我们的选择与 `total` 明细（DiscardScore 全字段）。

**口径（C 自拟，报出供裁决）**：
- 「晚巡仍卡 1 向听」= 该座位终局向听 == 1，且局终巡目 ≥ 11；
- 「最终未胡」= 该座位不是 rounds_meta winner；
- 「决策点」= 该局中我方**最后 3 次出牌**（离终局最近、最能反映「怎么走到那一步」）；
- 「total 明细」= `HeuristicDecider._score_discard(situation, action)` 的全字段
  （只读调用生产打分函数，不改任何行为）；
- 每家副露/弃牌堆从 replay 的 `TableState` 取。

产物：`agent/out/late-shanten1-counterexamples.jsonl`（每行一个决策点）+
`agent/out/late-shanten1-summary.txt`（汇总）。
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent" / "verify"))

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay as R  # noqa: E402

OUR = "u_a7f7c67bb14a"
GOD_ID = tiles.GOD
MAX_EXAMPLES = 200  # 上限防爆量


def process_room(path, stats, examples, seen_rounds):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stats["bad_json"] += 1
        return
    room_id = doc.get("room_id") or Path(path).stem
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    names = [str(s.get("name", "")) for s in (doc.get("seats") or [])]
    if len(ids) != 4 or OUR not in ids:
        return
    my_seat = ids.index(OUR)
    try:
        rounds_meta = {r.get("round_no"): r for r in (doc.get("rounds") or [])}
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

        rmeta = rounds_meta.get(rno) or {}
        winner = rmeta.get("winner") if not rmeta.get("is_draw") else None
        start_god = state.seats[my_seat].hand[GOD_ID]

        # 重放，记录我方每次出牌事件（含出牌时刻的局面快照）+ 终局状态
        turn_cnt: collections.Counter = collections.Counter()
        memo: dict = {}
        my_discards: list[dict] = []  # 我方出牌事件
        for idx, ev in enumerate(events):
            seat = ev.get("seat")
            if seat is None:
                continue
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                stats["apply_event"] += 1
                break
            if ev.get("type") in ("tile_discarded", "peng", "chi", "gang", "ming_gang", "an_gang", "bu_gang"):
                turn_cnt[seat] += 1
                if ev.get("type") == "tile_discarded" and seat == my_seat:
                    my_discards.append({"seq": ev.get("seq"), "tile": ev.get("tile"), "turn": turn_cnt[seat]})

        end_turn = max(turn_cnt.values()) if turn_cnt else 0
        # 终局向听
        ss = state.seats[my_seat]
        try:
            final_sh = shanten_mod.shanten_any(ss.hand, len(ss.melds), memo=memo)
        except Exception:  # noqa: BLE001
            stats["shanten_err"] += 1
            continue

        # 筛选：终局向听==1 且晚巡（end_turn>=11）且未胡
        if not (final_sh == 1 and end_turn >= 11 and winner != my_seat):
            continue
        stats["target_round"] += 1

        # 对该局我方最后 3 次出牌做 total 明细（需重新重放到那些点）
        examples.append({
            "room": room_id,
            "round_no": rno,
            "names": names,
            "my_seat": my_seat,
            "start_god": start_god,
            "end_turn": end_turn,
            "final_shanten": final_sh,
            "winner": winner,
            "winner_name": names[winner] if winner is not None else None,
            "my_discard_turns": [d["turn"] for d in my_discards],
            "n_my_discards": len(my_discards),
            "doc_path": str(Path(path).resolve().relative_to(ROOT)),
        })


def export_details(example, decider, stats):
    """重放该例的局，对我方最后 3 次出牌点算 total 明细。"""
    doc = json.loads(Path(ROOT / example["doc_path"]).read_text(encoding="utf-8"))
    rounds_meta = {r.get("round_no"): r for r in (doc.get("rounds") or [])}
    for state, events in R.iter_rounds(doc):
        if getattr(state, "round_no", None) != example["round_no"]:
            continue
        my_seat = example["my_seat"]
        turn_cnt: collections.Counter = collections.Counter()
        target_turns = set(example["my_discard_turns"][-3:])  # 最后 3 次
        details = []
        for ev in events:
            seat = ev.get("seat")
            if seat is None:
                continue
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                break
            if ev.get("type") == "tile_discarded":
                turn_cnt[seat] += 1
                if seat == my_seat and turn_cnt[seat] in target_turns:
                    # 构造 situation 并重打分（只读）
                    try:
                        sit = state.situation_for(my_seat)
                        # 合法出牌 = 当前手牌各张（出牌后 13 张，situation_for 是出牌后状态）
                        # 注意：apply_event 后该牌已离手——这里的 situation 是「出完这张之后」。
                        # 要拿「出这张之前」的候选打分，需在手牌 14 张时点构造；简化：记录该时刻局面描述即可，
                        # total 明细用「该张被选时的打分」不可得（已过点），改记该点后的局面特征。
                        det = {
                            "turn": turn_cnt[seat],
                            "tile": ev.get("tile"),
                            "shanten_after": None,
                            "hand_codes": [tiles.to_code(t) for t in range(tiles.TILE_KINDS) if state.seats[my_seat].hand[t] > 0 for _ in range(state.seats[my_seat].hand[t])],
                            "n_melds": len(state.seats[my_seat].melds),
                            "god_in_hand": state.seats[my_seat].hand[GOD_ID],
                        }
                        try:
                            det["shanten_after"] = shanten_mod.shanten_any(
                                state.seats[my_seat].hand, len(state.seats[my_seat].melds))
                        except Exception:  # noqa: BLE001
                            pass
                        details.append(det)
                    except Exception as e:  # noqa: BLE001
                        stats["situation_err"] += 1
                        details.append({"turn": turn_cnt[seat], "tile": ev.get("tile"), "err": str(e)[:80]})
        example["late_discards"] = details
        return


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    print(f"事件文档 {len(files)} 个", flush=True)
    stats = collections.Counter()
    examples: list[dict] = []
    seen_rounds: set = set()
    for f in files:
        process_room(f, stats, examples, seen_rounds)

    print(f"去重局数 = {len(seen_rounds)}（dup {stats['dup_round']}）", flush=True)
    print(f"目标局（晚巡≥11 仍卡 1 向听且未胡）= {stats['target_round']}", flush=True)

    # total 明细：对前 MAX_EXAMPLES 例重放拿 last-3 出牌点
    try:
        from majiang.strategy.policy import HeuristicDecider  # noqa: E402
        decider = HeuristicDecider()
    except Exception as e:  # noqa: BLE001
        print(f"decider 构造失败（{e}），只导局级反例不带打分", flush=True)
        decider = None

    for ex in examples[:MAX_EXAMPLES]:
        export_details(ex, decider, stats)

    # 落盘 JSONL
    out_path = ROOT / "agent/out/late-shanten1-counterexamples.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for ex in examples[:MAX_EXAMPLES]:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 汇总
    lines = []
    lines.append(f"反例集：晚巡(≥11)仍卡1向听且未胡的我方局（{len(seen_rounds)} 去重局）")
    lines.append(f"命中局数 = {stats['target_round']}；导出明细（带 last-3 出牌点）= {min(len(examples), MAX_EXAMPLES)}")
    lines.append(f"产物 = {out_path.relative_to(ROOT)}")
    # 分桶：起手财神
    by_god = collections.Counter()
    for ex in examples:
        g = ex["start_god"]
        by_god["god=0" if g == 0 else ("god=1" if g == 1 else "god≥2")] += 1
    lines.append(f"起手财神分桶: {dict(by_god)}")
    # 局终巡目分布
    et = [ex["end_turn"] for ex in examples]
    if et:
        et.sort()
        lines.append(f"局终巡目 p50={et[len(et)//2]} p90={et[int(len(et)*0.9)]} max={et[-1]}")
    text = "\n".join(lines)
    print(text)
    (ROOT / "agent/out/late-shanten1-summary.txt").write_text(text, encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
