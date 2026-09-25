#!/usr/bin/env python3
"""结构性不变量校验（独立实现，纯标准库，不 import 任何项目代码）。

输入：data/auto_sessions/*/events/*.json
每局校验（局边界 = blocks[].round_no 变化）：
  1. 守恒：34 种牌每种全场可见 ≤4（起手 53 + 摸牌），总量 136
  2. 出牌必在手：每次 tile_discarded 时该牌在出牌者手牌中
  3. 声称一致性：chi/peng/明杠 吃的是最近一次未被认领的出牌
  4. 牌墙：单局摸牌数 ≤ 136-53=83
  5. seq 连续性（缺号=采集丢失，单独报告，不判失败）
  6. rounds[] 摘要是流局记录的**有损视图**：实测三种行为并存（994 全收 / 110 只收非流局 /
     16 只收末位流局，与时间无关）。校验：摘要须命中三种规则之一且逐条内容与事件一致。
     **推论：流局数/有胡率只能以事件流 round_ended 为准。**

退出码：有 FAIL 级违反 → 1；全过 → 0。
用法：uv run python verify/invariants.py
"""
import collections
import glob
import json
import os
import sys

SUITS = [f"{n}{s}" for s in "wtb" for n in range(1, 10)]
HONORS = ["东", "南", "西", "北", "中", "发", "白"]
ALL_TILES = set(SUITS) | set(HONORS)
assert len(ALL_TILES) == 34, "牌种数应为 34"
WALL_TOTAL = 136
DEALT = 14 + 13 * 3  # 庄家 14，其余 13
MAX_DRAWS = WALL_TOTAL - DEALT  # 83；平台规则最后 20 张不摸，但不作硬判据

FAIL_KINDS = {
    "conserve_init", "conserve_draw", "discard_without_tile",
    "chi_hand_missing", "peng_hand_missing", "gang_hand_missing",
    "wall_overdraw", "missing_start_hands", "unknown_tile", "bad_seat",
}


def split_rounds(doc):
    """按 round_no 变化切局。blocks 是 ≤128 事件的分页块，同一局可跨多个 block；
    仅每局首 block 带非空 start_hands。"""
    rounds = []
    cur = None
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


class Violations:
    def __init__(self):
        self.items = []

    def add(self, kind, where, detail):
        self.items.append((kind, where, detail))

    def by_kind(self):
        return collections.Counter(k for k, _, _ in self.items)

    def fail_count(self):
        return sum(1 for k, _, _ in self.items if k in FAIL_KINDS)


def check_claim(vio, w2, seat, tile, need_from_hand, last_disc, hand, meld_sets):
    """chi/peng/明杠公共校验：吃的是最近一张未被认领的出牌；手牌够数。"""
    if last_disc is None or last_disc[0] != tile:
        vio.add("claim_not_last_discard", w2, f"claim={tile} last={last_disc}")
    elif last_disc[1]:
        vio.add("claim_already_claimed", w2, tile)
    else:
        last_disc[1] = True
    lacking = {t: c for t, c in need_from_hand.items() if hand[seat][t] < c}
    if lacking:
        vio.add("gang_hand_missing" if sum(need_from_hand.values()) >= 3 and len(need_from_hand) == 1 and list(need_from_hand.values())[0] >= 3 else "claim_hand_missing",
                w2, f"seat{seat} need={dict(need_from_hand)} have={dict(hand[seat])}")
        return False
    for t, c in need_from_hand.items():
        hand[seat][t] -= c
    return True


def check_round(path, rnd, vio, stats):
    rn, hands0, events = rnd["round_no"], rnd["hands"], rnd["events"]
    where = f"{os.path.basename(path)}#r{rn}"
    if hands0 is None:
        vio.add("missing_start_hands", where, "该局所有 block 均无 start_hands")
        return None
    if len(hands0) != 4:
        vio.add("bad_start_hands", where, f"len={len(hands0)}")
        return None

    hand = [collections.Counter(h) for h in hands0]
    dealer = rnd["dealer"]
    for s in range(4):
        for t in hand[s]:
            if t not in ALL_TILES:
                vio.add("unknown_tile", where, f"seat{s}:{t}")
        want = 14 if s == dealer else 13
        got = sum(hand[s].values())
        if got != want:
            vio.add("bad_hand_size_init", where, f"seat{s}={got} want={want} dealer={dealer}")

    visible = collections.Counter()
    for s in range(4):
        visible.update(hand[s])
    for t, c in visible.items():
        if c > 4:
            vio.add("conserve_init", where, f"{t}x{c}")

    meld_sets = [0, 0, 0, 0]  # 副露组数（补杠不加组）
    draws = 0
    last_disc = None  # [tile, claimed]
    n_round_ended = 0
    ended_events = []

    for e in events:
        et = e.get("type")
        seat = e.get("seat")
        tile = e.get("tile") or ""
        data = e.get("data") or {}
        seq = e.get("seq")
        w2 = f"{where}@{seq}"

        if et in ("tile_drawn", "tile_discarded", "chi", "peng", "gang"):
            if seat not in (0, 1, 2, 3):
                vio.add("bad_seat", w2, f"{et} seat={seat!r}")
                continue

        if et == "tile_drawn":
            if tile not in ALL_TILES:
                vio.add("unknown_tile", w2, repr(tile))
                continue
            hand[seat][tile] += 1
            visible[tile] += 1
            draws += 1
            if visible[tile] > 4:
                vio.add("conserve_draw", w2, f"{tile}x{visible[tile]}")
            if draws > MAX_DRAWS:
                vio.add("wall_overdraw", w2, f"draws={draws}")

        elif et == "tile_discarded":
            if tile not in ALL_TILES:
                vio.add("unknown_tile", w2, repr(tile))
                continue
            if hand[seat][tile] <= 0:
                vio.add("discard_without_tile", w2,
                        f"seat{seat} {tile} hand={dict(hand[seat])}")
            else:
                hand[seat][tile] -= 1
            want = 14 - 3 * meld_sets[seat]
            got = sum(hand[seat].values())
            if got != want - 1:
                vio.add("hand_size_at_discard", w2,
                        f"seat{seat} after={got} want={want - 1} melds={meld_sets[seat]}")
            last_disc = [tile, False]

        elif et == "chi":
            tiles = data.get("tiles") or []
            if tile not in tiles:
                vio.add("chi_claim_not_in_tiles", w2, f"tile={tile} tiles={tiles}")
            need = collections.Counter(tiles)
            if need[tile] > 0:
                need[tile] -= 1
                if need[tile] == 0:
                    del need[tile]
            if check_claim(vio, w2, seat, tile, need, last_disc, hand, meld_sets):
                meld_sets[seat] += 1
                stats["chi"] += 1

        elif et == "peng":
            need = collections.Counter({tile: 2})
            if check_claim(vio, w2, seat, tile, need, last_disc, hand, meld_sets):
                meld_sets[seat] += 1
                stats["peng"] += 1

        elif et == "gang":
            kind = data.get("kind")
            if kind == "an":
                need = collections.Counter({tile: 4})
                lacking = hand[seat][tile] < 4
                if lacking:
                    vio.add("gang_hand_missing", w2, f"an seat{seat} {tile} have={hand[seat][tile]}")
                else:
                    hand[seat][tile] -= 4
                    meld_sets[seat] += 1
                stats["gang_an"] += 1
            elif kind == "bu":
                if hand[seat][tile] < 1:
                    vio.add("gang_hand_missing", w2, f"bu seat{seat} {tile}")
                else:
                    hand[seat][tile] -= 1  # 补杠不加组数
                stats["gang_bu"] += 1
            elif kind == "ming":
                need = collections.Counter({tile: 3})
                if check_claim(vio, w2, seat, tile, need, last_disc, hand, meld_sets):
                    meld_sets[seat] += 1
                stats["gang_ming"] += 1
            else:
                vio.add("gang_unknown_kind", w2, repr(kind))

        elif et == "round_ended":
            n_round_ended += 1
            d = data
            if not d.get("draw") and seat not in (0, 1, 2, 3):
                vio.add("bad_seat", w2, f"winner seat={seat!r}")
            ended_events.append({
                "round_no": d.get("round_no"),
                "draw": bool(d.get("draw")),
                "winner": seat if seat in (0, 1, 2, 3) else None,
                "dealer": d.get("dealer"),
                "scores": d.get("scores"),
                "fan": d.get("fan"),
            })
            if d.get("round_no") != rn:
                vio.add("round_ended_round_no", w2,
                        f"event={d.get('round_no')} block={rn}")

    stats["rounds"] += 1
    stats["draws_total"] += draws
    stats["draws_max"] = max(stats["draws_max"], draws)
    if n_round_ended != 1 and rnd.get("file_status") == "finished":
        vio.add("round_ended_count", where, f"count={n_round_ended}")
    return ended_events


def check_file(path, vio, stats):
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as ex:
        vio.add("unreadable_file", path, repr(ex))
        return

    # seq 连续性（采集质量信号，非 FAIL）
    seqs = [e.get("seq") for b in doc.get("blocks", []) for e in (b.get("events") or [])]
    seqs = [s for s in seqs if isinstance(s, int)]
    if seqs:
        gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
        dup = len(seqs) - len(set(seqs))
        if gaps:
            stats["files_with_seq_gaps"] += 1
            stats["seq_gaps"] += gaps
        if dup:
            vio.add("seq_duplicate", os.path.basename(path), f"dup={dup}")

    status = doc.get("status")
    results = []
    for rnd in split_rounds(doc):
        rnd["file_status"] = status
        r = check_round(path, rnd, vio, stats)
        if r is not None:
            results.append((rnd["round_no"], r))
    decisive = [(rn, e) for rn, evs in results for e in evs if not e["draw"]]
    draws_ev = [(rn, e) for rn, evs in results for e in evs if e["draw"]]
    summary = doc.get("rounds") or []
    if status == "finished":
        # 三种实测收录规则：全收 / 只收非流局 / 非流局+末位流局
        cands = {
            "全收": decisive + draws_ev,
            "只收非流局": decisive,
            "非流局+末位": decisive + draws_ev[-1:],
        }
        matched_rule = None
        for name, cand in cands.items():
            if len(summary) == len(cand):
                matched_rule = name
                expect = cand
                break
        if matched_rule is None:
            vio.add("rounds_len_mismatch", os.path.basename(path),
                    f"rounds[]={len(summary)} 非流局={len(decisive)} 流局={len(draws_ev)}")
            expect = decisive + draws_ev
        else:
            stats[f"summary_rule_{matched_rule}"] += 1
        by_rn = {}
        for rn, e in expect:
            by_rn.setdefault(e["round_no"], []).append(e)
        for s in summary:
            evs = by_rn.get(s.get("round_no")) or []
            sw = s.get("winner")
            sw = None if sw in (-1, None) else sw
            m = [e for e in evs
                 if e["winner"] == sw
                 and e["dealer"] == s.get("dealer")
                 and e["scores"] == s.get("scores")]
            if not m:
                vio.add("rounds_entry_mismatch", os.path.basename(path),
                        f"summary={json.dumps(s, ensure_ascii=False)} "
                        f"events={json.dumps(evs, ensure_ascii=False)}")
    stats["files"] += 1
    stats[f"status_{status}"] += 1
    if not any(e.get("type") == "game_ended"
               for b in doc.get("blocks", []) for e in (b.get("events") or [])):
        stats["files_without_game_ended"] += 1


def main():
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    vio = Violations()
    stats = collections.Counter()
    for p in files:
        check_file(p, vio, stats)

    by = vio.by_kind()
    fail = vio.fail_count()
    report = {
        "snapshot": {
            "files": stats["files"],
            "files_finished": stats["status_finished"],
            "files_abandoned": stats["status_abandoned"],
            "files_without_game_ended": stats["files_without_game_ended"],
            "rounds": stats["rounds"],
        },
        "stats": {
            "draws_total": stats["draws_total"],
            "draws_max_per_round": stats["draws_max"],
            "chi": stats["chi"], "peng": stats["peng"],
            "gang_ming": stats["gang_ming"], "gang_bu": stats["gang_bu"],
            "gang_an": stats["gang_an"],
            "files_with_seq_gaps": stats["files_with_seq_gaps"],
            "seq_gaps": stats["seq_gaps"],
            "summary_rule_全收": stats["summary_rule_全收"],
            "summary_rule_只收非流局": stats["summary_rule_只收非流局"],
            "summary_rule_非流局+末位": stats["summary_rule_非流局+末位"],
        },
        "violations_by_kind": dict(by.most_common()),
        "fail_violations": fail,
        "violations_detail": [
            {"kind": k, "where": w, "detail": d} for k, w, d in vio.items
        ][:500],
    }
    os.makedirs("verify/out", exist_ok=True)
    with open("verify/out/invariants-latest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"files={stats['files']} finished={stats['status_finished']} "
          f"abandoned={stats['status_abandoned']} other_status="
          f"{stats['files'] - stats['status_finished'] - stats['status_abandoned']} "
          f"no_game_ended={stats['files_without_game_ended']} rounds={stats['rounds']}")
    print(f"draws_total={stats['draws_total']} draws_max/round={stats['draws_max']} "
          f"chi={stats['chi']} peng={stats['peng']} "
          f"gang(ming/bu/an)={stats['gang_ming']}/{stats['gang_bu']}/{stats['gang_an']}")
    print(f"seq: files_with_gaps={stats['files_with_seq_gaps']} gaps={stats['seq_gaps']}")
    print(f"summary_rule: 全收={stats['summary_rule_全收']} "
          f"只收非流局={stats['summary_rule_只收非流局']} "
          f"非流局+末位={stats['summary_rule_非流局+末位']}")
    print(f"violations: total={len(vio.items)} fail={fail}")
    for k, c in by.most_common():
        print(f"  {k}: {c}")
    print("report -> verify/out/invariants-latest.json")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
