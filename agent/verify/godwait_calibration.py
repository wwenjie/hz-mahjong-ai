"""G1 第 1 步（A 23:50 裁决）：持财神听牌点的听口种数「口径 vs 行为」判定。

判据（A 预登记）：
  对每个「我们出牌后：持财神(≥1) 且 听牌(shanten==0)」的决策点，算两个数——
    (i) 主口径：win.winning_draws(counts_after, meld_count) 的听口种数
        （wait_aware_tenpai 正在优化的数；counts_after = 出牌后的 13-3m 张手牌）
    (ii) 独立口径：暴力枚举（财神逐个指派为具体牌种，tests/test_win_reference.py
        的参考实现，与本仓主实现搜索结构完全独立）的听口种数
  一致率 ≥95% ⇒ 口径没问题 ⇒ G1 进第 2 步（行为加权）；
  >5% 不一致 ⇒ 先修口径（bug），并评估 wait_aware_tenpai 历史读数是否被污染。

重建口径：逐事件 apply_event（sim/replay.py），在我们 tile_discarded 事件**之前**
取状态（此时手牌含刚摸的牌=14-3m 张），减掉实际打出的那张即为 counts_after。

产物：agent/out/godwait-calibration.txt
用法: setsid .venv/bin/python agent/verify/godwait_calibration.py > agent/out/godwait-calibration.txt 2>&1
"""
import json, glob, os, sys
from collections import Counter
from itertools import combinations_with_replacement

sys.path.insert(0, "src")
from majiang.rules import tiles, win
from majiang.rules import shanten as sh
from majiang.rules.tiles import GOD, TILE_KINDS
from majiang.sim import replay as R

OUR_NAME = "凤凰-5531"
SAMPLE_LIMIT = 1500


# ---- 独立暴力实现（照抄 tests/test_win_reference.py 的参考实现）----
def _plain_regular(counts, sets_needed):
    first = next((t for t, a in enumerate(counts) if a), None)
    if first is None:
        return sets_needed == 0
    if sets_needed == 0:
        return False
    if counts[first] >= 3:
        counts[first] -= 3
        ok = _plain_regular(counts, sets_needed - 1)
        counts[first] += 3
        if ok:
            return True
    if tiles.run_is_valid(first) and counts[first + 1] and counts[first + 2]:
        counts[first] -= 1
        counts[first + 1] -= 1
        counts[first + 2] -= 1
        ok = _plain_regular(counts, sets_needed - 1)
        counts[first] += 1
        counts[first + 1] += 1
        counts[first + 2] += 1
        if ok:
            return True
    return False


def _plain_winning(counts, meld_count):
    sets_needed = 4 - meld_count
    work = list(counts)
    for tile in range(TILE_KINDS):
        if work[tile] >= 2:
            work[tile] -= 2
            ok = _plain_regular(work, sets_needed)
            work[tile] += 2
            if ok:
                return True
    return False


def _plain_seven_pairs(counts):
    if sum(counts) != 14:
        return False
    return all(a % 2 == 0 for a in counts) and sum(a // 2 for a in counts) == 7


def brute_is_winning(counts, meld_count):
    if sum(counts) != 14 - 3 * meld_count:
        return False
    wildcards = counts[GOD]
    base = list(counts)
    base[GOD] = 0
    targets = [t for t in range(TILE_KINDS) if t != GOD]
    for combo in combinations_with_replacement(targets, wildcards):
        trial = list(base)
        for tile in combo:
            trial[tile] += 1
        if _plain_winning(trial, meld_count):
            return True
        if meld_count == 0 and _plain_seven_pairs(trial):
            return True
    return False


def brute_winning_draws(counts, meld_count):
    found = []
    for tile in range(TILE_KINDS):
        if counts[tile] >= tiles.COPIES_PER_KIND:
            continue
        work = list(counts)
        work[tile] += 1
        if brute_is_winning(work, meld_count):
            found.append(tile)
    return tuple(found)


def hand_str(counts):
    return [tiles.label(t) for t in range(TILE_KINDS) for _ in range(counts[t])]


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    files = sorted(glob.glob("data/auto_sessions/*/events/*.json"))
    print(f"事件流: {len(files)}", flush=True)

    agree = 0
    disagree = 0
    diff_samples = []
    errors = Counter()
    stats = Counter()
    n_files_done = 0

    for fi, fpath in enumerate(files):
        if agree + disagree >= SAMPLE_LIMIT:
            break
        if fi % 500 == 0:
            print(f"  扫描 {fi}/{len(files)}，已判 {agree + disagree} 点（不一致 {disagree}）", flush=True)
        try:
            payload = json.load(open(fpath))
        except Exception:
            errors["load"] += 1
            continue
        seats = payload.get("seats", [])
        our_seat = None
        for si, s in enumerate(seats):
            if s.get("name") == OUR_NAME:
                our_seat = si
                break
        if our_seat is None:
            continue
        n_files_done += 1

        try:
            for state, events in R.iter_rounds(payload):
                for event in events:
                    # 在我们出牌事件**之前**取状态（手牌含刚摸的牌）
                    if event.get("type") == "tile_discarded" and event.get("seat") == our_seat and state.opened:
                        stats["our_discards"] += 1
                        tile = R._tile_of(event.get("tile"))
                        seat_state = state.seats[our_seat]
                        hand = list(seat_state.hand)
                        meld_count = len(seat_state.melds)
                        if tile is not None and 0 <= tile < len(hand) and hand[tile] > 0:
                            counts_after = list(hand)
                            counts_after[tile] -= 1
                            god_after = counts_after[GOD]
                            if god_after >= 1:
                                stats["god>=1"] += 1
                                try:
                                    s_after = sh.shanten_any(counts_after, meld_count=meld_count)
                                except Exception:
                                    s_after = None
                                    errors["shanten"] += 1
                                if s_after == 0:
                                    stats["god>=1_tenpai"] += 1
                                    try:
                                        main_waits = win.winning_draws(counts_after, meld_count)
                                        brute_waits = brute_winning_draws(counts_after, meld_count)
                                        if set(main_waits) == set(brute_waits):
                                            agree += 1
                                        else:
                                            disagree += 1
                                            if len(diff_samples) < 20:
                                                diff_samples.append({
                                                    "file": os.path.basename(fpath),
                                                    "hand_after": hand_str(counts_after),
                                                    "god_n": god_after,
                                                    "meld_count": meld_count,
                                                    "main_n": len(main_waits),
                                                    "brute_n": len(brute_waits),
                                                    "main_waits": [tiles.label(t) for t in main_waits],
                                                    "brute_waits": [tiles.label(t) for t in brute_waits],
                                                })
                                    except Exception:
                                        errors["compare"] += 1
                    R.apply_event(state, event)
        except Exception as e:
            errors[f"replay:{type(e).__name__}"] += 1

    total = agree + disagree
    print(f"\n=== 采样统计 ===")
    print(f"含我们的文件: {n_files_done} / {len(files)}")
    print(f"我们出牌点: {stats['our_discards']}  持财神(≥1)出牌后: {stats['god>=1']}  其中听牌: {stats['god>=1_tenpai']}")
    print(f"\n=== 口径对比结果 ===")
    print(f"总判定决策点: {total}")
    if total:
        print(f"一致:   {agree} ({agree/total*100:.2f}%)")
        print(f"不一致: {disagree} ({disagree/total*100:.2f}%)")
        verdict = "口径一致 ≥95% ⇒ 口径无 bug，G1 进第 2 步（行为加权）" if agree / total >= 0.95 else "不一致 >5% ⇒ 口径有 bug，先修 winning_draws（并评估 wait_aware_tenpai 历史读数污染）"
        print(f"\nA 判据: {verdict}")
    if diff_samples:
        print(f"\n=== 不一致样本（前 {len(diff_samples)} 个）===")
        for i, s in enumerate(diff_samples, 1):
            print(f"\n样本 {i} [{s['file']}] 财神={s['god_n']} 副露={s['meld_count']}")
            print(f"  出牌后手牌: {s['hand_after']}")
            print(f"  主口径({s['main_n']}种): {s['main_waits']}")
            print(f"  暴力口径({s['brute_n']}种): {s['brute_waits']}")
    print(f"\n错误统计: {dict(errors)}")
    print("CALIBRATION_DONE", flush=True)


if __name__ == "__main__":
    main()
