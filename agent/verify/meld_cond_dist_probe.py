"""1a：头部 bot 副露条件分布（向听×巡目×牌型×副露后胡率/番数）——A 2026-10-03 21:35 交办。

口径（A 21:35 原文）：
  对 topbot_gap_analysis.py 已认出的头部 20 bot，把它们的每次 peng/chi/gang 事件按下列键切片，
  **只统计事件本身、不做前进模拟**：
  - 向听状态：replay 复算该时刻该家的向听（副露前）、以及副露后向听（判「向听是否下降」）；
  - 巡目（该家已摸牌次数）4 档：1-4 / 5-8 / 9-12 / ≥13；
  - 牌型：役牌（箭/自风/场风）、宝牌/财神相关、是否对子碰第 3 组、吃的是否两侧（两面 vs 嵌张/边张）；
  - 结果：该家最后是否胡、胡时番数（从结算事件取）。
  产出：bot 在 巡目X × 向听Y × 牌型Z 下的副露率 + 副露后的胡牌率与均番（1c 闸门阈值的经验分布）。

实现要点（本仓库既有踩坑的规避）：
- 分块幂等落盘（chunk 目录带参数指纹 cs{chunk}-n{文件数}），DONE 标记收口；`--rooms` 冒烟。
- shanten 口径（实测 21:50）：replay 后手牌是「摸后未打」态（13,10,7,4 张），
  shanten() 期望 3n+1（13/10/7/4）⇒ 副露前直接用；副露后 replay 先移除整组（11 张），
  须把被打的那 1 张补回凑成 3m+1（10 张）再算。杠（kind=ming/an/bu）后不补打，手牌恰为
  3m+1，直接算。补牌用加杠事件的 data.tiles（明杠含第四张）。
- memo 生命周期限制在单事件内（A 00:55 裁决：全局缓存会膨胀到 7GB+）。
- 巡目 = 该家 tile_drawn 计数（含杠后补摸）。牌型键：
  yakuhai=箭牌(中发白)；god=财神(白板)；pair3rd=碰时手牌含该牌第3+张；
  chi_shape=ryanmen（所吃牌在顺子端点）vs kanchan/penchan；gang_kind=ming/an/bu。

用法：
  setsid .venv/bin/python agent/verify/meld_cond_dist_probe.py > agent/out/meld-cond.log 2>&1
  .venv/bin/python agent/verify/meld_cond_dist_probe.py --rooms 60   # 冒烟
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majiang.rules import shanten as sh_mod  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.sim import replay  # noqa: E402

TOP_BOTS = [
    "玄武-2346", "三杯猫", "铳一色14", "歪比巴卜肉蛋葱鸡", "爆头研究所",
    "Astra-0", "腾蛇-0638", "康陶应雀", "凤凰-2626", "glm-flash",
    "麒麟-7780", "白虎-0211", "豆包豆包帮我把其他AI电源拔掉",
    "Nomad", "双白平胡", "Kimi-K4.1", "菜菜子", "走马", "今晚打老虎",
    "晴总总，该请桂语山房了",
]
MELD_TYPES = {"peng", "chi", "gang"}

# 牌型维度键（多标签，一张牌可同时命中）
def tile_tags(tile: int) -> list[str]:
    tags = []
    if tile == tiles.GOD:
        tags.append("god")  # 财神（白板）
    if tiles.HONOR_START <= tile < tiles.HONOR_START + tiles.HONORS.index("白"):
        tags.append("honor_other")  # 风牌（东/南/西/北）
    name = tiles.to_code(tile)
    if name in ("中", "发"):
        tags.append("yakuhai")  # 箭牌（杭州麻将白=财神，箭牌只余中发）
    if not tags:
        tags.append("suit")
    return tags


def _parse_tile(raw) -> int | None:
    """事件里的牌是牌码字符串（'南'/'3t'），用 tiles.parse 转 int。"""
    if raw in (None, ""):
        return None
    try:
        return tiles.parse(str(raw))
    except Exception:  # noqa: BLE001
        return None


def chi_shape(eaten: int, meld_tiles_raw: list) -> str:
    """吃的是否两侧：所吃牌在顺子中间=嵌张；在端点且顺子为 1-2/8-9 边界=边张；否则两面。"""
    s = sorted(t for t in (_parse_tile(x) for x in meld_tiles_raw) if t is not None)
    if len(s) != 3:
        return "unknown"
    lo, mid, hi = s
    if eaten == mid:
        return "kanchan"  # 嵌张
    if eaten == lo:
        if hi % 9 == 8:  # 7-8-9
            return "penchan"
        return "ryanmen"
    if eaten == hi:
        if lo % 9 == 0:  # 1-2-3
            return "penchan"
        return "ryanmen"
    return "unknown"


def process_batch(batch: list[str], chunk_path: Path, uid_to_name: dict[str, str]) -> int:
    # meld 事件聚合：key=(who, turn_bucket, shanten_before, tag) ->
    #   [meld_n, win_n, fan_sum, shanten_drop_n]
    agg: dict = collections.defaultdict(lambda: [0, 0, 0, 0])
    # 机会分母：key=(who, turn_bucket, shanten_at_draw) -> draw_n
    draw_agg: dict = collections.defaultdict(int)
    rooms = 0
    for path in batch:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        seats = doc.get("seats", [])
        if len(seats) != 4:
            continue
        uid_seat = {}
        for si, s in enumerate(seats):
            u = str(s.get("user_id", ""))
            if u in uid_to_name:
                uid_seat[u] = si
        if not uid_seat:
            continue
        rooms += 1

        # 轮级结果：round_no -> (winner_seat, fan)
        round_result: dict[int, tuple[int, int]] = {}
        for blk in doc.get("blocks", []):
            for e in blk.get("events", []):
                if e.get("type") == "round_ended":
                    d = e.get("data") or {}
                    rn = d.get("round_no")
                    if rn is not None and not d.get("draw"):
                        # winner: seat with positive score delta
                        scores = d.get("scores") or []
                        win_seat = max(range(len(scores)), key=lambda i: scores[i]) if scores else e.get("seat")
                        round_result[rn] = (win_seat, d.get("fan", 0))

        # 事件级重放
        state = replay.from_payload(doc)
        draw_count = [0, 0, 0, 0]  # 巡目计数（含杠后补摸）
        round_no = 0
        for ev in replay.all_events(doc):
            t = ev.get("type")
            seat = ev.get("seat")
            if t == "tile_drawn" and seat is not None:
                draw_count[seat] += 1
                n = draw_count[seat]
                # 机会分母：此时该家向听
                hand = state.seats[seat].hand
                meld_n = len(state.seats[seat].melds)
                memo: dict = {}
                try:
                    s_now = sh_mod.shanten(list(hand), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    s_now = None
                if s_now is not None:
                    tb = "1-4" if n <= 4 else ("5-8" if n <= 8 else ("9-12" if n <= 12 else "13+"))
                    sb = str(s_now) if s_now <= 3 else "4+"
                    # 判断该 seat 是否目标 bot
                    for u, si in uid_seat.items():
                        if si == seat:
                            draw_agg[(uid_to_name[u], tb, sb)] += 1
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                continue
            if t in MELD_TYPES and seat is not None:
                is_target = any(si == seat for si in uid_seat.values())
                who = None
                for u, si in uid_seat.items():
                    if si == seat:
                        who = uid_to_name[u]
                before = list(state.seats[seat].hand)
                meld_n = len(state.seats[seat].melds)
                memo = {}
                try:
                    s_before = sh_mod.shanten(list(before), meld_n, memo=memo)
                except Exception:  # noqa: BLE001
                    s_before = None
                tile = _parse_tile(ev.get("tile"))
                data = ev.get("data") or {}
                try:
                    replay.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break
                after = list(state.seats[seat].hand)
                meld_n_a = len(state.seats[seat].melds)
                # 副露后向听：碰/吃后补回待打的那 1 张凑 3m+1；杠后恰为 3m+1
                s_after = None
                memo = {}
                try:
                    if t in ("peng", "chi"):
                        if sum(after) + 1 == 13 - 3 * meld_n_a + 1:
                            # after 是 11 张（应为 10），补 1 张任意不在手的不变牌？——不行，
                            # 正确做法：after(11)=before(13)-2(碰用)-1(待打)。待打那张未知，
                            # 但向听对「多出的那张」取 min ⇒ 直接对 11 张里每张试丢，取最优。
                            best = None
                            for tt in range(tiles.TILE_KINDS):
                                if after[tt] <= 0:
                                    continue
                                after[tt] -= 1
                                try:
                                    v = sh_mod.shanten(list(after), meld_n_a, memo=memo)
                                except Exception:  # noqa: BLE001
                                    v = None
                                after[tt] += 1
                                if v is not None and (best is None or v < best):
                                    best = v
                            s_after = best
                        else:
                            try:
                                s_after = sh_mod.shanten(list(after), meld_n_a, memo=memo)
                            except Exception:  # noqa: BLE001
                                s_after = None
                    else:  # gang：replay 已移除整组，手牌恰为 3m+1
                        try:
                            s_after = sh_mod.shanten(list(after), meld_n_a, memo=memo)
                        except Exception:  # noqa: BLE001
                            s_after = None
                except Exception:  # noqa: BLE001
                    s_after = None
                if is_target and who and s_before is not None:
                    n = draw_count[seat]
                    tb = "1-4" if n <= 4 else ("5-8" if n <= 8 else ("9-12" if n <= 12 else "13+"))
                    sb = str(s_before) if s_before <= 3 else "4+"
                    tags = tile_tags(tile) if isinstance(tile, int) else ["unknown"]
                    if t == "peng":
                        # 对子碰第 3 组：before 中该牌 ≥3（含财神？不，财神不可碰财神的判法复杂，按 raw 计数）
                        if isinstance(tile, int) and before[tile] >= 3:
                            tags.append("pair3rd")
                    elif t == "chi":
                        mt = data.get("tiles") or []
                        tags.append("chi_" + chi_shape(tile, mt))
                    elif t == "gang":
                        tags.append("gang_" + str(data.get("kind", "?")))
                    rn = round_no + 1  # 当前局号（round_ended 的 round_no 从 1 起，事件在其后）
                    win_seat, fan = round_result.get(rn, (None, 0))
                    won = 1 if win_seat == seat else 0
                    drop = 1 if (s_after is not None and s_after < s_before) else 0
                    for tag in tags:
                        a = agg[(who, tb, sb, tag)]
                        a[0] += 1
                        a[1] += won
                        a[2] += fan if won else 0
                        a[3] += drop
                continue
            if t == "round_ended":
                round_no = (ev.get("data") or {}).get("round_no", round_no + 1)
            try:
                replay.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                break

    payload = {
        "rooms": rooms,
        "meld_agg": {"||".join(str(x) for x in k): v for k, v in agg.items()},
        "draw_agg": {"||".join(str(x) for x in k): v for k, v in draw_agg.items()},
    }
    tmp = chunk_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(chunk_path)
    return rooms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="1a bot 副露条件分布（分块幂等）")
    ap.add_argument("--rooms", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=500)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    # name -> uid（与 topbot_gap_analysis.py 同口径：首次出现）
    name_to_uid: dict[str, str] = {}
    for fpath in files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[str(n)] = str(u)
        except Exception:  # noqa: BLE001
            continue
    missing = [n for n in TOP_BOTS if n not in name_to_uid]
    if missing:
        print(f"未命中 bot: {missing}", flush=True)
    uid_to_name = {name_to_uid[n]: n for n in TOP_BOTS if n in name_to_uid}
    print(f"目标 bot: {len(uid_to_name)}/{len(TOP_BOTS)}", flush=True)

    if args.rooms:
        step = max(1, len(files) // args.rooms)
        files = files[::step][: args.rooms]

    chunk_dir = ROOT / "agent" / "out" / f"meldcond-chunks/cs{args.chunk_size}-n{len(files)}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    print(f"事件流 {len(files)} 房 / {n_chunks} 块 -> {chunk_dir}", flush=True)

    for ci in range(n_chunks):
        chunk_path = chunk_dir / f"chunk-{ci:04d}.json"
        if chunk_path.exists():
            continue
        batch = files[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        rooms = process_batch(batch, chunk_path, uid_to_name)
        print(f"chunk {ci:04d}: {rooms} rooms -> {chunk_path}", flush=True)

    # 合并
    meld: dict = collections.defaultdict(lambda: [0, 0, 0, 0])
    draw: dict = collections.defaultdict(int)
    total_rooms = 0
    for cp in sorted(chunk_dir.glob("chunk-*.json")):
        d = json.loads(cp.read_text(encoding="utf-8"))
        total_rooms += d["rooms"]
        for k, v in d["meld_agg"].items():
            m = meld[k]
            for i in range(4):
                m[i] += v[i]
        for k, v in d["draw_agg"].items():
            draw[k] += v

    print(f"\nrooms={total_rooms}")
    print("\n=== 副露事件聚合（who × 巡目 × 副露前向听 × 牌型）===")
    print(f"{'who':<14} {'巡目':>4} {'向听':>4} {'牌型':<12} {'副露n':>6} {'胡率':>6} {'均番':>5} {'降向听%':>7}")
    for k in sorted(meld):
        who, tb, sb, tag = k.split("||")
        n, w, fs, dr = meld[k]
        if not n:
            continue
        print(f"{who:<14} {tb:>4} {sb:>4} {tag:<12} {n:>6} {w/n*100:>5.1f}% {fs/w if w else 0:>5.2f} {dr/n*100:>6.1f}%")

    print("\n=== 机会分母（who × 巡目 × 摸牌时向听 → 摸牌数）===")
    print(f"{'who':<14} {'巡目':>4} {'向听':>4} {'摸牌n':>7}")
    for k in sorted(draw):
        who, tb, sb = k.split("||")
        print(f"{who:<14} {tb:>4} {sb:>4} {draw[k]:>7}")

    (chunk_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
