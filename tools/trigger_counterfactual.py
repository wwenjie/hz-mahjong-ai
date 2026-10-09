"""触发点条件对拍（P1）：在**真机**触发点上量「吃 vs 不吃」的净得分差。

**为什么需要它**（2026-10-08 22:30 A 裁定 P1）：整场 A/B 对低剂量改动没有功效——
剂量 0.69 次/场、480 场配对 se≈0.055 ⇒ MDE≈0.15 分/场 ⇒ 单点效应 ≪ 检出下限。
条件对拍把「整场噪声」换成「同一局面的两条分支」，**点数**即样本量，功效高一个量级。

原理（尽量不引入近似）：
1. 房文件 `blocks[i].start_hands` 给出该段起点**四家精确起手**；
2. 牌墙 = 该段后续**真实摸牌序列**（倒序入栈 ⇒ `pop()` 按历史顺序摸）+ 未观测余牌；
3. 从段起点用**引擎自己的函数**（`round._draw` / `apply_discard` / `apply_chi` /
   `apply_peng` / `apply_minggang`）重放到触发点**之前** ⇒ 动作链、抓打圈、副露、牌河
   全由引擎维护，不复制语义；
4. 触发点处调 `resolve_responses`：基线分支用 baseline 决策器（历史真值＝pass），
   处理分支用 treatment 决策器（该吃就吃）⇒ 两分支仅此一处不同，差分可归因；
5. 续跑到局末（`round.play_round`），比较我方本局净分。

**已知局限（写在代码里，不藏在结论里）**：段（block）若从一局中途开始，`start_hands`
只给暗手、**不给已有副露** ⇒ 这类触发点的重建不可信。`--check` 会按「与普查数据集
（由 `replay` 独立记录的我方手牌计数）不一致」把它们暴露出来，对拍默认只跑校验通过的段。

用法::
    .venv/bin/python tools/trigger_counterfactual.py --points agent/out/trigger-points/all.jsonl --check 200
    .venv/bin/python tools/trigger_counterfactual.py --points <...> --limit 400 --treatment v7-keepchi
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majiang.cli import make_decider  # noqa: E402
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import (  # noqa: E402
    ANGANG,
    BUGANG,
    CHI,
    DISCARD,
    GANG,
    Action,
    legal_actions,
)
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay  # noqa: E402
from majiang.sim import round as R  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

OUR = "u_a7f7c67bb14a"
TOTAL_TILES = tiles.TILE_KINDS * tiles.COPIES_PER_KIND
Z_ALPHA = 1.959964
Z_POWER_80 = 0.8416212


class RebuildError(RuntimeError):
    """该触发点无法重建（起始手牌缺失等）——必须显式失败，不得退化成近似。"""


def resolve_point_file(raw: str) -> Path:
    """触发点里存的 `file` 是**产出时那台机器的绝对路径**。

    跨机跑（本机普查 → 远端对拍）时该绝对路径在远端不存在 ⇒ 按 `data/` 后缀挂回本仓 ROOT。
    这是 2026-10-09 03:20 远端首跑撞到的坑（`FileNotFoundError: /home/wuwenjie01/...`）。
    """
    path = Path(raw)
    try:
        if path.exists():
            return path
    except OSError:
        # 跨机时会撞 `PermissionError: /root/...`（另一端的数据目录），
        # 在 3.12 里 `Path.exists()` 会把它抛出来而不是返回 False ⇒ 这里显式吞掉再走后缀回退。
        pass
    marker = "/data/"
    index = raw.find(marker)
    if index != -1:
        candidate = Path(__file__).resolve().parents[1] / raw[index + 1 :]
        if candidate.exists():
            return candidate
    return path


def _hands_from_start(span: object) -> list[list[int]]:
    """把某一局的起始手牌转成计数；任一座位缺失（None）即无法重建。"""
    codes_list = list(getattr(span, "start_hands", ()) or ())
    if len(codes_list) != 4 or any(codes is None for codes in codes_list):
        raise RebuildError(f"该局起始手牌不完整（{len(codes_list)} 座，含缺失）")
    hands = [[0] * tiles.TILE_KINDS for _ in range(4)]
    for seat, codes in enumerate(codes_list):
        for code in codes:
            hands[seat][tiles.parse(code)] += 1
    return hands


def _wall_for_block(events: list, start_hands: list[list[int]]) -> tuple[list[int], int]:
    """牌墙：`pop()` 依次给出本局后续的真实摸牌，其后是未观测余牌（守恒推出）。"""
    future: list[int] = []
    for event in events:
        if event.get("type") == "tile_drawn":
            tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
            if tile is not None:
                future.append(tile)
    pool = [t for t in range(tiles.TILE_KINDS) for _ in range(tiles.COPIES_PER_KIND)]
    # 注意：start_hands 是**计数数组**（不是牌列表），必须按张数展开。
    for hand_counts in start_hands:
        for tile, amount in enumerate(hand_counts):
            for _ in range(amount):
                try:
                    pool.remove(tile)
                except ValueError:
                    pass
    for tile in future:
        try:
            pool.remove(tile)
        except ValueError:
            pass
    return pool + list(reversed(future)), len(pool)


def build_state(doc: dict, round_index: int) -> tuple[R.RoundState, list]:
    """按**局**重建：起始手牌与事件列表取自 `replay.round_spans`（与普查同一切分）。"""
    spans = replay.round_spans(doc)
    if not 0 <= round_index < len(spans):
        raise RebuildError(f"局序号 {round_index} 越界（共 {len(spans)} 局）")
    span = spans[round_index]
    start_hands = _hands_from_start(span)
    events = list(span.events)
    wall, _unknown = _wall_for_block(events, start_hands)
    state = R.RoundState(
        wall=wall,
        seats=[R.Seat(hand=list(hand)) for hand in start_hands],
        dealer=int(getattr(span, "dealer", 0) or 0),
        round_no=int(getattr(span, "round_no", 1) or 1),
        turn=int(getattr(span, "dealer", 0) or 0),
    )
    return state, events


def drive(state: R.RoundState, events: list, upto: int) -> tuple[collections.Counter, int | None]:
    """用引擎函数重放 ``events[:upto]``；返回（异常计数, 最后摸到的牌）。"""
    anomalies: collections.Counter = collections.Counter()
    drawn: int | None = None
    discarder: int | None = None
    for event in events[:upto]:
        kind = event.get("type")
        seat = event.get("seat")
        tile = replay._tile_of(event.get("tile"))  # noqa: SLF001
        data = event.get("data") or {}
        if kind == "tile_drawn" and isinstance(seat, int):
            got = R._draw(state, seat)  # noqa: SLF001
            if got != tile:
                anomalies["摸牌不一致"] += 1
            drawn = got
            state.turn = seat
        elif kind == "tile_discarded" and isinstance(seat, int) and tile is not None:
            R.apply_discard(state, seat, tile, drawn)
            drawn = None
            discarder = seat
        elif kind == "chi" and isinstance(seat, int) and tile is not None and discarder is not None:
            run = [
                t for t in (replay._tile_of(c) for c in (data.get("tiles") or ()))  # noqa: SLF001
                if t is not None
            ]
            used = [t for t in run if t != tile]
            if len(used) != 2:
                anomalies["chi缺tiles"] += 1
            else:
                try:
                    R.apply_chi(state, seat, discarder, tile, Action(CHI, tile=tile, tiles=(used[0], used[1])))
                except Exception:  # noqa: BLE001
                    anomalies["chi失败"] += 1
        elif kind == "peng" and isinstance(seat, int) and tile is not None and discarder is not None:
            try:
                R.apply_peng(state, seat, discarder, tile)
            except Exception:  # noqa: BLE001
                anomalies["peng失败"] += 1
        elif kind in ("gang", "minggang", "ming_gang") and isinstance(seat, int) and tile is not None:
            # **必须看 `data.kind`**（事件流实际取值：`ming`/`bu`/`an`）。
            # 2026-10-09 12:20 B' 报的「加杠(bugang)重建 bug」根因就在这里：旧代码把补杠当作
            # 明杠或新杠处理 ⇒ 既有的碰**没有原地升级**成杠 ⇒ 同一牌种出现 7 张 ⇒ 守恒失败，
            # 约 6–7% 的点无法保真重建（被自动剔除、损失剂量）。
            gkind = str(data.get("kind") or "")
            try:
                if gkind in ("bu", "bugang"):
                    R.apply_gang(state, seat, Action(GANG, tile=tile, gang_kind=BUGANG))
                elif gkind in ("an", "angang"):
                    R.apply_gang(state, seat, Action(GANG, tile=tile, gang_kind=ANGANG))
                elif gkind == "ming" or discarder is not None:
                    R.apply_minggang(state, seat, discarder if discarder is not None else 0, tile)
                else:
                    anomalies["gang未知kind"] += 1
            except Exception:  # noqa: BLE001
                anomalies["gang失败"] += 1
    return anomalies, drawn


def conservation(state: R.RoundState) -> int:
    """Σ(手牌+牌河+副露) + 牌墙 —— 应为 136。"""
    total = len(state.wall)
    for seat in state.seats:
        total += sum(seat.hand) + len(seat.discards) + sum(len(m.tiles) for m in seat.melds)
    return total


def rebuild(doc: dict, point: dict) -> tuple[R.RoundState, list, collections.Counter, int | None]:
    """按点重建局面。`point["block"]` 是**局序号**（普查用 `iter_rounds` 的切分）。"""
    state, events = build_state(doc, point["block"])
    if int(getattr(replay.round_spans(doc)[point["block"]], "round_no", 0)) != int(point["round_no"]):
        raise RebuildError("局序号与实际 round_no 不符（切分漂移）")
    anomalies, drawn = drive(state, events, point["ev_index"])
    return state, events, anomalies, drawn


def check(points: list[dict], limit: int) -> None:
    """保真校验：守恒 + 与普查数据集里 `replay` 独立记录的我方手牌交叉验证。"""
    bad_cons = bad_hand = unbuildable = 0
    for point in points[:limit]:
        try:
            doc = json.loads(resolve_point_file(point["file"]).read_text(encoding="utf-8"))
        except OSError:
            unbuildable += 1
            continue
        try:
            state, _events, anomalies, _drawn = rebuild(doc, point)
        except RebuildError as error:
            unbuildable += 1
            if unbuildable <= 3:
                print(f"  不可重建 {point['room']} 局{point['block']}: {error}")
            continue
        ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
        mine = ids.index(OUR) if OUR in ids else None
        if conservation(state) != TOTAL_TILES:
            bad_cons += 1
            if bad_cons <= 3:
                print(f"  守恒不符 {point['room']} 局{point['block']} ev{point['ev_index']}:"
                      f" {conservation(state)} != {TOTAL_TILES}")
        if mine is not None and list(state.seats[mine].hand) != list(point["hand_counts"]):
            bad_hand += 1
            if bad_hand <= 3:
                print(f"  手牌不符 {point['room']} 局{point['block']} ev{point['ev_index']}")
        if anomalies:
            print(f"  重放异常 {point['room']} 局{point['block']}: {dict(anomalies)}")
    total = min(limit, len(points))
    print(f"校验 {total} 个触发点：不可重建 {unbuildable}（{unbuildable / max(1, total):.1%}），"
          f"守恒不符 {bad_cons}，手牌不符 {bad_hand}"
          f" ⇒ {'**通过**' if bad_cons == 0 and bad_hand == 0 else '**部分不可信，只取通过者**'}")


def run_one(doc: dict, point: dict, deciders: dict, mine: int, mode: str = "response",
            force_tile: bool = False, force_trigger: bool = False) -> dict:
    """两分支各重建一次状态，从触发点续跑到局末。

    **两种模式**：
    - `response`（默认，吃法类）：触发点＝他家弃牌、我方响应窗口；分支差在「吃不吃/吃哪张」。
    - `discard`（出牌类）：触发点＝**我方自己的弃牌**；分支差在「拆不拆对子」等。

    `force_tile=True`（仅 discard 模式）：**treatment 分支强制打 `point["arm_tile"]`**，
    而不是让处理臂自己决策——用于「早听窄 vs 晚听宽」这类**反事实强制**对照
    （用户 2026-10-09 10:11 的机制缺口：v5 按构造不做跨向听权衡）。

    **任何单点异常都必须被收敛成 `ok=False`**：全量数万点上总有个别重建/引擎边界
    （实测：有单点续跑抛 `ShantenError: 暗手牌张数应为 13，实际 14`），
    不能让一个坏点把整个进程池打掉。
    """
    outcomes = {}
    for branch in ("baseline", "treatment"):
        try:
            state, events, anomalies, drawn = rebuild(doc, point)
            prev = events[point["ev_index"]]
            if prev.get("type") != "tile_discarded":
                return {"ok": False, "why": "触发事件不是弃牌"}
            ours = deciders[branch]
            pick = [ours if s == mine else deciders["opponents"] for s in range(4)]
            if mode == "discard":
                if int(prev.get("seat", -1)) != mine:
                    return {"ok": False, "why": "触发事件不是我方弃牌"}
                state.turn = mine  # PHASE_DRAW 的 legal_actions 要求 turn == seat
                sit = R.situation_for(state, mine, PHASE_DRAW, drawn=drawn)
                legal = tuple(legal_actions(sit))
                if force_tile and branch == "treatment":
                    wanted = int(point.get("arm_tile", -1))
                    chosen = next(
                        (a for a in legal if a.kind == DISCARD and a.tile == wanted), None
                    )
                    if chosen is None:
                        return {"ok": False, "why": "强制牌不在合法候选内"}
                else:
                    chosen = ours.choose(sit, legal, budget_ms=1000)
                    if chosen is None or chosen.kind != DISCARD:
                        return {"ok": False, "why": "决策器没给出弃牌"}
                R.apply_discard(state, mine, chosen.tile, drawn)
                claim = R.resolve_responses(state, mine, chosen.tile, pick)
                if claim is None:
                    nxt, need_draw = (mine + 1) % 4, True
                else:
                    nxt, need_draw = claim
                outcome = R.play_round(state, pick, current=nxt, drawn=None, need_draw=need_draw)
                flag = int(point["hand_counts"][chosen.tile]) >= 2  # 这次出牌是否拆掉一个对子
                label = "breaks_pair"
            else:
                discarder = int(prev["seat"])
                # 吃牌窗口只给下家（引擎 `resolve_responses` 的 `chi_seat=(discarder+1)%4`）。
                # 普查按「我方有 ≥2 种吃法」抽、未筛下家身份 ⇒ 这里必须排除，否则是假触发点。
                if (discarder + 1) % 4 != mine:
                    return {"ok": False, "why": "我方不是吃牌窗口（非下家）"}
                offered = point["offered"]
                R.apply_discard(state, discarder, offered, None)
                if force_trigger and branch == "treatment":
                    # **定向臂**：只在**这一次响应窗口**用处理臂裁决，之后一律用基线决策器
                    # ⇒ 差分可干净归因到「这一次吃」。
                    trig_pick = [
                        deciders["treatment"] if s == mine else deciders["opponents"]
                        for s in range(4)
                    ]
                    claim = R.resolve_responses(state, discarder, offered, trig_pick)
                    pick = [
                        deciders["baseline"] if s == mine else deciders["opponents"]
                        for s in range(4)
                    ]
                else:
                    claim = R.resolve_responses(state, discarder, offered, pick)
                if claim is None:
                    nxt, need_draw = (discarder + 1) % 4, True
                else:
                    nxt, need_draw = claim
                outcome = R.play_round(state, pick, current=nxt, drawn=None, need_draw=need_draw)
                flag = claim is not None and claim[0] == mine
                label = "took_chi"
        except RebuildError as error:
            return {"ok": False, "why": f"不可重建: {error}"}
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "why": f"异常:{type(error).__name__}"}
        outcomes[branch] = {
            "score": outcome.scores[mine],
            "win": outcome.winner == mine,
            "flow": outcome.is_flow,
            "conservation_ok": conservation(state) == TOTAL_TILES,
            "anomalies": dict(anomalies),
            label: flag,
            "picked": chosen.tile if mode == "discard" else None,
        }
    base_flag = outcomes["baseline"][label]
    treat_flag = outcomes["treatment"][label]
    # **两种模式的「触发」定义不同，必须分开写**（2026-10-09 13:40 自纠）：
    # - `response`：基线（v5）没吃、处理臂吃了 ⇒ 触发（「处理分支新做了动作」）。
    #   旧代码在加 discard 模式时把它统一成 `base and not treat`，**静默反转了响应模式语义**
    #   ⇒ `equalchi` 那轮报「确实触发 0 / 反向 1654」（那 1,654 个其实就是触发点）。
    # - `discard`：两分支**实际打的牌不同**（`force_tile` 时最忠实）。
    if mode == "discard":
        triggered = outcomes["baseline"]["picked"] != outcomes["treatment"]["picked"]
    else:
        triggered = treat_flag and not base_flag
    return {
        "ok": True,
        "diff": outcomes["treatment"]["score"] - outcomes["baseline"]["score"],
        "baseline": outcomes["baseline"],
        "treatment": outcomes["treatment"],
        "triggered": triggered,
        "both_chi": base_flag and treat_flag,
        "both_pass": not base_flag and not treat_flag,
        "conserved": outcomes["baseline"]["conservation_ok"] and outcomes["treatment"]["conservation_ok"],
    }


def _build_deciders(baseline: str, treatment: str, opponents: str) -> dict:
    """三套决策器：我方基线 / 我方处理 / 其余三座（两个分支共用，保证只有我方那处不同）。"""
    return {
        "baseline": make_decider(baseline, Mode.QUALIFIER),
        "treatment": make_decider(treatment, Mode.QUALIFIER),
        "opponents": make_decider(opponents, Mode.QUALIFIER),
    }


_WORKER: dict = {}


def _init_worker(baseline: str, treatment: str, opponents: str, mode: str,
                 force_tile: bool = False, force_trigger: bool = False) -> None:
    _WORKER["deciders"] = _build_deciders(baseline, treatment, opponents)
    _WORKER["mode"] = mode
    _WORKER["force"] = force_tile
    _WORKER["force_trigger"] = force_trigger


def _mine_of(file: str) -> int:
    doc = json.loads(resolve_point_file(file).read_text(encoding="utf-8"))
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    return ids.index(OUR) if OUR in ids else -1


def _run_point(payload: tuple[str, dict]) -> dict:
    """子进程入口：每个点自己读文件、自己重建（决策器由 initializer 建好复用）。"""
    file, point = payload
    try:
        doc = json.loads(resolve_point_file(file).read_text(encoding="utf-8"))
    except OSError:
        # 跨机/增量采集时该房可能不存在（远端缺新采集的房、或本地缺远端的房）⇒
        # 只跳过该点。**不能让它把整个进程池打掉**（2026-10-09 13:20 实测踩过）。
        return {"ok": False, "why": "房文件缺失(跨机/增量)"}
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    if OUR not in ids:
        return {"ok": False, "why": "无我方座位"}
    return run_one(
        doc, point, _WORKER["deciders"], ids.index(OUR),
        _WORKER.get("mode", "response"), _WORKER.get("force", False),
        _WORKER.get("force_trigger", False),
    )


def _accumulate(
    out: dict, point: dict, stats: collections.Counter, diffs: list[float], rows: list[dict]
) -> None:
    if not out.get("ok"):
        stats[f"失败:{out.get('why')}"] += 1
        return
    if not out["conserved"]:
        stats["守恒异常(已排除)"] += 1
        return
    if out["triggered"]:
        stats["确实触发"] += 1
        diffs.append(float(out["diff"]))
    elif out["both_chi"]:
        stats["两分支同决策(基线也触发)"] += 1
    elif out["both_pass"]:
        stats["两分支同决策(都不触发)"] += 1
    else:
        stats["反向(基线不触发/处理触发)"] += 1
    # 逐点文件**自带全部分层字段**（手牌/副露/财神/向听/klass）⇒ 事后分层不必回读原局。
    rows.append({**point, **out})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="触发点条件对拍（吃 vs 不吃）")
    ap.add_argument("--points", required=True, help="普查数据集 JSONL")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--jobs", type=int, default=1, help="并行进程数")
    ap.add_argument(
        "--force-trigger",
        action="store_true",
        help="response 模式：只在触发窗口用处理臂，之后回到基线（定向臂，去掉下游污染）",
    )
    ap.add_argument(
        "--force-tile",
        action="store_true",
        help="discard 模式：treatment 分支强制打 point[arm_tile]（反事实强制）",
    )
    ap.add_argument("--mode", default="response", choices=("response", "discard"),
                    help="response=吃法类触发点；discard=出牌层（拆对子）触发点")
    ap.add_argument("--klass", default="漏吃", help="只对这些类别做对拍；空串=全部")
    ap.add_argument("--treatment", default="v7-keepchi")
    ap.add_argument("--baseline", default="v5")
    ap.add_argument("--opponents", default="v5")
    ap.add_argument("--check", type=int, default=0, help=">0 时只跑保真校验")
    ap.add_argument("--out", default="", help="逐点结果 JSONL（可选）")
    args = ap.parse_args(argv)

    all_points = [
        json.loads(line)
        for line in Path(args.points).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.klass:
        all_points = [p for p in all_points if p.get("klass") == args.klass]
    print(f"数据集 {len(all_points)} 个触发点（klass={args.klass or '全部'}）")

    if args.check:
        check(all_points, args.check)
        return 0

    deciders = _build_deciders(args.baseline, args.treatment, args.opponents)

    diffs: list[float] = []
    stats: collections.Counter = collections.Counter()
    rows: list[dict] = []
    points = all_points[: args.limit] if args.limit > 0 else all_points
    payloads = [(point["file"], point) for point in points]
    if args.jobs > 1:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(args.baseline, args.treatment, args.opponents, args.mode, args.force_tile,
                      args.force_trigger),
        ) as pool:
            results = pool.map(_run_point, payloads, chunksize=4)
            for index, (point, out) in enumerate(zip(points, results)):
                _accumulate(out, point, stats, diffs, rows)
                if (index + 1) % 100 == 0:
                    print(f"  ... {index + 1}/{len(points)}", flush=True)
    else:
        for index, (file, point) in enumerate(payloads):
            out = run_one(
                json.loads(resolve_point_file(file).read_text(encoding="utf-8")),
                point,
                deciders,
                _mine_of(file),
                args.mode,
                args.force_tile,
                args.force_trigger,
            )
            _accumulate(out, point, stats, diffs, rows)
            if (index + 1) % 50 == 0:
                print(f"  ... {index + 1}/{len(points)}", flush=True)

    if args.out and rows:
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
    if len(diffs) < 2:
        print(f"样本不足 {dict(stats)}")
        return 1
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    se = math.sqrt(variance / len(diffs))
    mde = (Z_ALPHA + Z_POWER_80) * se
    print(f"\n对拍 {len(diffs)} 点：{dict(stats)}")
    print(f"  我方本局净分差（treatment − baseline）均值 {mean:+.3f}  标准误 {se:.3f}  "
          f"t {mean / se:+.2f}  95%CI [{mean - Z_ALPHA * se:+.3f}, {mean + Z_ALPHA * se:+.3f}]  "
          f"MDE(80%) {mde:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
