"""向量级保真度 diff（保真度门·门②）：**同进程双构造器**干净路线。

**交办**：A 2026-10-06 03:38 ——「在同一个进程里跑两个构造器：同一批 `Situation` 分别过
(a) `stage_a_dataset_export.py` 的点构造器（`candidate_rows`）、(b) `BotLikeDecider.feature_rows`，
逐点 diff 34 维，并**分开报**「抛错数 / 缺候选数 / 维度不符数 / 不一致维度索引」。
判据：不一致率 **= 0**（否则就是 train-serve skew，`botlike` 作 `field` 的读数不可信）。

**为什么是同进程**：数据集（`agent/out/stage-a-dataset-chunks/cs200-n2000`）存的是**向量**、不是局面，
回放定位同一局面有 replay 时序坑（A 03:38 第一次尝试 5760 点全跳过，且三成因混计无法诊断）。
本脚本复用导出器**自己的遍历逻辑**（`process_batch` 同款的 replay 循环），在**每个决策点当场**同时调两个构造器——
不经过「数据集存向量」这个障碍，也不受 replay 定位坑影响。

**null→0 透镜**：Stage B 训练（`tools/stage_b_gbdt.py:featurize`）对 None 一律 `or 0.0`，
所以 diff 在「None 归 0 后」的 34 维空间进行（这才是训练/服务真正消费的向量）。

**用法**：
    .venv/bin/python agent/verify/botlike_vector_diff.py [--rooms N] [--seed 42]

**输出**：JSON 到 stdout（同时落 `agent/out/botlike-vector-diff.json`）：
    points / candidates / mismatch_points / mismatch_vectors / mismatch_dims
    errors.export_* / errors.serve_*（抛错、缺候选、维度不符分开计数）
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent" / "verify"))

from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay as R  # noqa: E402
from majiang.strategy import features as features_mod  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.botlike import BotLikeDecider  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

import stage_a_dataset_export as exp  # noqa: E402  (candidate_rows = 数据底座构造器)

N_SITUATION = 29
N_CAND = 5
DIM_NAMES = [f"sit[{i}]" for i in range(N_SITUATION)] + [
    "main_total", "shanten", "wait_copies", "ukeire_exact", "wait_kinds",
]


def serve_rows(botlike: BotLikeDecider, sit, visible):
    """服务侧构造器：feature_rows → 34 维（null→0 透镜在比对时统一施加）。"""
    cands = [a for a in legal_actions(sit) if a.kind == DISCARD]
    if not cands:
        return [], "no_candidates"
    rows, _kept = botlike.feature_rows(sit, cands)
    out = []
    for action, row in zip(cands, rows, strict=True):
        out.append((action.tile, row))
    return out, None


def norm(v):
    """Stage B featurize 的 null→0 透镜（`c.get(x) or 0.0` 对 None/0 都给 0.0）。"""
    if v is None:
        return 0.0
    return float(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="botlike 向量级保真度 diff（同进程双构造器）")
    ap.add_argument("--rooms", type=int, default=20, help="房数（A 交办口径 10~20；默认 20）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-6, help="逐维容差（situation 侧 round 4 位）")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    random.seed(args.seed)
    rooms = random.sample(files, min(args.rooms, len(files)))
    rooms.sort()

    v6 = versions.build("v6", Mode.QUALIFIER)  # 导出器同款 decider（供 candidate_rows 用）
    botlike = BotLikeDecider()  # 服务侧构造器（GuardedDecider 不需要——这里直接调 feature_rows）

    # name -> uid（与导出器同一套目标 bot 识别）
    name_to_uid = {}
    for fpath in rooms:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception:  # noqa: BLE001
            pass
    targets = {name_to_uid[n]: n for n in exp.TOP_BOTS if n in name_to_uid}

    errors: collections.Counter = collections.Counter()
    mismatch_dims: collections.Counter = collections.Counter()
    mismatch_examples: list[dict] = []
    n_points = n_cands = 0
    mismatch_points = mismatch_vectors = 0
    n_rooms = 0

    for path in rooms:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            errors["room_read"] += 1
            continue
        seats_meta = doc.get("seats") or []
        ids = [str(s.get("user_id", "")) for s in seats_meta]
        if len(ids) != 4:
            continue
        seat_of_uid = {}
        for si, s in enumerate(seats_meta):
            u = str(s.get("user_id", ""))
            if u in targets:
                seat_of_uid[si] = targets[u]
        if not seat_of_uid:
            continue
        n_rooms += 1
        room = Path(path).stem
        try:
            rounds = list(R.iter_rounds(doc))
        except Exception:  # noqa: BLE001
            errors["iter_rounds"] += 1
            continue
        for state, events in rounds:
            for ev in events:
                et = ev.get("type")
                seat = ev.get("seat")
                if not isinstance(seat, int) or not (0 <= seat < 4):
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et == exp.DRAWN:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if et != exp.DISCARDED or seat not in seat_of_uid:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if not state.opened:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                bot_tile = R._tile_of(ev.get("tile"))  # noqa: SLF001
                if bot_tile is None:
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                # ---- 同一 Situation，双构造器 ----
                try:
                    sit = state.situation_for(seat, phase=PHASE_DRAW, drawn=None)
                    acts = legal_actions(sit)
                    chosen = v6.choose(sit, acts, budget_ms=600)
                except Exception:  # noqa: BLE001
                    errors["situation_or_choose"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if chosen is None:
                    errors["no_choice"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                try:
                    sit_feats = [round(float(x), 4) for x in features_mod.extract(sit)]
                except Exception:  # noqa: BLE001
                    errors["export_features"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                visible = exp.shanten_mod.visible_counts(
                    sit.hand.counts,
                    [meld.tiles for meld in sit.all_melds],
                    sit.discards,
                )
                # (a) 数据底座构造器
                exp_rows, cerr = exp.candidate_rows(v6, sit, bot_tile, chosen.tile, visible)
                errors.update({f"export_{k}": v for k, v in cerr.items()})
                # (b) 服务侧构造器
                try:
                    srv_rows, srv_err = serve_rows(botlike, sit, visible)
                except Exception:  # noqa: BLE001
                    errors["serve_feature_rows"] += 1
                    srv_rows, srv_err = [], "exception"
                if srv_err:
                    errors[f"serve_{srv_err}"] += 1
                if not exp_rows:
                    errors["export_no_rows"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                if not srv_rows:
                    errors["serve_no_rows"] += 1
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                # ---- 逐点比对（tile 对齐；tile 集不合 = 维度不符/结构性错位）----
                exp_by_tile = {r["tile"]: r for r in exp_rows}
                srv_by_tile = dict(srv_rows)
                n_points += 1
                if set(exp_by_tile) != set(srv_by_tile):
                    errors["tile_set_mismatch"] += 1
                    mismatch_points += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append({
                            "room": room, "kind": "tile_set",
                            "export_only": sorted(set(exp_by_tile) - set(srv_by_tile)),
                            "serve_only": sorted(set(srv_by_tile) - set(exp_by_tile)),
                        })
                    try:
                        R.apply_event(state, ev)
                    except Exception:  # noqa: BLE001
                        break
                    continue
                point_bad = False
                for tile in sorted(exp_by_tile):
                    e, s = exp_by_tile[tile], srv_by_tile[tile]
                    n_cands += 1
                    e_vec = list(sit_feats) + [
                        e["main_total"], float(e["shanten"]),
                        e["wait_copies"], e["ukeire_exact"], e["wait_kinds"],
                    ]
                    s_vec = list(s[0:N_SITUATION]) + list(s[N_SITUATION:])
                    if len(e_vec) != N_SITUATION + N_CAND or len(s_vec) != N_SITUATION + N_CAND:
                        errors["dim_mismatch"] += 1
                        continue
                    bad_dims = []
                    for di in range(N_SITUATION + N_CAND):
                        ev_v = norm(e_vec[di])
                        sv_v = norm(s_vec[di])
                        # situation 侧导出器 round 4 位 → 容差；候选侧导出器 main_total round 6 位
                        tol = max(args.tol, 5e-5 if di < N_SITUATION else 5e-7)
                        if abs(ev_v - sv_v) > tol:
                            bad_dims.append((di, ev_v, sv_v))
                    if bad_dims:
                        point_bad = True
                        mismatch_vectors += 1
                        for di, ev_v, sv_v in bad_dims:
                            mismatch_dims[f"{di}:{DIM_NAMES[di]}"] += 1
                        if len(mismatch_examples) < 5:
                            mismatch_examples.append({
                                "room": room, "kind": "vector", "tile": tile,
                                "dims": [
                                    {"dim": DIM_NAMES[di], "export": ev_v, "serve": sv_v}
                                    for di, ev_v, sv_v in bad_dims[:8]
                                ],
                            })
                if point_bad:
                    mismatch_points += 1
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    break

    report = {
        "rooms": n_rooms,
        "target_bots": len(targets),
        "points": n_points,
        "candidate_vectors": n_cands,
        "mismatch_points": mismatch_points,
        "mismatch_vectors": mismatch_vectors,
        "mismatch_rate_vectors": (mismatch_vectors / n_cands) if n_cands else None,
        "mismatch_dims": dict(mismatch_dims.most_common()),
        "errors": dict(errors.most_common()),
        "examples": mismatch_examples,
        "verdict": "PASS" if (n_cands and mismatch_vectors == 0 and not errors.get("tile_set_mismatch")) else "FAIL",
    }
    out_path = ROOT / "agent" / "out" / "botlike-vector-diff.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"\n-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
