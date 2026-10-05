#!/usr/bin/env python
"""门②（botlike 向量保真度）同进程双构造器 diff（A 2026-10-06 03:42 [待C] 收口任务）。

**规格**（A 03:42③，照抄）：
- 用**导出器自己的取点循环**（`stage_a_dataset_export` 的 situation 构造口径），对**同一个 `Situation`** 同时得到
  (a) 导出器 `candidate_rows` 的 34 维，(b) `BotLikeDecider.feature_rows`（= `candidate_features` 公共函数）的 34 维；**逐点 diff**。
- **分开报**：抛错数 / 缺候选数 / 维度不符数 / **每个不一致维度的点数** + **第一个反例的完整 34 维**。
- 若仍有差异 ⇒ 附该点的 `melds / discards / visible` 三个原始量（A 03:41 反例 1 指向「副露后被取走的牌是否仍计入 visible」）。
- **判据**：不一致率 **= 0**。

**口径对齐说明（C 侧）**：
- 局面特征 round 精度：导出器存盘时 round 到 4 位，本探针 diff 用**原始 float**（两侧都未 round）⇒ 容差 1e-6。
- `main_total` 两侧都用 `versions.build("v6")`（A 03:40① 已修）。
- 次级字段（`wait_copies`/`ukeire_exact`/`wait_kinds`）：两侧都用**公共函数 `candidate_features`** 的条件分支（`do_tie`/`tied`/`wait_aware`）。

产物：`agent/out/botlike-fidelity-diff-v2.log`。

用法：`.venv/bin/python agent/verify/botlike_fidelity_diff_v2.py [--rooms 20] [--seed 42]`
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

from majiang.rules import shanten as shanten_mod  # noqa: E402
from majiang.rules.action import DISCARD, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay as R  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy import candidate_features as CF  # noqa: E402
from majiang.strategy.botlike import BotLikeDecider  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

import stage_a_dataset_export as EXP  # noqa: E402

OUR = "u_a7f7c67bb14a"


def build_targets(files):
    name_to_uid = {}
    for fpath in files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
            for s in d.get("seats", []):
                n, u = s.get("name"), s.get("user_id")
                if n and u and n not in name_to_uid:
                    name_to_uid[n] = u
        except Exception:  # noqa: BLE001
            pass
    return {name_to_uid[n]: n for n in EXP.TOP_BOTS if n in name_to_uid}


def process_room(path, decider_v6, botlike, stats, targets, first_counterexample, max_points_per_room=12):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stats["bad_json"] += 1
        return
    ids = [str(s.get("user_id", "")) for s in (doc.get("seats") or [])]
    if len(ids) != 4 or OUR not in ids:
        stats["no_our_seat"] += 1
        return
    bot_seats = [i for i, u in enumerate(ids) if u in targets]
    if not bot_seats:
        stats["no_target_bot"] += 1
        return
    try:
        rounds = list(R.iter_rounds(doc))
    except Exception:  # noqa: BLE001
        stats["iter_rounds"] += 1
        return
    pts_in_room = 0
    for state, events in rounds:
        if pts_in_room >= max_points_per_room:
            break
        for ev in events:
            if pts_in_room >= max_points_per_room:
                break
            etype = ev.get("type")
            seat = ev.get("seat")
            if etype != "tile_drawn" or seat not in bot_seats:
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    stats["apply_event"] += 1
                    break
                continue
            tile = R._tile_of(ev.get("tile"))  # noqa: SLF001
            if tile is None:
                try:
                    R.apply_event(state, ev)
                except Exception:  # noqa: BLE001
                    stats["apply_event"] += 1
                    break
                continue
            try:
                R.apply_event(state, ev)
            except Exception:  # noqa: BLE001
                stats["apply_event"] += 1
                break
            try:
                sit = state.situation_for(seat, phase=PHASE_DRAW, drawn=tile)
            except Exception:  # noqa: BLE001
                stats["situation"] += 1
                continue

            # === (a) 导出器原始口径（candidate_rows）===
            visible = shanten_mod.visible_counts(
                sit.hand.counts,
                [meld.tiles for meld in sit.all_melds],
                sit.discards,
            )
            exp_rows, exp_err = EXP.candidate_rows(decider_v6, sit, tile, -1, visible)
            for k, v in exp_err.items():
                stats[f"exp_{k}"] += v

            # === (b) botlike 口径（candidate_features 公共函数）===
            cands = [a for a in legal_actions(sit) if a.kind == DISCARD]
            try:
                bl_rows, bl_kept = botlike.feature_rows(sit, cands)
            except Exception:  # noqa: BLE001
                stats["botlike_feature_rows"] += 1
                continue

            # === 逐点 diff ===
            if len(exp_rows) != len(bl_rows):
                stats["cand_count_mismatch"] += 1
                continue
            bl_by_tile = {a.tile: r for a, r in zip(bl_kept, bl_rows)}
            sit_feat = list(__import__("majiang.strategy.features", fromlist=["extract"]).extract(sit))
            point_ok = True
            bad_dims = []
            for er in exp_rows:
                t = er["tile"]
                br = bl_by_tile.get(t)
                if br is None:
                    stats["missing_candidate"] += 1
                    point_ok = False
                    continue
                exp_vec = sit_feat + [
                    float(er["main_total"]),
                    float(er["shanten"]),
                    float(er["wait_copies"]) if er["wait_copies"] is not None else 0.0,
                    float(er["ukeire_exact"]) if er["ukeire_exact"] is not None else 0.0,
                    float(er["wait_kinds"]) if er["wait_kinds"] is not None else 0.0,
                ]
                if len(exp_vec) != len(br):
                    stats["dim_mismatch"] += 1
                    point_ok = False
                    continue
                for di, (x, y) in enumerate(zip(exp_vec, br)):
                    if abs(float(x) - float(y)) > 1e-6:
                        stats[f"dim_{di}_mismatch"] += 1
                        point_ok = False
                        bad_dims.append(di)
            stats["points_total"] += 1
            if not point_ok:
                stats["points_mismatch"] += 1
                # 存第一个反例的完整信息（A 03:42③ 要求）
                if not first_counterexample["saved"] and bad_dims:
                    first_counterexample["saved"] = True
                    first_counterexample["room"] = Path(path).stem
                    first_counterexample["seat"] = seat
                    first_counterexample["tile"] = tile
                    first_counterexample["bad_dims"] = bad_dims[:10]
                    first_counterexample["exp_vec_sample"] = [round(float(x), 6) for x in exp_vec]
                    first_counterexample["bl_vec_sample"] = [round(float(y), 6) for y in br]
                    # 原始量（A 03:41 反例 1 要求）
                    first_counterexample["melds"] = [m.tiles for m in sit.all_melds]
                    first_counterexample["discards_count"] = len(sit.discards)
                    first_counterexample["visible_god"] = visible[34] if len(visible) > 34 else None
                    first_counterexample["hand_counts_sum"] = sum(sit.hand.counts)
                    first_counterexample["meld_count"] = sit.hand.meld_count
            pts_in_room += 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="botlike 向量保真度 diff v2（同进程双构造器）")
    ap.add_argument("--rooms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "agent" / "out" / "botlike-fidelity-diff-v2.log"))
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    rng = random.Random(args.seed)
    files = rng.sample(files, min(args.rooms, len(files)))

    decider_v6 = versions.build("v6", Mode.QUALIFIER)
    botlike = BotLikeDecider()
    stats = collections.Counter()
    targets = build_targets(files)
    first_ce = {"saved": False}

    print(f"rooms={len(files)} seed={args.seed} 目标 bot: {len(targets)}/{len(EXP.TOP_BOTS)}", flush=True)
    for f in files:
        process_room(f, decider_v6, botlike, stats, targets, first_ce)

    tot = stats["points_total"]
    mis = stats["points_mismatch"]
    rate = (mis / tot * 100) if tot else 0.0
    print(f"\n== 汇总 ==", flush=True)
    print(f"决策点总数 = {tot}", flush=True)
    print(f"不一致点数 = {mis}  ({rate:.2f}%)", flush=True)
    print(f"判据：不一致率必须 = 0 ⇒ {'✅ 过门' if mis == 0 else '❌ 未过门'}", flush=True)

    print(f"\n== 错误/跳过分布 ==", flush=True)
    for k in sorted(stats):
        if k.startswith(("exp_", "missing", "cand_", "situation", "apply", "bad_", "no_", "iter_", "botlike_")):
            print(f"  {k} = {stats[k]}", flush=True)

    # 每个不一致维度的点数（A 03:42③ 要求）
    dim_keys = [k for k in stats if k.startswith("dim_") and k.endswith("_mismatch")]
    if dim_keys:
        print(f"\n== 每个不一致维度的点数 ==", flush=True)
        for k in sorted(dim_keys, key=lambda x: -stats[x]):
            print(f"  {k}: {stats[k]} 点", flush=True)

    # 第一个反例完整信息（A 03:42③ 要求）
    if first_ce["saved"]:
        print(f"\n== 第一个反例（{first_ce['room']} seat{first_ce['seat']} tile{first_ce['tile']}）==", flush=True)
        print(f"  不一致维度（前 10）: {first_ce['bad_dims']}", flush=True)
        print(f"  原始量: melds={first_ce['melds']} discards={first_ce['discards_count']} "
              f"hand_sum={first_ce['hand_counts_sum']} meld_count={first_ce['meld_count']}", flush=True)
        print(f"  导出器 34 维样本: {first_ce['exp_vec_sample'][:10]}...", flush=True)
        print(f"  botlike 34 维样本: {first_ce['bl_vec_sample'][:10]}...", flush=True)
        for di in first_ce["bad_dims"][:5]:
            print(f"    dim{di}: exp={first_ce['exp_vec_sample'][di]:.6f} bl={first_ce['bl_vec_sample'][di]:.6f}", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps({"stats": dict(stats), "first_counterexample": first_ce}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nstats -> {out}", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
