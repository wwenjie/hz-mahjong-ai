#!/usr/bin/env python
"""botlike 向量级保真度 diff（A 2026-10-06 03:38 [待C] 交办）：同一进程双构造器，逐点 diff 34 维向量。

**规格**（A 03:38③，照抄）：
- 在同样的 10~20 个房上，**同一批 `Situation`** 分别过
  (a) `stage_a_dataset_export.py` 的点构造器 `candidate_rows`
  (b) `BotLikeDecider.feature_rows`
- **逐点 diff 34 维**（29 维局面特征 + 5 候选字段），分开报「抛错数 / 缺候选数 / 维度不符数 / 不一致维度索引」
- **判据**：不一致率 **= 0**（否则就是 train-serve skew，`botlike` 作 `field` 的读数不可信）

**为什么由 C 做更可靠**（A 03:38③）：双构造器在**同一次遍历**里跑（不经过「数据集存向量、不存局面」障碍），
也不受 replay 时序坑（第一张牌之前 situation_for 抛错）影响——本探针只扫 `tile_drawn` 之后的我方摸牌点，
与训练底座的事件口径一致。

**额外口径说明（C 自拟，A 判读）**：
导出器的 `wait_copies/ukeire_exact/wait_kinds` **只在顶层 shanten 并列的候选上计算**（其余为 None）；
`BotLikeDecider.feature_rows` **无条件计算**。Stage B 训练侧 `featurize` 是 `None → 0.0`。
⇒ 本探针 diff 时把导出器侧的 None 按**训练口径**映射为 0.0 再比（这样比的就是「喂给模型的向量」是否逐位一致）。

产物：`agent/out/botlike-fidelity-diff.log`。

用法：`.venv/bin/python agent/verify/botlike_fidelity_diff.py [--rooms 20] [--seed 42]`
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
from majiang.rules import tiles  # noqa: E402
from majiang.rules.action import DISCARD, HU, legal_actions  # noqa: E402
from majiang.rules.situation import PHASE_DRAW  # noqa: E402
from majiang.sim import replay as R  # noqa: E402
from majiang.strategy import versions  # noqa: E402
from majiang.strategy.botlike import BotLikeDecider  # noqa: E402
from majiang.strategy.policy import Mode  # noqa: E402

import stage_a_dataset_export as EXP  # noqa: E402

OUR = "u_a7f7c67bb14a"
TOP_BOTS = EXP.TOP_BOTS  # 与底座同名单（**昵称**，需 name→uid 映射）


def build_targets(files):
    """照底座 1b v2 口径：先扫一遍文件建 name→uid 映射，再取 TOP_BOTS 的 uid 集。"""
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
    return {name_to_uid[n]: n for n in TOP_BOTS if n in name_to_uid}


def process_room(path: str, decider_v6, botlike, stats, targets, max_points_per_room: int = 12):
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
            # 只在「目标 bot 的摸牌点」构造 Situation（与底座口径一致：决策点 = bot 出牌点）
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
            # 先 apply 摸牌（手牌变 14 张），与底座同
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
            # === (a) 导出器构造器（底座口径）===
            visible = shanten_mod.visible_counts(
                sit.hand.counts,
                [meld.tiles for meld in sit.all_melds],
                sit.discards,
            )
            bot_tile = tile  # 此处只为构造候选行；is_bot 标记不影响向量 diff
            v5_tile = -1
            exp_rows, exp_err = EXP.candidate_rows(decider_v6, sit, bot_tile, v5_tile, visible)
            for k, v in exp_err.items():
                stats[f"exp_{k}"] += v
            # === (b) botlike 构造器 ===
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
            # 建立 tile → botlike 行 的索引
            bl_by_tile = {}
            for a, row in zip(bl_kept, bl_rows):
                bl_by_tile[a.tile] = row
            point_ok = True
            for er in exp_rows:
                t = er["tile"]
                br = bl_by_tile.get(t)
                if br is None:
                    stats["missing_candidate"] += 1
                    point_ok = False
                    continue
                # 导出器 5 个候选字段（None → 0.0，训练口径）
                exp_vec = list(sit_features(sit)) + [
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
            stats["points_total"] += 1
            if not point_ok:
                stats["points_mismatch"] += 1
            pts_in_room += 1


def sit_features(sit):
    """局面 29 维特征（与两构造器共用同一函数）。

    **不许用 id(sit) 做缓存 key**（coordinator 2026-10-06 04:05 修）：Python 会复用已回收对象
    的内存地址，跨点串点 ⇒ dim 0-28 大面积假不一致（96 点 92 点 mismatch、dim 计数 830 > 点数）。
    正确性优先：本探针每点只算一次，无需缓存。
    """
    from majiang.strategy import features

    return list(features.extract(sit))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="botlike 向量级保真度 diff")
    ap.add_argument("--rooms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "agent" / "out" / "botlike-fidelity-diff.log"))
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "data/auto_sessions/*/events/*.json")))
    rng = random.Random(args.seed)
    files = rng.sample(files, min(args.rooms, len(files)))

    decider_v6 = versions.build("v6", Mode.QUALIFIER)
    botlike = BotLikeDecider()
    stats = collections.Counter()
    targets = build_targets(files)

    print(f"rooms={len(files)} seed={args.seed} 目标 bot: {len(targets)}/{len(TOP_BOTS)}", flush=True)
    for f in files:
        process_room(f, decider_v6, botlike, stats, targets)

    tot = stats["points_total"]
    mis = stats["points_mismatch"]
    rate = (mis / tot * 100) if tot else 0.0
    print(f"\n== 汇总 ==", flush=True)
    print(f"决策点总数 = {tot}", flush=True)
    print(f"不一致点数 = {mis}  ({rate:.2f}%)", flush=True)
    print(f"判据：不一致率必须 = 0 ⇒ {'✅ 过门' if mis == 0 else '❌ 未过门'}", flush=True)
    print(f"\n== 错误/跳过分布 ==", flush=True)
    for k in sorted(stats):
        if k.startswith(("dim_", "exp_", "missing", "cand_", "situation", "apply", "bad_", "no_", "iter_", "botlike_")):
            print(f"  {k} = {stats[k]}", flush=True)
    # 把不一致维度索引单列（A 03:38③ 要求）
    dim_keys = [k for k in stats if k.startswith("dim_") and k.endswith("_mismatch")]
    if dim_keys:
        print(f"\n== 不一致维度索引（前 20）==", flush=True)
        for k in sorted(dim_keys, key=lambda x: -stats[x])[:20]:
            print(f"  {k}: {stats[k]} 次", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nstats -> {out}", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
