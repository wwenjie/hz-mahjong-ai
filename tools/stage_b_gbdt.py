#!/usr/bin/env python3
"""Stage B：全量模仿学习（A 01:20 裁决立项）——GBDT 先行，候选级打分 + argmax 出牌。

规格（A 01:20③）：
- 数据：C 底座（903 房 / 67,427 点）+ 29 维 features.extract（防 train-serve skew）
- 切分：按房分组（禁逐点随机）+ val/test 分开报——切分沿用 A′ 的 seed=42，同 82.2% 直接对照
- 模型：GBDT 先（sklearn HistGradientBoostingClassifier，快、可解释、要特征重要性），欠拟合再上 MLP
- 接入形态：先独立臂 --decider ml（外壳归 A；本脚本只产模型 + 一致率报告）

设计（B' 01:22 落 THREAD 请 A 知悉）：
- 候选级样本：每个候选 = 29 维局面特征 + 5 个候选级字段（main_total/shanten/wait_copies/ukeire_exact/wait_kinds）
  标签 = is_bot（bot 是否选了这张）；推理 = 对候选面逐候选打分取 argmax
- 这样候选面信息不丢，且与 A′ 的键结构直接可比

风险门（A 01:20④，本脚本只出 test 一致率 + 特征重要性）：
- 门④.1 rollout 一致率：依赖 A 的 ml decider 外壳跑自对弈，本脚本不做
- 门④.2 空干预门：入队前用 divergence_gate 跑，本脚本不做

用法：
    uv run python tools/stage_b_gbdt.py [--data agent/out/stage-a-dataset-chunks/cs200-n2000] \
        [--out agent/out/stage-b-gbdt.txt] [--seed 42]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


CAND_FEATS = ["main_total", "shanten", "wait_copies", "ukeire_exact", "wait_kinds"]
N_SITUATION = 29


def load(path: str):
    files = sorted(glob.glob(f"{path}/chunk-*.json"))
    if not files:
        raise FileNotFoundError(f"no chunk-*.json under {path}")
    pts = []
    for f in files:
        d = json.load(open(f))
        pts.extend(d.get("points", []))
    return pts


def split_by_room(pts, seed=42, ratios=(0.6, 0.2, 0.2)):
    rooms = sorted({p["room"] for p in pts})
    rng = random.Random(seed)
    rng.shuffle(rooms)
    n = len(rooms)
    tr = set(rooms[: int(n * ratios[0])])
    va = set(rooms[int(n * ratios[0]): int(n * (ratios[0] + ratios[1]))])
    te = set(rooms[int(n * (ratios[0] + ratios[1])):])
    return ([p for p in pts if p["room"] in tr],
            [p for p in pts if p["room"] in va],
            [p for p in pts if p["room"] in te])


def featurize(p, c):
    """候选级特征向量：29 维局面 + 5 维候选字段（null→0）。"""
    sit = p.get("features") or [0.0] * N_SITUATION
    cand = [
        c.get("main_total", 0.0) or 0.0,
        c.get("shanten", 0.0) or 0.0,
        c.get("wait_copies") or 0.0,
        c.get("ukeire_exact") or 0.0,
        c.get("wait_kinds") or 0.0,
    ]
    return sit + cand


def build_xy(points):
    X, y = [], []
    for p in points:
        cands = p.get("candidates") or []
        if not any(c.get("is_bot") for c in cands):
            continue
        for c in cands:
            X.append(featurize(p, c))
            y.append(1 if c.get("is_bot") else 0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)


def agreement(points, model) -> tuple[float, float, float]:
    """逐点 top-1 一致率（argmax 打分），报 总体/非并列/并列 三栏。"""
    n = hit = nt_n = nt_hit = t_n = t_hit = 0
    for p in points:
        cands = p.get("candidates") or []
        if not cands or not any(c.get("is_bot") for c in cands):
            continue
        X = np.asarray([featurize(p, c) for c in cands], dtype=np.float32)
        scores = model.predict_proba(X)[:, 1]
        pred_tile = cands[int(np.argmax(scores))]["tile"]
        bot_tile = next(c["tile"] for c in cands if c.get("is_bot"))
        hit_flag = pred_tile == bot_tile
        n += 1
        hit += hit_flag
        mt = [c["main_total"] for c in cands]
        mx = max(mt)
        is_tie = sum(1 for v in mt if abs(v - mx) < 1e-9) > 1
        if is_tie:
            t_n += 1; t_hit += hit_flag
        else:
            nt_n += 1; nt_hit += hit_flag
    return (hit / n if n else 0.0,
            nt_hit / nt_n if nt_n else 0.0,
            t_hit / t_n if t_n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="agent/out/stage-a-dataset-chunks/cs200-n2000")
    ap.add_argument("--out", default="agent/out/stage-b-gbdt.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--max-leaf-nodes", type=int, default=63)
    ap.add_argument("--min-samples-leaf", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    pts = load(args.data)
    train_pts, val_pts, test_pts = split_by_room(pts, seed=args.seed)
    print(f"决策点: train {len(train_pts)} / val {len(val_pts)} / test {len(test_pts)}（按房分组, seed={args.seed}）", flush=True)

    X_train, y_train = build_xy(train_pts)
    print(f"候选级训练样本: {X_train.shape[0]}（特征 {X_train.shape[1]} 维 = 29 局面 + 5 候选）", flush=True)
    print(f"正例占比: {y_train.mean():.4f}", flush=True)

    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.seed,
    )
    print(f"\n训练 GBDT（max_iter={args.max_iter}, lr={args.learning_rate}, leaves={args.max_leaf_nodes}）...", flush=True)
    model.fit(X_train, y_train)
    print(f"训练完成，耗时 {time.time()-t0:.0f}s，实际迭代 {model.n_iter_}", flush=True)

    # 三栏一致率
    va, vnt, vt = agreement(val_pts, model)
    ta, tnt, tt = agreement(test_pts, model)
    # A′ v2 对照基线（同切分）：test 82.2% / 88.3% / 73.1%
    lines = [
        "=" * 64,
        "Stage B（GBDT 候选级打分）— 三栏一致率（对照 A′ v2 同切分基线）",
        "=" * 64,
        f"{'模型':<22}{'总体':>10}{'非并列':>10}{'并列':>10}",
        f"{'生产 v6（C 实测）':<22}{'81.4%':>10}{'87.6%':>10}{'72.2%':>10}",
        f"{'A′ v2 拟合（线性键）':<22}{'82.2%':>10}{'88.3%':>10}{'73.1%':>10}",
        f"{'Stage B GBDT val':<22}{va:>10.4f}{vnt:>10.4f}{vt:>10.4f}",
        f"{'Stage B GBDT test':<22}{ta:>10.4f}{tnt:>10.4f}{tt:>10.4f}",
        f"\nval-test 差: {va-ta:+.4f}（差大 = 过拟合）",
        f"判据（A 01:20④.2 前置）: test 总体 {ta:.4f}；与 A′ 线性键差 {ta-0.8222:+.4f}",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    # 特征重要性（A 01:20③：回答「29 维里哪些维度手工键漏了」）
    try:
        from sklearn.inspection import permutation_importance
        # 用 val 子集做 permutation importance（全量太贵）
        X_val, y_val = build_xy(val_pts[:3000])
        imp = permutation_importance(model, X_val, y_val, n_repeats=3, random_state=args.seed, n_jobs=4)
        feat_names = [f"sit[{i}]" for i in range(N_SITUATION)] + CAND_FEATS
        order = np.argsort(-imp.importances_mean)
        imp_lines = ["\n特征重要性 Top15（permutation, val 子集 n=3000 点）:"]
        for i in order[:15]:
            imp_lines.append(f"  {feat_names[i]:<22} {imp.importances_mean[i]:.4f} ± {imp.importances_std[i]:.4f}")
        print("\n".join(imp_lines), flush=True)
        report += "\n" + "\n".join(imp_lines)
    except Exception as e:
        print(f"特征重要性失败: {e}", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    # 保存模型（供 A 的 ml decider 外壳用）
    try:
        import joblib
        mpath = args.out.replace(".txt", ".joblib")
        joblib.dump(model, mpath)
        print(f"模型已存: {mpath}", flush=True)
    except Exception as e:
        print(f"模型保存失败: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
