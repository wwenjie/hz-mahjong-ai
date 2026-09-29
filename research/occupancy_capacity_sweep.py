#!/usr/bin/env python3
"""agent-c · 占用模型「容量扫描」——回答用户 17:46 的问题：
「输入维度高，是不是要多训轮次才能发挥效果？」

**要分开的两种解释**：
  (A) 欠拟合/容量不足 ⇒ 加大 max_iter / max_depth / 去掉早停，一致率应显著上升；
  (B) 信息不在这些特征里 ⇒ 怎么加容量都平，且打不过单变量 `seen` 查表。

**做法**：`scan` 一次（特征提取是瓶颈），然后在同一份 train/test 上扫一组容量配置，
报每个配置的 `n_iter_`（真实迭代数）、test MAE/r、以及五项一致率；
取最好配置与 ① 基线做 **McNemar 配对检验**（同一批评估点）。

用法::

    nice -n 19 uv run python research/occupancy_capacity_sweep.py --train-rooms 90 --test-rooms 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from occupancy_gonogo import CPK, scan, score  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rooms", type=int, default=90)
    ap.add_argument("--test-rooms", type=int, default=60)
    ap.add_argument("--cap-per-room", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--iters", type=int, nargs="+", default=[60, 250, 1000, 3000])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 6, 10])
    args = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingRegressor

    print("提取特征（一次）…", flush=True)
    Xtr, Ytr, _, utr, _ = scan(args.train_rooms, args.cap_per_room, args.seed)
    Xte, Yte, ev, ute, tite = scan(args.test_rooms, args.cap_per_room, args.seed + 1)
    print(f"  train 房={utr} 行={len(Ytr)} · test 房={ute} 行={len(Yte)} · 评估点={len(ev)}", flush=True)

    # 单变量 seen 查表基线（零学习）
    si = np.clip(np.rint(Xtr[:, 1] * CPK).astype(int), 0, CPK)
    seen_lookup = np.array(
        [float(Ytr[si == k].mean()) if (si == k).any() else 0.0 for k in range(CPK + 1)]
    )

    print("=" * 96, flush=True)
    print(f"{'max_iter':>8} {'depth':>5} {'n_iter_':>7} {'MAE':>6} {'r':>6} {'slope':>6} "
          f"{'① V':>7} {'★ hat':>7} {'☆ seen':>7}", flush=True)
    print("-" * 96, flush=True)

    detail_best = None
    best_hat = -1
    best_cfg = None
    for depth in args.depths:
        for it in args.iters:
            m = HistGradientBoostingRegressor(
                max_iter=it, learning_rate=0.08, max_depth=depth,
                random_state=args.seed, early_stopping=False,
            )
            m.fit(Xtr, Ytr)
            p = m.predict(Xte)
            mae = float(np.mean(np.abs(p - Yte)))
            r = float(np.corrcoef(p, Yte)[0, 1]) if float(np.std(p)) else float("nan")
            slope = float(np.cov(p, Yte)[0, 1] / np.var(Yte)) if float(np.var(Yte)) else float("nan")
            det: dict = {}
            n, sh, ss, sv, sp, sb = score(ev, lambda f: m.predict(f),
                                          seen_lookup=seen_lookup, detail=det)
            print(f"{it:>8} {depth:>5} {int(getattr(m, 'n_iter_', -1)):>7} {mae:>6.3f} {r:>6.3f} "
                  f"{slope:>6.3f} {sv / n:>6.1%} {sh / n:>6.1%} {ss / n:>6.1%}", flush=True)
            if sh / n > best_hat:
                best_hat = sh / n
                best_cfg = (it, depth)
                detail_best = det
    print("=" * 96, flush=True)
    print(f"最好配置 = max_iter={best_cfg[0]}, depth={best_cfg[1]} ⇒ ★ hat 一致率 {best_hat:.1%}", flush=True)

    # 配对检验：最好 hat vs ① V（同一批评估点，McNemar 精确检验）
    if detail_best:
        h = np.asarray(detail_best["hat"], dtype=bool)
        v = np.asarray(detail_best["v"], dtype=bool)
        b = int(np.sum(h & ~v))   # hat 对、V 错
        c = int(np.sum(~h & v))   # hat 错、V 对
        n_disc = b + c
        if n_disc:
            from math import comb
            k = min(b, c)
            p_two = sum(comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc) * 2
            p_two = min(1.0, p_two)
        else:
            p_two = 1.0
        print(f"McNemar（★ hat vs ① V，n={len(h)}）：hat 独对={b} · V 独对={c} · "
              f"双尾 p≈{p_two:.3f} ⇒ {'显著' if p_two < 0.05 else '不显著'}", flush=True)
        print("判读：若容量翻 50 倍后 ★ 仍打不过 ①（p 不显著），则『欠拟合』解释被否，"
              "结论指向『信息不在 34 维特征里』。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
