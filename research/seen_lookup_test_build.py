#!/usr/bin/env python3
"""B' 20:00 review 要求：seen_lookup 用**测试集**构建后再评同选率（排除训练集泄漏）。

对照：C 19:10 报的 72.8% 是「训练集查表、训练集评估」的泄漏口径。
本脚本：同批 test 60 房、同 seed=20260929，scan 参数与 occupancy_gonogo.py 完全一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import numpy as np

from occupancy_gonogo import CPK, score, scan


def main() -> int:
    # 与主脚本一致的 test 侧：seed+1 = 20260930
    Xte, Yte, ev, ute, tite = scan(60, 150, 20260929 + 1, "base")
    print(f"test 房={ute} 行={len(Yte)} · 评估点={len(ev)}", flush=True)

    # ★ 关键差异：seen_lookup 用 Xte/Yte（测试集）构建 —— B' 认可的无泄漏口径
    seen_idx_te = np.clip(np.rint(Xte[:, 1] * CPK).astype(int), 0, CPK)
    seen_lookup_te = np.array(
        [float(Yte[seen_idx_te == k].mean()) if (seen_idx_te == k).any() else 0.0
         for k in range(CPK + 1)]
    )
    print(
        "[seen查表·test构建] E[opp|seen] = "
        + ", ".join(f"{k}:{v:.2f}" for k, v in enumerate(seen_lookup_te)),
        flush=True,
    )

    # predict_fn 不需要（seen 臂不用模型），给个恒 0 占位；hat 数字忽略
    n, sh, ss, sv, sp, sb = score(
        ev, lambda f: np.zeros(len(f)), seen_lookup=seen_lookup_te, mode="base"
    )
    print("=" * 70)
    print(f"评估点 = {n}")
    print(f"  ① 现有口径 V          与全信息 T 同选：{sv}/{n} = {sv / n:.1%}")
    print(f"  ② 比例摊派（无信息）   与全信息 T 同选：{sp}/{n} = {sp / n:.1%}")
    print(f"  ④ 行为加权（手写读牌） 与全信息 T 同选：{sb}/{n} = {sb / n:.1%}")
    print(f"  ☆ seen 查表（test构建）与全信息 T 同选：{ss}/{n} = {ss / n:.1%}")
    print(f"  对照：C 19:10 训练集构建口径 = 72.8%（泄漏，乐观偏置）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
