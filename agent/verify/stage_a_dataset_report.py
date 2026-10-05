#!/usr/bin/env python
"""Stage A′ 数据底座**合并分析**（A 2026-10-05 23:05 口径）。

输入：`agent/out/stage-a-dataset-chunks/cs200-n2000/chunk-*.json`（`stage_a_dataset_export.py` 产物）。
只读、零平台请求；不依赖 numpy。

报告项（A 23:05② 裁决 + 22:15 Stage A′ 规格）：
1. 总决策点数、rooms、错误桶计数；
2. **主键并列率**（每个点 candidates 里 `main_total` 顶层并列的点占比——复核 22:45 冒烟的 36.8%）；
3. **v6 top-1 一致率基线**（agree 的总均值）——对照 1b v2 的 80.5%；
4. **★ 报告拆两栏**（A 23:05②）：非并列点 vs 并列点的 v6 一致率分列
   ——A′ 判据是 test 一致率 ≥84%，拆栏才能看清「天花板被哪一层卡住」；
5. 次级键覆盖度自检：并列点里 `wait_copies`/`ukeire_exact` 非空占比（两级键拟合的数据可行性）；
6. 有财神决策点占比（门②「以有财神桶为单位」的分母）。

用法：`.venv/bin/python agent/verify/stage_a_dataset_report.py`（等 4 分片 DONE 后跑）
"""
from __future__ import annotations

import collections
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "agent" / "out" / "stage-a-dataset-chunks" / "cs200-n2000"


def _top_tie(cands: list[dict]) -> bool:
    """该点 candidates 的 main_total 是否存在顶层并列（≥2 个候选并列最大）。"""
    if len(cands) < 2:
        return False
    top = max(c["main_total"] for c in cands)
    return sum(1 for c in cands if c["main_total"] == top) >= 2


def main() -> int:
    files = sorted(glob.glob(str(CHUNK_DIR / "chunk-*.json")))
    if not files:
        print("无 chunk，探针未出数", flush=True)
        return 1

    n_rooms = 0
    n_points = 0
    errors: collections.Counter = collections.Counter()
    agree_n = 0
    tie_n = 0                      # 主键并列点数
    tie_agree = 0                  # 并列点里 v6 一致数
    nontie_n = 0
    nontie_agree = 0               # 非并列点里 v6 一致数
    god_points = 0                 # 有财神点数
    tie_secondary_ok = 0           # 并列点里次级键（wait_copies 或 ukeire_exact）非空数
    n_candidates_total = 0

    for f in files:
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        n_rooms += d.get("rooms", 0)
        errors.update(d.get("errors", {}))
        for p in d.get("points", []):
            n_points += 1
            n_candidates_total += len(p.get("candidates", []))
            agree = 1 if p.get("agree") else 0
            agree_n += agree
            if p.get("god_n", 0) >= 1:
                god_points += 1
            if _top_tie(p.get("candidates", [])):
                tie_n += 1
                tie_agree += agree
                # 次级键覆盖：并列点里该候选面是否带 wait_copies/ukeire_exact
                if any(c.get("wait_copies") is not None or c.get("ukeire_exact") is not None
                       for c in p.get("candidates", [])):
                    tie_secondary_ok += 1
            else:
                nontie_n += 1
                nontie_agree += agree

    print(f"chunks={len(files)}  rooms={n_rooms}  决策点={n_points}  errors={dict(errors)}", flush=True)
    print(f"候选总数={n_candidates_total}  （平均 {n_candidates_total/max(n_points,1):.1f} 候选/点）", flush=True)
    print(flush=True)

    base = agree_n / max(n_points, 1)
    print(f"== v6 top-1 一致率基线 ==  {agree_n}/{n_points} = {base:.1%}  （1b v2 参考 80.5%）", flush=True)
    print(flush=True)

    tie_rate = tie_n / max(n_points, 1)
    print(f"== 主键（main_total）顶层并列率 ==  {tie_n}/{n_points} = {tie_rate:.1%}  （22:45 冒烟 36.8%）", flush=True)
    print(flush=True)

    print("== ★ 拆两栏（A 23:05②：非并列 vs 并列）==", flush=True)
    if nontie_n:
        print(f"  非并列点：{nontie_agree}/{nontie_n} = {nontie_agree/nontie_n:.1%} 一致", flush=True)
    if tie_n:
        print(f"  并列点  ：{tie_agree}/{tie_n} = {tie_agree/tie_n:.1%} 一致", flush=True)
    print(flush=True)

    print(f"== 次级键覆盖（两级键拟合可行性）==  并列点里 wait_copies/ukeire_exact 非空：{tie_secondary_ok}/{tie_n} = {tie_secondary_ok/max(tie_n,1):.1%}", flush=True)
    print(f"== 有财神决策点占比 ==  {god_points}/{n_points} = {god_points/max(n_points,1):.1%}  （门②分母）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
