"""合并 `tools/run_tb_census.sh` 的分片产物，并按 `gap` 分层。

为什么分层：`gap = scores[0].total − 破平层选中者.total`。
- `gap > 0` ⇒ **真·覆盖主分**（破平层换掉了主分最高者）——本轴要测的机制；
- `gap == 0` ⇒ 两张牌主分**完全相等**，破平层只是在真平局里按进张选 —— 这是 tie-break
  的**正当作用面**，必须单独成层，否则两件事会混成一个读数。
同时补 `arm_tile` 字段（`trigger_counterfactual.py --force-tile` 读它）＝ `maxtotal_tile`。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="合并破平层普查分片（补 arm_tile + 分层）")
    ap.add_argument("--glob", default="agent/out/trigger-points/tb-shard*.jsonl")
    ap.add_argument("--out-dir", default="agent/out/trigger-points")
    args = ap.parse_args()

    root = Path(args.glob).parent.parent.parent
    shards = sorted(root.glob(Path(args.glob).name))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in shards:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"没有分片产物：{args.glob}")

    gaps: collections.Counter = collections.Counter()
    cover: list[dict] = []
    tie: list[dict] = []
    for row in rows:
        row["arm_tile"] = int(row["maxtotal_tile"])
        gap = float(row.get("gap", 0.0))
        gaps["gap>0" if gap > 0 else "gap==0"] += 1
        (cover if gap > 0 else tie).append(row)

    for name, subset in (("tb-all", rows), ("tb-cover", cover), ("tb-tie", tie)):
        target = out_dir / f"{name}.jsonl"
        with target.open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name:<10} {len(subset):>7} 行 → {target}")
    by_shanten = collections.Counter(r["klass"] for r in cover)
    print(f"  gap>0 分层：{dict(sorted(by_shanten.items()))}")
    print(f"  gap>0 幅度分布：")
    buckets: collections.Counter = collections.Counter()
    for row in cover:
        gap = float(row["gap"])
        key = "(0,0.5)" if gap < 0.5 else "[0.5,1)" if gap < 1 else "[1,2)" if gap < 2 else "≥2"
        buckets[key] += 1
    for key in ("(0,0.5)", "[0.5,1)", "[1,2)", "≥2"):
        if buckets[key]:
            print(f"    {key:<8} {buckets[key]:>7}  ({buckets[key] / len(cover):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
