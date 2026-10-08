#!/usr/bin/env python3
"""训练记录：给一份数据分片算指纹，连同超参与环境一起落盘。

为什么要有它：`notes/OWNERSHIP.md` 的教训里，四次被推翻有三次是「仪表/叙事」问题，
而不是模型问题——**没有指纹的记录无法事后判定「两次跑的是不是同一批数据」**。

用法::

    uv run python research/record.py --data 'data/value_train.part*.npz' \\
        --name value-v1 --seed 20260924 --rounds 4000 --mode qualifier

产物：``research/records/<name>.json``（可入库；数据本身在 data/ 不入库）。

指纹口径：
- 每个分片记 ``sha256`` 与字节数
- 合并指纹 = 对「分片名排序后的 sha256 列表」再哈希，与文件路径无关，只与内容有关
"""
import argparse
import json
import os
import subprocess
import sys
import time

from fingerprint import dataset_fingerprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "research", "records")


def env_info():
    info = {"python": sys.version.split()[0], "packages": {}}
    for mod in ("numpy", "scikit-learn", "torch"):
        try:
            m = __import__(mod)
            info["packages"][mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info["packages"][mod] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["gpu"] = out or None
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="数据分片 glob，例如 'data/value_train.part*.npz'")
    ap.add_argument("--name", required=True, help="记录名（决定文件名）")
    ap.add_argument("--seed", type=int, required=True, help="随机种子（必填，缺失即拒绝）")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--extra", default=None, help="额外超参 JSON 串")
    args = ap.parse_args()

    if args.seed is None:                      # argparse 已强制，这里只是双保险
        print("拒绝执行：必须显式传入 --seed（否则结论失去可比性）", file=sys.stderr)
        sys.exit(2)

    shards, combined, total_bytes = dataset_fingerprint(args.data)
    if not shards:
        print(f"拒绝执行：glob 未匹配到任何文件：{args.data}", file=sys.stderr)
        sys.exit(2)

    record = {
        "name": args.name,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data": {"glob": args.data, "shard_count": len(shards),
                 "total_bytes": total_bytes,
                 "combined_sha256": combined, "shards": shards},
        "params": {"seed": args.seed, "rounds": args.rounds, "mode": args.mode,
                   "extra": json.loads(args.extra) if args.extra else None},
        "env": env_info(),
    }

    os.makedirs(RECORDS, exist_ok=True)
    out = os.path.join(RECORDS, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"记录已写入 {out}")
    print(f"  分片 {len(shards)} 个，共 {record['data']['total_bytes']} 字节")
    print(f"  合并指纹 {combined}")
    print(f"  种子 {args.seed} / 局数 {args.rounds} / 模式 {args.mode}")
    print(f"  环境 python {record['env']['python']}, "
          f"numpy={record['env']['packages']['numpy']}, "
          f"torch={record['env']['packages']['torch']}, gpu={record['env']['gpu']}")


if __name__ == "__main__":
    main()
