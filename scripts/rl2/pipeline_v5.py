#!/usr/bin/env python
"""v5 BC 后续 pipeline：合并分片 → 重训 BC → A/B 对拍。

前提：data/bc_v5_parts/train_part*.npz 已全部落盘。

用法：
    PYTHONPATH=src:/home/wuwenjie01/majiang_ai/src python scripts/pipeline_v5.py
"""

from __future__ import annotations

import glob
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
PARTS = DATA / "bc_v5_parts"

VENV_PYTHON = "/home/wuwenjie01/majiang_ai/.venv/bin/python"
PYTHONPATH = "src:/home/wuwenjie01/majiang_ai/src"


def merge_shards():
    """合并分片 → bc_v5_train.npz。"""
    files = sorted(glob.glob(str(PARTS / "train_part*.npz")))
    if not files:
        print("❌ 无分片文件")
        return False

    print(f"合并 {len(files)} 个分片...")
    ds = [np.load(f) for f in files]
    total = sum(len(d["y"]) for d in ds)
    print(f"  各分片: {[len(d['y']) for d in ds]}")
    print(f"  总样本: {total}")

    if total < 200_000:
        print(f"⚠️ 样本不足 200k（{total}），但仍继续")

    keys = [k for k in ds[0].keys() if k not in ("x_flat", "y")]
    merged = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    merged["x_flat"] = np.concatenate([d["x_flat"] for d in ds])
    merged["y"] = np.concatenate([d["y"] for d in ds])

    out = DATA / "bc_v5_train.npz"
    np.savez_compressed(out, **merged)
    print(f"✅ 合并完成 -> {out} ({total} samples)")
    return True


def gen_valid():
    """生成 valid 集（20 matches）。"""
    out = DATA / "bc_v5_valid.npz"
    if out.exists():
        d = np.load(out)
        print(f"✅ valid 已存在: {len(d['y'])} samples")
        return True

    print("生成 valid 集（20 matches）...")
    cmd = [
        VENV_PYTHON, str(ROOT / "scripts" / "gen_bc_data_v5.py"),
        "--matches", "20", "--rounds", "8",
        "--seed", "771014", "--out", str(out),
    ]
    env = {"PYTHONPATH": PYTHONPATH, "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(f"❌ valid 生成失败: {r.stderr[-200:]}")
        return False
    print(f"✅ {r.stdout.strip()}")
    return True


def train_bc():
    """重训 BC。"""
    out = RUNS / "bc_v5.pt"
    train_data = DATA / "bc_v5_train.npz"
    valid_data = DATA / "bc_v5_valid.npz"

    if not train_data.exists():
        print("❌ bc_v5_train.npz 不存在")
        return False

    print("开始 BC 重训...")
    cmd = [
        VENV_PYTHON, str(ROOT / "scripts" / "train_bc_v2.py"),
        "--train-data", str(train_data),
        "--valid-data", str(valid_data),
        "--epochs", "30", "--batch-size", "64",
        "--d-model", "128", "--nhead", "8", "--num-layers", "4",
        "--lr", "1e-3", "--out", str(out),
    ]
    env = {"PYTHONPATH": PYTHONPATH, "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0:
        print(f"❌ BC 训练失败: {r.stderr[-200:]}")
        return False
    print(f"✅ BC 训练完成 -> {out}")
    return True


def ab_test():
    """A/B 对拍：bc_v5 vs heuristic。"""
    model = RUNS / "bc_v5.pt"
    if not model.exists():
        print("❌ bc_v5.pt 不存在")
        return False

    print("开始 A/B 对拍（40 场 × 8 局）...")
    cmd = [
        VENV_PYTHON, str(ROOT / "scripts" / "run_ab_full.py"),
        "--model", str(model),
        "--matches", "40", "--rounds", "8",
        "--seed", "20261001",
    ]
    env = {"PYTHONPATH": PYTHONPATH, "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0:
        print(f"❌ A/B 对拍失败: {r.stderr[-200:]}")
        return False
    print("✅ A/B 对拍完成")
    return True


def main():
    print("=" * 60)
    print("v5 BC 后续 pipeline")
    print("=" * 60)

    steps = [
        ("合并分片", merge_shards),
        ("生成 valid 集", gen_valid),
        ("BC 重训", train_bc),
        ("A/B 对拍", ab_test),
    ]

    for name, fn in steps:
        print(f"\n--- {name} ---")
        t0 = time.time()
        ok = fn()
        dt = time.time() - t0
        if not ok:
            print(f"\n❌ {name} 失败，终止 pipeline")
            return 1
        print(f"   用时 {dt:.1f}s")

    print("\n" + "=" * 60)
    print("✅ Pipeline 全部完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
