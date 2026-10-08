#!/usr/bin/env python3
"""NN 训练骨架：显存预算硬上限 + 检查点续跑 + 训练记录 + 纯 Python 可推理的产物。

设计要点（对应 `openspec/changes/openclaw-deploy/tasks.md` 6.4/6.5）：

1. **显存预算硬上限**：`torch.cuda.set_per_process_memory_fraction`，默认 6000MB（本机 8151MB）。
   超预算直接抛错，不靠「小心点」。
2. **检查点续跑**：每个 epoch 落检查点（模型 + 优化器 + epoch + 最佳验证 + 指纹）。
   环境会周期性回收长跑进程（`notes/OWNERSHIP.md` 记录过），没有检查点的训练等于白跑。
3. **产物纯 Python 可推理**：权重 + 归一化参数存 JSON，推理侧照 `strategy/gbdt.py` 的模式手写前向，
   运行路径不引入 torch（`src/majiang/**` 零第三方依赖是硬约束）。
4. **种子必填**：省略即拒绝执行。
5. **特征只来自公开信息**：直接用 `features.py` 的 29 维，见 `research/FEATURES.md`。

用法::

    uv run python research/train_nn.py --train 'data/value_train.part*.npz' \\
        --valid 'data/value_valid.part*.npz' --name nn-value-v1 --seed 20260927 --epochs 20
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from fingerprint import dataset_fingerprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "research", "records")
CKPT_DIR = os.path.join(ROOT, "research", "ckpt")


def load_xy(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"拒绝执行：glob 未匹配到任何文件：{pattern}")
    xs, ys = [], []
    for path in files:
        with np.load(path) as doc:
            xs.append(doc["x"])
            ys.append(doc["y"])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None, help="产物路径（默认 research/artifacts/<name>.json）")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--max-vram-mb", type=int, default=6000)
    ap.add_argument("--ckpt", default=None, help="检查点路径（默认 research/ckpt/<name>.pt）")
    ap.add_argument("--resume", action="store_true", help="存在检查点时从检查点续跑")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vram_budget = None
    if dev == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        vram_budget = min(1.0, args.max_vram_mb * 1024 * 1024 / total)
        torch.cuda.set_per_process_memory_fraction(vram_budget)
        if not args.quiet:
            print(f"GPU {torch.cuda.get_device_properties(0).name}，"
                  f"显存上限 {args.max_vram_mb}MB（占 {vram_budget:.1%}）")

    xtr, ytr = load_xy(args.train)
    xva, yva = load_xy(args.valid)
    shards, data_fp, data_bytes = dataset_fingerprint(args.train)

    norm = {"mean": xtr.mean(axis=0).tolist(), "std": xtr.std(axis=0).tolist()}
    std = np.where(np.asarray(norm["std"]) == 0, 1.0, np.asarray(norm["std"]))

    def prep(x):
        return torch.tensor((x - np.asarray(norm["mean"])) / std, dtype=torch.float32)

    xtr_t, ytr_t = prep(xtr).to(dev), torch.tensor(ytr, dtype=torch.float32).to(dev)
    xva_t, yva_t = prep(xva).to(dev), torch.tensor(yva, dtype=torch.float32).to(dev)

    model = MLP(xtr.shape[1], args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt = args.ckpt or os.path.join(CKPT_DIR, f"{args.name}.pt")
    start_epoch, best_valid, history = 0, float("inf"), []
    if args.resume and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        if state.get("data_fingerprint") != data_fp:
            raise SystemExit(f"拒绝续跑：检查点绑定的数据指纹 {state['data_fingerprint'][:16]}… "
                             f"与当前数据 {data_fp[:16]}… 不一致（换数据必须重训）")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch, best_valid, history = state["epoch"] + 1, state["best_valid"], state["history"]
        if not args.quiet:
            print(f"从检查点续跑：epoch {start_epoch} 起，历史最佳 valid {best_valid:.4f}")

    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    n = xtr_t.shape[0]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        total_loss = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * idx.numel()
        model.eval()
        with torch.no_grad():
            valid = float(torch.nn.functional.mse_loss(model(xva_t), yva_t))
        best_valid = min(best_valid, valid)
        history.append({"epoch": epoch, "train_mse": total_loss / n, "valid_mse": valid})
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "best_valid": best_valid, "history": history, "seed": args.seed,
                    "hyperparams": vars(args), "data_fingerprint": data_fp}, ckpt)
        if not args.quiet:
            print(f"epoch {epoch}: train {total_loss / n:.4f}  valid {valid:.4f}  "
                  f"best {best_valid:.4f}")

    out = args.out or os.path.join(ROOT, "research", "artifacts", f"{args.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    weights = {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()}
    artifact = {
        "name": args.name, "kind": "mlp-mse",
        "architecture": {"n_in": int(xtr.shape[1]), "hidden": args.hidden,
                         "activation": "relu", "output": "linear"},
        "normalization": norm, "weights": weights,
        "data_fingerprint": data_fp, "seed": args.seed,
        "best_valid_mse": best_valid, "device": dev,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False)

    record = {
        "name": args.name, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hyperparams": {k: v for k, v in vars(args).items() if k not in ("resume", "quiet")},
        "seed": args.seed,
        "data": {"train_glob": args.train, "valid_glob": args.valid,
                 "shard_count": len(shards), "total_bytes": data_bytes,
                 "combined_sha256": data_fp},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda, "device": dev,
                "gpu": torch.cuda.get_device_properties(0).name if dev == "cuda" else None,
                "vram_budget_fraction": vram_budget},
        "samples": {"train": int(n), "valid": int(xva.shape[0])},
        "history": history, "best_valid_mse": best_valid,
        "artifact": out, "checkpoint": ckpt,
    }
    os.makedirs(RECORDS, exist_ok=True)
    rec_path = os.path.join(RECORDS, f"train-{args.name}.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n产物 {out}\n记录 {rec_path}\n最佳 valid MSE {best_valid:.4f}")
    print("下一步：research/compare.py 与基线同批样本对拍，未超噪声底就记「无显著增益」。")


if __name__ == "__main__":
    main()
