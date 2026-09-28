#!/usr/bin/env python3
"""T1：NN vs GBDT 价值函数对拍（同一份数据、同一套 29 维特征）。

回答的问题：把价值模型从 GBDT 换成轻量 MLP，**表示能力**上到底有没有增益？
（端到端策略增益是另一回事，由 T2/T3 与 `ab_test.py` 回答——本脚本只做回归质量对拍。）

数据：`data/value_post.part000.npz`（训练，139,669 条）/ `data/value_valid.part000.npz`
（验证，13,653 条，独立种子按整局分开）。**不重新生成数据**，因此几乎不占 CPU。

约束（照 `research/FEATURES.md` 与 `src/**` 硬约束）：
- 只用公开信息特征——直接复用 `data/` 里已生成的 29 维向量，不新增特征。
- 训练用 torch（离线，dev 组）；推理侧不引入 torch（纯 Python 前向另见 `pure_forward.py`）。
- 确定性：固定种子；显存预算硬上限。
- CPU 谦让：`OMP_NUM_THREADS=1`、torch 线程数 1、`nice -n 19`（A 的实验在跑）。

用法::

    nice -n 19 uv run python research/t1_nn_vs_gbdt.py --epochs 200
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# 必须在 import torch 之前设线程数，否则不生效
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "research" / "records"
MAX_VRAM_MB = 2000  # 本机 8151MB，A 的实验在跑，给足余量但设硬上限


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as doc:
        return doc["x"].astype(np.float64), doc["y"].astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（手算：排名后 Pearson）。"""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math_sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


def math_sqrt(v: float) -> float:
    import math

    return math.sqrt(v)


def metrics(y, pred, label):
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((pred - y) ** 2) / ss) if ss else 0.0
    rho = spearman(pred, y)
    print(f"  {label:28s} MSE {mse:9.4f}  MAE {mae:7.4f}  R2 {r2:+.4f}  Spearman {rho:+.4f}")
    return {"mse": mse, "mae": mae, "r2": r2, "spearman": rho}


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(n_in, hidden), torch.nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        layers += [
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        ]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(ROOT / "data" / "value_post.part000.npz"))
    ap.add_argument("--valid", default=str(ROOT / "data" / "value_valid.part000.npz"))
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--wd", type=float, default=0.0, help="权重衰减")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--name", default="t1-nn-vs-gbdt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(MAX_VRAM_MB / 8151.0, 0)

    Xtr, ytr = load_npz(args.train)
    Xva, yva = load_npz(args.valid)
    print(f"训练 {Xtr.shape} 目标均值 {ytr.mean():.3f} 标准差 {ytr.std():.3f}")
    print(f"验证 {Xva.shape} 目标均值 {yva.mean():.3f} 标准差 {yva.std():.3f}")
    print(f"device={device} 线程=1 nice=19\n")

    # 标准化（在训练集上拟合）
    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - mu) / sigma
    Xva_n = (Xva - mu) / sigma
    ymu, ysigma = float(ytr.mean()), float(ytr.std())

    results: dict[str, dict] = {}

    # --- 对照臂 1：常数（验证集均值）---
    results["constant"] = metrics(yva, np.full_like(yva, yva.mean()), "常数")

    # --- 对照臂 2：现有 GBDT（纯 Python 求值）---
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from majiang.strategy.value import ValueModel  # noqa: E402

    t0 = time.time()
    gbdt = ValueModel.load(ROOT / "models" / "value_model.json")
    pred_g = np.array([gbdt.predict(row) for row in Xva])
    results["gbdt_existing"] = metrics(yva, pred_g, "GBDT（现有 models/）")
    print(f"    （纯 Python 求值用时 {time.time() - t0:.1f}s）")

    # --- 对照臂 3：同数据重训的 sklearn GBDT（排除数据差异）---
    from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

    t0 = time.time()
    sk = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=args.seed
    )
    sk.fit(Xtr, ytr)
    pred_s = sk.predict(Xva)
    results["gbdt_retrained"] = metrics(yva, pred_s, "GBDT（同数据重训）")
    print(f"    （训练+预测用时 {time.time() - t0:.1f}s）")

    # --- 候选臂：MLP ---
    model = MLP(Xtr.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.MSELoss()
    Xt = torch.tensor(Xtr_n, dtype=torch.float32, device=device)
    yt = torch.tensor((ytr - ymu) / ysigma, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva_n, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    best = {"epoch": -1, "val_mse": float("inf")}
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * idx.numel()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy() * ysigma + ymu
        val_mse = float(np.mean((pv - yva) ** 2))
        history.append({"epoch": epoch, "train_mse_std": total / n, "val_mse": val_mse})
        if val_mse < best["val_mse"] - 1e-4:
            best = {"epoch": epoch, "val_mse": val_mse, "pred": pv}
        elif epoch - best["epoch"] >= args.patience:
            print(f"  早停 @ epoch {epoch}（最优 epoch {best['epoch']}）")
            break
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train {total / n:.4f}  val_mse {val_mse:.4f}")
    results["mlp"] = metrics(yva, best["pred"], f"MLP（hidden {args.hidden}）")
    print(f"    （用时 {time.time() - t0:.1f}s，最优 epoch {best['epoch']}）")

    # --- 配对检验：MLP 是否显著优于现有 GBDT（逐样本平方误差差）---
    from scipy import stats  # noqa: E402

    d = (pred_g - yva) ** 2 - (best["pred"] - yva) ** 2  # 正 = MLP 误差更小
    t_stat, p_val = stats.ttest_1samp(d, 0.0)
    print(f"\n配对（现有 GBDT − MLP 的平方误差差）：均值 {d.mean():+.4f} "
          f"t {t_stat:+.2f} p {p_val:.2e}")

    # --- 按财神数分桶（对齐 B 锁定的「恰好 1 张财神」病灶）---
    gods = Xva[:, 5].astype(int)
    buckets = {}
    print("\n按手留财神数分桶的 MSE（现有 GBDT vs MLP）：")
    for g in sorted(set(gods.tolist())):
        m = gods == g
        if m.sum() < 50:
            continue
        gm = float(np.mean((pred_g[m] - yva[m]) ** 2))
        nm = float(np.mean((best["pred"][m] - yva[m]) ** 2))
        buckets[f"gods={g}"] = {"n": int(m.sum()), "mse_gbdt": gm, "mse_mlp": nm}
        print(f"  财神 {g} 张（n={int(m.sum()):5d}）  GBDT {gm:9.4f}  MLP {nm:9.4f}  差 {gm - nm:+8.4f}")

    out = {
        "name": args.name,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": args.seed,
        "hyperparams": vars(args),
        "data": {"train": args.train, "valid": args.valid,
                 "n_train": int(Xtr.shape[0]), "n_valid": int(Xva.shape[0])},
        "env": {"torch": torch.__version__, "device": device},
        "results": results,
        "paired_gbdt_minus_mlp": {"mean": float(d.mean()), "t": float(t_stat), "p": float(p_val)},
        "by_gods": buckets,
        "history_tail": history[-5:],
    }
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n记录写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
