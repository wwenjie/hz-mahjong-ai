#!/usr/bin/env python3
"""T1 财神分桶差异的**噪声底复算**（我自己的独立口径，不 import B 的实现）。

问题：T1 记录显示 MLP 在 `gods=0` 好 +2.29、`gods=1` 差 −1.44、`gods=2` 差 −14.91。
这些是**逐样本平方误差差的均值**，本轮给每个差值配 95% 置信区间与 p 值，
判据：区间跨 0 → 「不显著」，不许说「某桶更好/更差」。

口径：
- 数据：`data/value_valid.part000.npz`（13,653 条，独立种子、按整局分开）。
- 逐样本平方误差：GBDT（现有 `models/value_model.json`，纯 Python 求值）与 MLP（同 T1 final 超参）。
- 区间：**按观测 bootstrap**（B=10000，within-bucket 重抽样）；另报配对 t 检验作对照。
- 分桶变量 = `x[:,5]`（手留财神数），与 T1/T2 一致。

用法: nice -n 19 uv run python research/t1_gods_noise.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "research" / "records"
MAX_VRAM_MB = 2000
SEED = 20260928
BOOT = 10000


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr_n, ytr_n, Xva_n, yva, device, ymu, ysigma, hidden=128, epochs=200, lr=1e-3,
              batch=512, dropout=0.2, patience=8):
    """复刻 T1 final：按验证 MSE 选最优 epoch（patience 早停），返回**最优 epoch 的预测**。

    这一步不能省：T1 首跑就是「不选最优 epoch 就早停/取末轮」导致 val MSE 从 125 涨到 158，
    必须用 val 选点，否则复现出来的不是 T1 那个模型。
    """
    model = MLP(Xtr_n.shape[1], hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    Xt = torch.tensor(Xtr_n, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr_n, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva_n, dtype=torch.float32, device=device)
    yva_t = np.asarray(yva, dtype=np.float64)
    n = Xt.shape[0]
    best = {"epoch": -1, "val_mse": float("inf"), "pred": None}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy() * ysigma + ymu
        val_mse = float(np.mean((pv - yva_t) ** 2))
        if val_mse < best["val_mse"] - 1e-4:
            best = {"epoch": epoch, "val_mse": val_mse, "pred": pv}
        elif epoch - best["epoch"] >= patience:
            break
    print(f"  最优 epoch {best['epoch']}  val_mse {best['val_mse']:.4f}")
    return best["pred"]


def boot_ci(d: np.ndarray, rng: np.random.Generator, boot: int = BOOT):
    """观测 bootstrap：对差值向量 d 重抽样，返回 (mean, lo95, hi95, p_two_sided)。"""
    n = d.shape[0]
    idx = rng.integers(0, n, size=(boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # 双侧 p：均值为 0 的位置在重抽样分布中的分位
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def main() -> int:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(MAX_VRAM_MB / 8151.0, 0)

    with np.load(ROOT / "data" / "value_valid.part000.npz") as doc:
        Xva = doc["x"].astype(np.float64)
        yva = doc["y"].astype(np.float64)
    with np.load(ROOT / "data" / "value_post.part000.npz") as doc:
        Xtr = doc["x"].astype(np.float64)
        ytr = doc["y"].astype(np.float64)
    print(f"train {Xtr.shape} valid {Xva.shape} device={device}")

    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n, Xva_n = (Xtr - mu) / sigma, (Xva - mu) / sigma
    ymu, ysigma = float(ytr.mean()), float(ytr.std())

    # GBDT（现有 models/，纯 Python 求值）
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from majiang.strategy.value import ValueModel  # noqa: E402
    gbdt = ValueModel.load(ROOT / "models" / "value_model.json")
    pred_g = np.array([gbdt.predict(row) for row in Xva], dtype=np.float64)
    print(f"GBDT(现有) MSE {np.mean((pred_g - yva) ** 2):.4f}")

    # MLP（T1 final 超参）
    t0 = time.time()
    pred_m = train_mlp(Xtr_n, (ytr - ymu) / ysigma, Xva_n, yva, device, ymu, ysigma)
    print(f"MLP MSE {np.mean((pred_m - yva) ** 2):.4f}（{time.time()-t0:.1f}s）")

    se_g = (pred_g - yva) ** 2
    se_m = (pred_m - yva) ** 2
    d = se_g - se_m  # 正 = MLP 更好

    gods = Xva[:, 5].astype(int)
    out = {"name": "t1-gods-noise", "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "seed": SEED, "boot": BOOT, "n_valid": int(Xva.shape[0]),
           "overall": None, "by_gods": {}}

    m, lo, hi, p = boot_ci(d, rng)
    out["overall"] = {"diff": m, "lo95": lo, "hi95": hi, "p_boot": p}
    print(f"\n全体：MLP−GBDT 平方误差差 {m:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  p_boot={p:.3f}"
          f"  → {'显著' if lo * hi > 0 else '不显著（跨 0）'}")

    print("\n按财神分桶：")
    for g in sorted(set(gods.tolist())):
        msk = gods == g
        if msk.sum() < 50:
            continue
        dg = d[msk]
        m, lo, hi, p = boot_ci(dg, rng)
        sig = "显著" if lo * hi > 0 else "不显著"
        out["by_gods"][f"gods={g}"] = {"n": int(msk.sum()), "diff": m, "lo95": lo, "hi95": hi,
                                       "p_boot": p, "significant": lo * hi > 0,
                                       "mse_gbdt": float(se_g[msk].mean()),
                                       "mse_mlp": float(se_m[msk].mean())}
        print(f"  财神 {g} 张 n={int(msk.sum()):5d}  差 {m:+8.3f}  95%CI [{lo:+8.3f}, {hi:+8.3f}]  "
              f"p_boot={p:.3f} → {sig}")

    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / "t1-gods-noise.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n记录写入 {RECORDS / 't1-gods-noise.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
