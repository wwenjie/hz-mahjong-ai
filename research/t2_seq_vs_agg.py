#!/usr/bin/env python3
"""T2：**序列表示**是否给价值模型带来增益（对照 T1 的「换模型家族无用」）。

问题（T1 之后的下一步）：29 维特征里描述对手弃牌的 4 个特征只用了**张数**
（`discards_0..3 = len(discards[seat])`），弃牌的**内容与顺序**在特征层就被丢掉了。
本脚本用同一批样本做**三级消融**，把「弃牌信息」的贡献逐层分离：

| 臂 | 弃牌信息 | 说明 |
|---|---|---|
| A `agg` | 只有张数 | 现有 29 维（= 线上 GBDT 的表示） |
| B `agg+multiset` | 张数 + **是哪些牌**（袋，无序） | 新增「内容」 |
| C `agg+seq` | 张数 + 内容 + **顺序** | 新增「时序」 |

若 C > B 才有理由说「顺序有增益」；若 B ≈ C，则只有内容有增益、顺序无用。
再加两个模型家族对照臂（GBDT）以确认结论不是 MLP 独有的：

- `constant`：常数预测（下界）
- `gbdt_agg`：sklearn GBDT on 29 维（与线上同族）
- `mlp_agg`：MLP on 29 维
- `mlp_multiset` / `mlp_seq`：见上表

口径与 `tools/gen_value_data.py` 一致（「打后」局面、本人决策、标签 = 本局本人最终得分）。
训练/验证来自**不同种子**的两个生成批次，天然按整局分开、无同局泄漏。

约束：零依赖路径不受影响（本脚本只在离线研究用 torch/sklearn）；推理侧导出见 `pure_forward.py`。
CPU 谦让：`nice -n 19`、OMP/MKL 线程 1、torch 线程 1；GPU 显存硬上限。

用法::

    nice -n 19 uv run python research/t2_seq_vs_agg.py \\
        --train 'data/value_seq_train.w*.npz' --valid 'data/value_seq_valid.w*.npz'
"""
from __future__ import annotations

import argparse
import glob
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
SEATS = 4
TILE_KINDS = 34
PAD = TILE_KINDS
VOCAB = TILE_KINDS + 1
MAX_SEQ = 24


def load_glob(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"glob 未匹配到文件：{pattern}")
    xs, seqs, ys = [], [], []
    for path in files:
        with np.load(path) as doc:
            xs.append(doc["x"].astype(np.float32))
            seqs.append(doc["seq"].astype(np.int64))
            ys.append(doc["y"].astype(np.float64))
    return np.concatenate(xs), np.concatenate(seqs), np.concatenate(ys), files


def multiset_of(seq: np.ndarray) -> np.ndarray:
    """(N, 4, MAX_SEQ) 有序序列 → (N, 4*34) 无序袋（各牌种出现次数）。"""
    n = seq.shape[0]
    out = np.zeros((n, SEATS * TILE_KINDS), dtype=np.float32)
    for seat in range(SEATS):
        block = out[:, seat * TILE_KINDS:(seat + 1) * TILE_KINDS]
        for tile in range(TILE_KINDS):
            block[:, tile] = (seq[:, seat, :] == tile).sum(axis=1)
    return out


def rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩（含并列），跟 scipy.stats.rankdata 的默认口径一致，但不引依赖。"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, a.shape[0] + 1)
    # 处理并列：同值取平均秩
    sa = a[order]
    i = 0
    n = sa.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（手算：秩后 Pearson）。"""
    rp = rankdata(pred)
    ry = rankdata(y)
    rp -= rp.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rp * rp).sum() * (ry * ry).sum()))
    return float((rp * ry).sum() / denom) if denom else 0.0


def metrics(y, pred, label, log=True):
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((pred - y) ** 2) / ss) if ss else 0.0
    rho = spearman(pred, y)
    if log:
        print(f"  {label:26s} MSE {mse:9.4f}  MAE {mae:7.4f}  R2 {r2:+.4f}  Spearman {rho:+.4f}")
    return {"mse": mse, "mae": mae, "r2": r2, "spearman": rho}


class MLP(torch.nn.Module):
    """可选：把 4 家弃牌序列（共享权重编码）与聚合特征拼接。"""

    def __init__(self, n_agg: int, n_extra: int, hidden: int, hidden_seq: int = 16, dropout: float = 0.2):
        super().__init__()
        self.use_seq = n_extra == 0 and False
        self.encoder = None
        n_in = n_agg + n_extra
        if n_extra > 0:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_extra // SEATS, hidden_seq), torch.nn.ReLU(),
            )
            n_in = n_agg + hidden_seq * SEATS
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, agg, extra):
        if extra is not None and self.encoder is not None:
            # extra: (N, 4, F) → 按座位共享编码 → 拼接
            encoded = self.encoder(extra)                 # (N, 4, h)
            encoded = encoded.reshape(encoded.shape[0], -1)
            agg = torch.cat([agg, encoded], dim=1)
        return self.trunk(agg).squeeze(-1)


class SeqEncoder(torch.nn.Module):
    """对手弃牌序列编码：嵌入 → GRU（四座位共享权重）→ 末态拼接。"""

    def __init__(self, hidden: int = 16, embed: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, embed, padding_idx=PAD)
        self.gru = torch.nn.GRU(embed, hidden, batch_first=True)
        self.out = hidden

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (N*4, MAX_SEQ) —— 四家共享同一编码器
        emb = self.embed(seq)
        _, h = self.gru(emb)
        return h.squeeze(0)


class NetWithSeq(torch.nn.Module):
    def __init__(self, n_agg: int, hidden: int, hidden_seq: int = 16, dropout: float = 0.2):
        super().__init__()
        self.seq_enc = SeqEncoder(hidden_seq)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(n_agg + hidden_seq * SEATS, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, agg: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        n = seq.shape[0]
        flat = seq.reshape(n * SEATS, MAX_SEQ)
        h = self.seq_enc(flat).reshape(n, SEATS * self.seq_enc.out)
        return self.trunk(torch.cat([agg, h], dim=1)).squeeze(-1)


class DeepSetEncoder(torch.nn.Module):
    """与 SeqEncoder **同参数量级**的「无序」编码：同一 Embedding，但只做屏蔽均值池化。

    用途：把「顺序是否有贡献」从「参数化是否更好」里分离出来。
    序列臂用 Embedding+GRU（有顺序），本臂用同一 Embedding+均值池化（无顺序），
    两者对「内容」的表达能力可比，差别只在**用不用顺序**。
    """

    def __init__(self, embed: int = 8, hidden: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, embed, padding_idx=PAD)
        self.proj = torch.nn.Linear(embed, hidden)
        self.out = hidden

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        mask = (seq != PAD).float().unsqueeze(-1)          # (N*4, MAX_SEQ, 1)
        emb = self.embed(seq) * mask                        # 屏蔽 PAD
        cnt = mask.sum(dim=1).clamp(min=1.0)               # (N*4, 1)
        pooled = emb.sum(dim=1) / cnt                       # 顺序无关
        return torch.relu(self.proj(pooled))


class NetWithDeepSet(torch.nn.Module):
    def __init__(self, n_agg: int, hidden: int, hidden_seq: int = 16, dropout: float = 0.2):
        super().__init__()
        self.enc = DeepSetEncoder(hidden=hidden_seq)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(n_agg + hidden_seq * SEATS, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, agg: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        n = seq.shape[0]
        flat = seq.reshape(n * SEATS, MAX_SEQ)
        h = self.enc(flat).reshape(n, SEATS * self.enc.out)
        return self.trunk(torch.cat([agg, h], dim=1)).squeeze(-1)


def predict_batched(model, pack, device, batch=2048):
    """分批前向，避免 seq 臂（展开为 N*4 条序列过 GRU）一次性吃满显存。

    踩过的坑：T2 分析在 GRU 臂上 CUDA OOM，根因是验证一次性前向整个验证集。
    """
    model.eval()
    outs = []
    n = pack[0].shape[0]
    with torch.no_grad():
        for i in range(0, n, batch):
            part = tuple(None if t is None else t[i:i + batch] for t in pack)
            outs.append(model(*part).cpu().numpy())
    return np.concatenate(outs)


def train_torch(model, pack, pack_va, ytr, yva, ymu, ysigma, *, lr, epochs, batch, wd, device, patience):
    """通用训练循环；``pack`` 是喂给 forward 的训练张量元组。

    每个 epoch 末在验证集上算 MSE，**返回最优 epoch 的验证预测**（不是末轮）。
    这一步不能省：T1 首跑就是漏了它（末轮/过拟合），val MSE 从 125 涨到 158，
    得出完全不同的桶结论。`patience` > 0 时早停。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    n = ytr.shape[0]
    yt = torch.tensor((ytr - ymu) / ysigma, dtype=torch.float32, device=device)
    yva_arr = np.asarray(yva, dtype=np.float64)
    best = {"epoch": -1, "val_mse": float("inf"), "pred": None}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            # 包内可能含 None（如 mlp_agg 的序列占位），None 原样传递、不索引
            inputs = tuple(None if t is None else t[idx] for t in pack)
            loss = loss_fn(model(*inputs), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        pv = predict_batched(model, pack_va, device) * ysigma + ymu
        val_mse = float(np.mean((pv - yva_arr) ** 2))
        if val_mse < best["val_mse"] - 1e-4:
            best = {"epoch": epoch, "val_mse": val_mse, "pred": pv}
        elif patience and epoch - best["epoch"] >= patience:
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(ROOT / "data" / "value_seq_train.w*.npz"))
    ap.add_argument("--valid", default=str(ROOT / "data" / "value_seq_valid.w*.npz"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--name", default="t2-seq-vs-agg")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(MAX_VRAM_MB / 8151.0, 0)

    Xtr, Str, ytr, ftr = load_glob(args.train)
    Xva, Sva, yva, fva = load_glob(args.valid)
    print(f"训练 {Xtr.shape} 序列 {Str.shape} 来自 {len(ftr)} 分片")
    print(f"验证 {Xva.shape} 序列 {Sva.shape} 来自 {len(fva)} 分片")
    print(f"device={device} 线程=1\n")

    Mtr, Mva = multiset_of(Str), multiset_of(Sva)
    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n, Xva_n = (Xtr - mu) / sigma, (Xva - mu) / sigma
    ymu, ysigma = float(ytr.mean()), float(ytr.std())

    # 袋特征标准化（非负计数）
    mmu, msig = Mtr.mean(0), Mtr.std(0) + 1e-9
    Mtr_n, Mva_n = (Mtr - mmu) / msig, (Mva - mmu) / msig

    dev = torch.device(device)
    Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32, device=dev)
    Xva_t = torch.tensor(Xva_n, dtype=torch.float32, device=dev)
    Mtr_t = torch.tensor(Mtr_n, dtype=torch.float32, device=dev)
    Mva_t = torch.tensor(Mva_n, dtype=torch.float32, device=dev)
    Str_t = torch.tensor(Str, dtype=torch.long, device=dev)
    Sva_t = torch.tensor(Sva, dtype=torch.long, device=dev)

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    results["constant"] = metrics(yva, np.full_like(yva, yva.mean()), "常数")

    from sklearn.ensemble import GradientBoostingRegressor

    t0 = time.time()
    sk = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=args.seed)
    sk.fit(Xtr, ytr)
    preds["gbdt_agg"] = sk.predict(Xva)
    results["gbdt_agg"] = metrics(yva, preds["gbdt_agg"], "GBDT（29 维）")
    print(f"    （{time.time() - t0:.1f}s）")

    def run_arm(name, model, pack_tr, pack_va, label):
        t0 = time.time()
        best = train_torch(model, pack_tr, pack_va, ytr, yva, ymu, ysigma, lr=args.lr,
                           epochs=args.epochs, batch=args.batch, wd=args.wd, device=dev,
                           patience=args.patience)
        out = best["pred"]
        preds[name] = out
        results[name] = metrics(yva, out, label)
        print(f"    （{time.time() - t0:.1f}s，最优 epoch {best['epoch']}）")

    hidden = args.hidden

    # 臂 1：MLP on 29 维
    run_arm("mlp_agg", MLP(29, 0, hidden, dropout=0.2).to(dev), (Xtr_t, None), (Xva_t, None),
            "MLP（29 维）")
    # 臂 2：MLP on 29 维 + 无序袋
    run_arm("mlp_multiset", MLP(29, SEATS * TILE_KINDS, hidden, dropout=0.2).to(dev),
            (Xtr_t, Mtr_t.reshape(-1, SEATS, TILE_KINDS)), (Xva_t, Mva_t.reshape(-1, SEATS, TILE_KINDS)),
            "MLP（29 维 + 弃牌袋）")
    # 臂 3：MLP on 29 维 + 有序序列
    run_arm("mlp_seq", NetWithSeq(29, hidden, dropout=0.2).to(dev), (Xtr_t, Str_t), (Xva_t, Sva_t),
            "MLP（29 维 + 弃牌序列）")
    # 臂 4：MLP + 同参数量的**无序**编码（顺序 vs 内容的公平对照）
    run_arm("mlp_deepset", NetWithDeepSet(29, hidden, dropout=0.2).to(dev),
            (Xtr_t, Str_t), (Xva_t, Sva_t), "MLP（29 维 + 弃牌无序嵌入）")

    # 配对检验：序列 vs 袋 vs 仅聚合
    from scipy import stats

    pairs = [
        ("mlp_seq", "mlp_deepset", "序列 − 无序嵌入（顺序的贡献，参数对等）"),
        ("mlp_deepset", "mlp_agg", "无序嵌入 − 仅聚合（嵌入内容化的贡献）"),
        ("mlp_seq", "mlp_multiset", "序列 − 原始袋（参数化的贡献）"),
        ("mlp_seq", "mlp_agg", "序列 − 仅聚合（净贡献）"),
        ("mlp_agg", "gbdt_agg", "MLP − GBDT（模型家族的贡献）"),
    ]
    print("\n配对检验（正值=前者误差更小）：")
    paired = {}
    for a, b, label in pairs:
        d = (preds[b] - yva) ** 2 - (preds[a] - yva) ** 2
        t_stat, p_val = stats.ttest_1samp(d, 0.0)
        paired[f"{a}_minus_{b}"] = {"label": label, "mean": float(d.mean()),
                                    "t": float(t_stat), "p": float(p_val)}
        print(f"  {label:26s} 均值 {d.mean():+8.4f}  t {t_stat:+6.2f}  p {p_val:.3e}")

    # 按财神分桶（对齐 B 的病灶）
    gods = Xva[:, 5].astype(int)
    print("\n按手留财神数分桶（只列最能说明问题的两臂）：")
    buckets = {}
    for g in sorted(set(gods.tolist())):
        m = gods == g
        if m.sum() < 50:
            continue
        row = {"n": int(m.sum())}
        for arm in ("gbdt_agg", "mlp_agg", "mlp_seq"):
            row[f"mse_{arm}"] = float(np.mean((preds[arm][m] - yva[m]) ** 2))
        buckets[f"gods={g}"] = row
        print(f"  财神 {g} 张 n={int(m.sum()):5d}  GBDT {row['mse_gbdt_agg']:9.4f}  "
              f"MLP {row['mse_mlp_agg']:9.4f}  +序列 {row['mse_mlp_seq']:9.4f}")

    out = {
        "name": args.name,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": args.seed,
        "hyperparams": vars(args),
        "data": {"train_shards": ftr, "valid_shards": fva,
                 "n_train": int(Xtr.shape[0]), "n_valid": int(Xva.shape[0])},
        "env": {"torch": torch.__version__, "device": device},
        "results": results,
        "paired": paired,
        "by_gods": buckets,
    }
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n记录写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
