#!/usr/bin/env python
"""预计算基座模型在微调集上的 6 类动作分布（供 v2 训练的 KL 锚定使用）。

动机：WSL2 实测——训练循环内穿插 no_grad 推理会触发 CUDA 死锁
（日志冻结 + 100% CPU 自旋）。把基座前向移到独立的纯推理进程，
训练进程只读预计算结果，彻底消除混合负载。

输出: data/finetune_v2_base_probs.npy  (N,6) float32，与 finetune_indices_v2.npy 对齐
"""
import sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nnrl2.model_v7 import MahjongTransformerV7

OBS_KEY_MAP = {
    "obs_hand": "hand", "obs_discards": "discards", "obs_melds": "melds",
    "obs_action_history": "action_history", "obs_scores": "scores",
    "obs_god": "god", "obs_wall_remaining": "wall_remaining",
    "obs_turn": "turn", "obs_phase": "phase", "obs_target": "target",
    "obs_seat": "seat", "obs_dealer": "dealer",
}


def main():
    device = "cuda"
    indices = np.load("data/finetune_indices_v2.npy")
    n = len(indices)
    print(f"微调索引: {n} 条", flush=True)

    ckpt = torch.load("runs/bc_v7_base.pt", map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = MahjongTransformerV7(
        d_model=a.get("d_model", 128), nhead=a.get("nhead", 8),
        num_layers=a.get("num_layers", 4),
        dim_feedforward=a.get("dim_feedforward", 512),
        dropout=a.get("dropout", 0.1)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    d = np.load("data/audit_v56_with_model.npz", allow_pickle=True)
    obs = {}
    for nk, mk in OBS_KEY_MAP.items():
        arr = d[nk][indices]
        dt = np.float32 if arr.dtype in (np.float32, np.float64) else np.int64
        obs[mk] = torch.from_numpy(arr.astype(dt)).to(device)
    print(f"数据已加载到 {device}", flush=True)

    probs = np.zeros((n, 6), dtype=np.float32)
    bs = 1024
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            ob = {k: v[s:e] for k, v in obs.items()}
            ap, _ = model.predict_full(ob)
            probs[s:e] = ap.cpu().numpy()
            if (s // bs) % 5 == 0:
                print(f"  {e}/{n} ({time.perf_counter()-t0:.1f}s)", flush=True)
    np.save("data/finetune_v2_base_probs.npy", probs)
    print(f"已保存 data/finetune_v2_base_probs.npy shape={probs.shape} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
