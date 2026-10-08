#!/usr/bin/env python
"""bc_v7 piao 微调 v2 验收评估：与 v1 完全同一子集口径（piao 1,285 + 听牌 16,979 + 随机对照 5,000）。

同时评估基座与 v2 微调模型，输出对比表 + 验收判定，写
outputs/v7_piao_finetune_v2_report.json。

用法:
    .venv/bin/python scripts/eval_subset_v2.py [--model runs/bc_v7_piao_finetuned_v2.pt]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nnrl2.model_v7 import MahjongTransformerV7

KMAP = {
    "obs_hand": "hand", "obs_discards": "discards", "obs_melds": "melds",
    "obs_action_history": "action_history", "obs_scores": "scores",
    "obs_god": "god", "obs_wall_remaining": "wall_remaining",
    "obs_turn": "turn", "obs_phase": "phase", "obs_target": "target",
    "obs_seat": "seat", "obs_dealer": "dealer",
}
ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = MahjongTransformerV7(
        d_model=a.get("d_model", 128), nhead=a.get("nhead", 8),
        num_layers=a.get("num_layers", 4),
        dim_feedforward=a.get("dim_feedforward", 512),
        dropout=a.get("dropout", 0.1)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def infer_subset(model, obs, cand_mask, bs=1024):
    """推理并按 candidates_mask 屏蔽非法动作后 argmax（与 audit 存储口径一致）。"""
    n = len(next(iter(obs.values())))
    chunks = []
    for s in range(0, n, bs):
        e = min(s + bs, n)
        ob = {k: v[s:e] for k, v in obs.items()}
        ap, _ = model.predict_full(ob)
        ap = ap * cand_mask[s:e] + (1.0 - cand_mask[s:e]) * (-1.0)
        chunks.append(ap.argmax(dim=1).cpu().numpy())
    return np.concatenate(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/bc_v7_base.pt")
    ap.add_argument("--model", default="runs/bc_v7_piao_finetuned_v2.pt")
    ap.add_argument("--data", default="data/audit_v56_with_model.npz")
    ap.add_argument("--piao-indices", default="data/piao_indices.npy")
    ap.add_argument("--out", default="outputs/v7_piao_finetune_v2_report.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.data, allow_pickle=True)
    piao_idx = np.load(args.piao_indices)
    shanten = d["shanten"]
    chosen = d["chosen_action_kind"].astype(np.int64)

    # === 与 v1 eval_subset.py 完全一致的子集构造（seed=42）===
    tp_idx = np.where(shanten == -1)[0]
    rng = np.random.RandomState(42)
    non_special = np.where(shanten != -1)[0]
    non_special = np.setdiff1d(non_special, piao_idx)
    rand_idx = rng.choice(non_special, size=min(5000, len(non_special)), replace=False)
    eval_idx = np.unique(np.concatenate([piao_idx, tp_idx, rand_idx]))
    print(f"评估子集: {len(eval_idx)} 条 (piao={len(piao_idx)}, "
          f"tingpai={len(tp_idx)}, random={len(rand_idx)})", flush=True)

    # 加载子集到 GPU
    t0 = time.perf_counter()
    obs = {}
    for nk, mk in KMAP.items():
        a = d[nk][eval_idx]
        dt = np.float32 if a.dtype in (np.float32, np.float64) else np.int64
        obs[mk] = torch.from_numpy(a.astype(dt)).to(device)
    cand_mask = torch.from_numpy(
        d["candidates_mask"][eval_idx].astype(np.float32)).to(device)
    print(f"GPU加载: {time.perf_counter()-t0:.1f}s "
          f"mem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # 两个模型分别推理（分批 1024，candidates 掩码口径）
    base_model = load_model(args.base, device)
    pred_base = infer_subset(base_model, obs, cand_mask)
    del base_model; torch.cuda.empty_cache()
    ft_model = load_model(args.model, device)
    pred_ft = infer_subset(ft_model, obs, cand_mask)
    del ft_model; torch.cuda.empty_cache()
    print("推理完成", flush=True)

    idx_to_pos = {v: i for i, v in enumerate(eval_idx)}
    chosen_sub = chosen[eval_idx]

    def acc(pred, mask):
        return float((pred[mask] == chosen_sub[mask]).mean() * 100) if mask.sum() else 0.0

    # piao / 听牌位置
    piao_pos = np.array([idx_to_pos[i] for i in piao_idx])
    tp_pos = np.array([idx_to_pos[i] for i in tp_idx])
    all_mask = np.ones(len(eval_idx), dtype=bool)

    rpt = {"subset_size": int(len(eval_idx)),
           "model": args.model, "base": args.base}

    def block(pred):
        return {
            "overall": round(acc(pred, all_mask), 2),
            "piao": round(acc(pred, piao_pos), 2),
            "tingpai": round(acc(pred, tp_pos), 2),
            "per_action": {},
        }

    rb, rf = block(pred_base), block(pred_ft)
    for ai, nm in enumerate(ACTION_NAMES):
        m = chosen_sub == ai
        if m.sum() == 0:
            continue
        rb["per_action"][nm] = round(acc(pred_base, m), 2)
        rf["per_action"][nm] = round(acc(pred_ft, m), 2)
        rb["per_action"][nm + "_n"] = int(m.sum())

    rpt["base"] = rb
    rpt["finetuned_v2"] = rf

    # === 验收判定 ===
    checks = {
        "piao >= 90": rf["piao"] >= 90.0,
        "tingpai >= 82 且 >= base-3pp": (
            rf["tingpai"] >= 82.0 and rf["tingpai"] >= rb["tingpai"] - 3.0),
        "chi >= 75": rf["per_action"].get("chi", 0) >= 75.0,
        "peng >= 75": rf["per_action"].get("peng", 0) >= 75.0,
        "gang >= 75": rf["per_action"].get("gang", 0) >= 75.0,
        "pass >= 93": rf["per_action"].get("pass", 0) >= 93.0,
        "overall >= 86": rf["overall"] >= 86.0,
    }
    hard_fail = (rf["per_action"].get("chi", 0) < 75.0
                 or rf["per_action"].get("peng", 0) < 75.0
                 or rf["per_action"].get("gang", 0) < 75.0
                 or rf["overall"] < 86.0)
    passed = all(checks.values())
    rpt["acceptance"] = {
        "checks": checks,
        "passed": passed,
        "hard_fail": hard_fail,
        "recommendation": (
            "PASS: v2 微调达标，可进入 A/B 评测" if passed else
            "FAIL: v2 未达标，建议转方案 A（混合架构）"),
    }

    # === 打印 ===
    print("\n=== v2 对比报告 ===")
    print(f"子集总体:  base={rb['overall']:.2f}%  ft={rf['overall']:.2f}%  "
          f"d={rf['overall']-rb['overall']:+.2f}pp")
    print(f"听牌点:    base={rb['tingpai']:.2f}%  ft={rf['tingpai']:.2f}%  "
          f"d={rf['tingpai']-rb['tingpai']:+.2f}pp")
    print(f"Piao点:    base={rb['piao']:.2f}%  ft={rf['piao']:.2f}%  "
          f"d={rf['piao']-rb['piao']:+.2f}pp")
    for nm in ACTION_NAMES:
        if nm in rb["per_action"]:
            b, f = rb["per_action"][nm], rf["per_action"][nm]
            print(f"  {nm:>8s}: base={b:.2f}%  ft={f:.2f}%  d={f-b:+.2f}pp")
    print("\n验收:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    print(f"\n结论: {rpt['acceptance']['recommendation']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rpt, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    main()
