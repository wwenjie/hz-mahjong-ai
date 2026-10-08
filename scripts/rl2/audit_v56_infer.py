#!/usr/bin/env python
"""阶段二审计：加载 bc_v7_base，对 audit_v56.npz 全量推理（GPU batch），输出模型预测。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnrl2.model_v7 import MahjongTransformerV7

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]


def main():
    t0 = time.time()
    data_path = Path("data/audit_v56.npz")
    model_path = Path("runs/bc_v7_base.pt")
    out_path = Path("data/audit_v56_with_model.npz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = MahjongTransformerV7(
        d_model=args.get("d_model", 128),
        nhead=args.get("nhead", 8),
        num_layers=args.get("num_layers", 4),
        dim_feedforward=args.get("dim_feedforward", 512),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"model loaded: {model_path}")

    # Load data
    d = np.load(data_path, allow_pickle=True)
    n = len(d["obs_hand"])
    print(f"samples: {n}")

    key_map = {
        "obs_hand": "hand",
        "obs_discards": "discards",
        "obs_melds": "melds",
        "obs_action_history": "action_history",
        "obs_scores": "scores",
        "obs_god": "god",
        "obs_wall_remaining": "wall_remaining",
        "obs_turn": "turn",
        "obs_phase": "phase",
        "obs_target": "target",
        "obs_seat": "seat",
        "obs_dealer": "dealer",
    }

    # Output arrays
    model_top1_action = np.zeros(n, dtype=np.int8)        # masked top1
    model_top1_tile = np.zeros(n, dtype=np.int8)           # masked top1 tile
    model_prob_chosen = np.zeros(n, dtype=np.float32)       # raw a_prob of chosen action
    model_action_probs = np.zeros((n, 6), dtype=np.float32) # full raw action probs
    model_tile_prob_chosen = np.zeros(n, dtype=np.float32)  # tile prob of chosen tile (discard only)
    model_top1_action_raw = np.zeros(n, dtype=np.int8)      # raw top1 (no mask)
    model_top1_tile_raw = np.zeros(n, dtype=np.int8)        # raw top1 tile (no mask)
    model_prob_chosen_masked = np.zeros(n, dtype=np.float32) # masked a_prob of chosen action

    batch_size = 1024
    n_batches = (n + batch_size - 1) // batch_size

    for bi in range(n_batches):
        s, e = bi * batch_size, min((bi + 1) * batch_size, n)
        bs = e - s

        obs = {}
        for npz_key, model_key in key_map.items():
            arr = d[npz_key][s:e]
            t = torch.from_numpy(arr).to(device)
            if t.dtype in (torch.int8, torch.int16, torch.int32):
                t = t.long()
            elif t.dtype in (torch.float32, torch.float64):
                t = t.float()
            obs[model_key] = t

        with torch.no_grad():
            a_probs, t_probs = model.predict_full(obs)
            a_np = a_probs.cpu().numpy()
            t_np = t_probs.cpu().numpy()

        # Candidate masks
        cmask = d["candidates_mask"][s:e]       # (bs, 6) float
        tmask = d["candidates_tile_mask"][s:e]  # (bs, 34) float

        # Masked action probs (set illegal to -1 for argmax)
        a_masked = a_np * cmask + (1.0 - cmask) * (-1.0)
        top1_a_masked = np.argmax(a_masked, axis=1).astype(np.int8)

        # Masked tile probs
        t_masked = t_np * tmask + (1.0 - tmask) * (-1.0)
        top1_t_masked = np.argmax(t_masked, axis=1).astype(np.int8)

        # Raw top1
        top1_a_raw = np.argmax(a_np, axis=1).astype(np.int8)
        top1_t_raw = np.argmax(t_np, axis=1).astype(np.int8)

        chosen_a = d["chosen_action_kind"][s:e]
        chosen_t = d["chosen_tile"][s:e]

        # Raw prob of chosen action
        prob_chosen_raw = a_np[np.arange(bs), chosen_a]

        # Masked prob of chosen action: p(chosen) / sum(p * mask)
        masked_sum = (a_np * cmask).sum(axis=1)
        masked_sum = np.where(masked_sum < 1e-9, 1.0, masked_sum)
        prob_chosen_masked = prob_chosen_raw / masked_sum

        # Tile prob of chosen tile
        tile_prob_chosen = np.full(bs, -1.0, dtype=np.float32)
        disc_mask = chosen_a == 0
        if disc_mask.any():
            ct = chosen_t[disc_mask].clip(0, 33)
            tile_prob_chosen[disc_mask] = t_np[disc_mask, ct]

        model_top1_action[s:e] = top1_a_masked
        model_top1_tile[s:e] = top1_t_masked
        model_prob_chosen[s:e] = prob_chosen_raw
        model_action_probs[s:e] = a_np
        model_tile_prob_chosen[s:e] = tile_prob_chosen
        model_top1_action_raw[s:e] = top1_a_raw
        model_top1_tile_raw[s:e] = top1_t_raw
        model_prob_chosen_masked[s:e] = prob_chosen_masked

        if (bi + 1) % 200 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f"  batch {bi+1}/{n_batches} ({e}/{n}) elapsed={elapsed:.0f}s eta={eta:.0f}s")

    # Save
    out_data = {k: d[k] for k in d.keys()}
    out_data["model_top1_action"] = model_top1_action
    out_data["model_top1_tile"] = model_top1_tile
    out_data["model_prob_chosen"] = model_prob_chosen
    out_data["model_action_probs"] = model_action_probs
    out_data["model_tile_prob_chosen"] = model_tile_prob_chosen
    out_data["model_top1_action_raw"] = model_top1_action_raw
    out_data["model_top1_tile_raw"] = model_top1_tile_raw
    out_data["model_prob_chosen_masked"] = model_prob_chosen_masked

    np.savez_compressed(out_path, **out_data)
    print(f"saved: {out_path} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
