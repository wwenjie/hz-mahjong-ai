#!/usr/bin/env python3
"""Convert audit-format npz (from rebuild_situations_from_logs.py) to bc_v7 training format,
with match_id-based train/valid split to prevent data leakage.

Key mapping:
  obs_hand           -> hand           int8
  obs_discards       -> discards       int8
  obs_melds          -> melds          int8
  obs_action_history -> action_history int8
  obs_scores         -> scores         int32
  obs_god            -> god            int8
  obs_wall_remaining -> wall_remaining int32
  obs_turn           -> turn           int8
  obs_phase          -> phase          int8
  obs_target         -> target         int8
  obs_seat           -> seat           int8
  obs_dealer         -> dealer         int8 (fill 0 if missing, print warning)
  chosen_action_kind -> y_action       int64
  chosen_tile        -> y_tile         int64
  candidates_mask    -> avail          float32
  game_id            -> match_id       (extracted: a_XXXX_r1_b0_t0 -> a_XXXX)

Outputs (方案 B — train/valid separated, no match_id in training npz):
  - {out_prefix}_train_split.npz  (~95%, no match_id, allow_pickle=False compatible)
  - {out_prefix}_valid.npz        (~5%, no match_id)
  - {out_prefix}_split.json       (match_id -> train/valid mapping)
  - {out_prefix}_full.npz         (all data with match_id, for reference)

Validation checks:
  1. avail legality: y_action ∈ avail → 100%
  2. tile legality: discard y_tile ∈ [0,33] and hand[y_tile] > 0
  3. pass y_tile values
  4. sample count + action distribution
  5. match_id coverage and split statistics
  6. no object arrays in train/valid npz
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

AUDIT_TO_V7 = {
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

TARGET_DTYPES = {
    "hand": np.int8,
    "discards": np.int8,
    "melds": np.int8,
    "action_history": np.int8,
    "scores": np.int32,
    "god": np.int8,
    "wall_remaining": np.int32,
    "turn": np.int8,
    "phase": np.int8,
    "target": np.int8,
    "seat": np.int8,
    "dealer": np.int8,
}

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]

# Reference distribution from bc_v7_train_v2_for_53838.npz
REFERENCE_DIST = {
    "discard": 78.0,
    "pass": 11.5,
    "chi": 4.1,
    "peng": 3.5,
    "gang": 0.7,
}


def extract_match_id(game_id: str) -> str:
    """Extract match_id from game_id.
    Format: a_XXXX_rN_bN_tN -> a_XXXX
    """
    # Split on _r followed by a digit
    idx = game_id.find("_r")
    if idx > 0:
        return game_id[:idx]
    return game_id


def convert(audit_path: str, out_prefix: str, report_path: str, valid_frac: float = 0.05, seed: int = 42):
    report_lines = []

    def rpt(msg=""):
        print(msg)
        report_lines.append(msg)

    rpt(f"{'='*70}")
    rpt(f"bc_v7 conversion report: {Path(audit_path).name}")
    rpt(f"{'='*70}")
    rpt()

    # Load audit data
    audit = np.load(audit_path, allow_pickle=True)
    N = len(audit["chosen_action_kind"])
    rpt(f"Input: {audit_path}")
    rpt(f"Total samples: {N}")
    rpt()

    # --- Build output arrays ---
    out = {}
    missing_dealer = False

    for audit_key, v7_key in AUDIT_TO_V7.items():
        if audit_key in audit:
            out[v7_key] = audit[audit_key].astype(TARGET_DTYPES[v7_key])
        elif audit_key == "obs_dealer":
            rpt("WARNING: obs_dealer not found in audit output, filling with 0")
            out["dealer"] = np.zeros(N, dtype=np.int8)
            missing_dealer = True
        else:
            rpt(f"ERROR: required key {audit_key} missing from audit output!")
            sys.exit(1)

    out["y_action"] = audit["chosen_action_kind"].astype(np.int64)
    out["y_tile"] = audit["chosen_tile"].astype(np.int64)
    out["avail"] = audit["candidates_mask"].astype(np.float32)

    # Extract match_id from game_id
    game_ids = audit["game_id"]
    match_ids = np.array([extract_match_id(g) for g in game_ids], dtype=object)
    out["match_id"] = match_ids

    # --- Validation checks ---
    y_action = out["y_action"]
    y_tile = out["y_tile"]
    avail = out["avail"]
    hand = out["hand"]

    all_pass = True

    # Check 1: avail legality — y_action ∈ avail
    action_in_avail = avail[np.arange(N), y_action] > 0
    pct_avail = action_in_avail.sum() / N * 100
    rpt(f"[CHECK 1] avail legality: y_action ∈ avail = {action_in_avail.sum()}/{N} ({pct_avail:.2f}%)")
    if pct_avail < 100.0:
        bad_idx = np.where(~action_in_avail)[0]
        rpt(f"  FAIL: {len(bad_idx)} samples have y_action NOT in avail!")
        rpt(f"  First 10 bad indices: {bad_idx[:10].tolist()}")
        for bi in bad_idx[:5]:
            rpt(f"    sample {bi}: y_action={y_action[bi]}, avail={avail[bi].tolist()}")
        all_pass = False
    else:
        rpt("  PASS")
    rpt()

    # Check 2: tile legality — discard y_tile ∈ [0,33] and hand[y_tile] > 0
    discard_mask = y_action == 0
    n_discard = discard_mask.sum()
    discard_tiles = y_tile[discard_mask]
    discard_hands = hand[discard_mask]

    tile_in_range = (discard_tiles >= 0) & (discard_tiles <= 33)
    n_range_ok = tile_in_range.sum()

    range_idx = np.where(discard_mask)[0][tile_in_range]
    hand_has_tile = discard_hands[tile_in_range, discard_tiles[tile_in_range]] > 0
    n_hand_ok = hand_has_tile.sum()

    rpt(f"[CHECK 2] tile legality (discard, n={n_discard}):")
    rpt(f"  y_tile ∈ [0,33]: {n_range_ok}/{n_discard} ({n_range_ok/n_discard*100:.2f}%)")
    rpt(f"  hand[y_tile] > 0: {n_hand_ok}/{n_range_ok} (of in-range)")
    if n_range_ok < n_discard:
        bad = discard_tiles[~tile_in_range]
        rpt(f"  FAIL: {n_discard - n_range_ok} tiles out of range: {np.unique(bad, return_counts=True)}")
        all_pass = False
    if n_hand_ok < n_range_ok:
        rpt(f"  FAIL: {n_range_ok - n_hand_ok} tiles not in hand")
        all_pass = False
    if n_range_ok == n_discard and n_hand_ok == n_range_ok:
        rpt("  PASS")
    rpt()

    # Check 3: pass y_tile values
    pass_mask = y_action == 5
    n_pass = pass_mask.sum()
    pass_tiles = y_tile[pass_mask]
    pass_unique, pass_counts = np.unique(pass_tiles, return_counts=True)
    rpt(f"[CHECK 3] pass y_tile values (n={n_pass}):")
    for v, c in zip(pass_unique, pass_counts):
        rpt(f"  y_tile={v}: {c} ({c/n_pass*100:.1f}%)")
    rpt()

    # Check 4: sample count + action distribution
    rpt(f"[CHECK 4] sample count: {N}")
    rpt()
    rpt("  Action distribution:")
    action_counts = Counter(y_action.tolist())
    for a in range(6):
        cnt = action_counts.get(a, 0)
        pct = cnt / N * 100
        ref = REFERENCE_DIST.get(ACTION_NAMES[a], 0)
        rpt(f"    {ACTION_NAMES[a]:8s}: {cnt:7d} ({pct:5.1f}%)  [ref: {ref:.1f}%]")
    rpt()

    # Check 5: match_id coverage
    match_id_list = match_ids.tolist()
    match_counter = Counter(match_id_list)
    n_matches = len(match_counter)
    rpt(f"[CHECK 5] match_id coverage: {n_matches} unique matches")
    match_sizes = sorted(match_counter.values(), reverse=True)
    rpt(f"  Samples per match: min={min(match_sizes)}, max={max(match_sizes)}, "
        f"mean={np.mean(match_sizes):.1f}, median={np.median(match_sizes):.0f}")
    rpt(f"  Top 5 matches by sample count:")
    for mid, cnt in match_counter.most_common(5):
        rpt(f"    {mid}: {cnt}")
    rpt()

    # --- Split by match_id ---
    rpt(f"{'='*70}")
    rpt(f"Splitting by match_id (valid_frac={valid_frac}, seed={seed})")
    rpt(f"{'='*70}")

    unique_matches = sorted(match_counter.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_matches)

    n_valid_matches = max(1, int(len(unique_matches) * valid_frac))
    valid_matches = set(unique_matches[:n_valid_matches])
    train_matches = set(unique_matches[n_valid_matches:])

    split_map = {}
    for m in unique_matches:
        split_map[m] = "valid" if m in valid_matches else "train"

    valid_mask = np.array([m in valid_matches for m in match_id_list])
    train_mask = ~valid_mask

    n_train = train_mask.sum()
    n_valid = valid_mask.sum()

    rpt(f"  Total matches: {n_matches}")
    rpt(f"  Train matches: {len(train_matches)} ({len(train_matches)/n_matches*100:.1f}%)")
    rpt(f"  Valid matches: {len(valid_matches)} ({len(valid_matches)/n_matches*100:.1f}%)")
    rpt(f"  Train samples: {n_train} ({n_train/N*100:.1f}%)")
    rpt(f"  Valid samples: {n_valid} ({n_valid/N*100:.1f}%)")
    rpt()

    # Verify no leakage
    train_mids = set(match_ids[train_mask].tolist())
    valid_mids = set(match_ids[valid_mask].tolist())
    overlap = train_mids & valid_mids
    if overlap:
        rpt(f"  FAIL: {len(overlap)} match_ids appear in both train and valid!")
        all_pass = False
    else:
        rpt("  No leakage: train and valid share no match_ids ✓")
    rpt()

    # Train action distribution
    rpt("  Train action distribution:")
    train_action_counts = Counter(y_action[train_mask].tolist())
    for a in range(6):
        cnt = train_action_counts.get(a, 0)
        pct = cnt / n_train * 100
        rpt(f"    {ACTION_NAMES[a]:8s}: {cnt:7d} ({pct:5.1f}%)")
    rpt()

    rpt("  Valid action distribution:")
    valid_action_counts = Counter(y_action[valid_mask].tolist())
    for a in range(6):
        cnt = valid_action_counts.get(a, 0)
        pct = cnt / n_valid * 100 if n_valid > 0 else 0
        rpt(f"    {ACTION_NAMES[a]:8s}: {cnt:7d} ({pct:5.1f}%)")
    rpt()

    # --- Save outputs ---
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full npz with match_id (for reference)
    full_path = f"{out_prefix}_full.npz"
    rpt(f"Saving full (with match_id) to {full_path} ...")
    np.savez_compressed(full_path, **out)
    rpt(f"  Saved {N} samples")

    # 2. Train split (no match_id, no object arrays)
    train_path = f"{out_prefix}_train_split.npz"
    train_out = {}
    for k, v in out.items():
        if k == "match_id":
            continue
        train_out[k] = v[train_mask]
    rpt(f"Saving train split (no match_id) to {train_path} ...")
    np.savez_compressed(train_path, **train_out)
    rpt(f"  Saved {n_train} samples")

    # 3. Valid split (no match_id)
    valid_path = f"{out_prefix}_valid.npz"
    valid_out = {}
    for k, v in out.items():
        if k == "match_id":
            continue
        valid_out[k] = v[valid_mask]
    rpt(f"Saving valid split (no match_id) to {valid_path} ...")
    np.savez_compressed(valid_path, **valid_out)
    rpt(f"  Saved {n_valid} samples")

    # 4. Split mapping JSON
    split_json_path = f"{out_prefix}_split.json"
    rpt(f"Saving split mapping to {split_json_path} ...")
    with open(split_json_path, "w") as f:
        json.dump(split_map, f, indent=2)
    rpt(f"  Saved {len(split_map)} match_id -> split mappings")
    rpt()

    # Check 6: verify no object arrays in train/valid npz
    rpt("[CHECK 6] Verify no object arrays in train/valid npz:")
    for name, path in [("train", train_path), ("valid", valid_path)]:
        d = np.load(path, allow_pickle=False)
        has_object = any(d[k].dtype == object for k in d.keys())
        if has_object:
            rpt(f"  FAIL: {name} npz contains object arrays!")
            all_pass = False
        else:
            rpt(f"  {name}: no object arrays ✓ (loadable with allow_pickle=False)")
        rpt(f"  {name} keys: {sorted(d.keys())}")
        for k in sorted(d.keys()):
            rpt(f"    {k}: {d[k].shape} {d[k].dtype}")
    rpt()

    if all_pass:
        rpt("ALL CHECKS PASSED ✓")
    else:
        rpt("SOME CHECKS FAILED ✗")

    # Write report
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport written to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert audit npz to bc_v7 training format with match_id split"
    )
    parser.add_argument("--audit", required=True, help="Input audit npz path")
    parser.add_argument("--out-prefix", required=True,
                        help="Output prefix (e.g., data/bc_v7_config1)")
    parser.add_argument("--report", required=True, help="Output report txt path")
    parser.add_argument("--valid-frac", type=float, default=0.05,
                        help="Fraction of matches for validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    args = parser.parse_args()
    convert(args.audit, args.out_prefix, args.report, args.valid_frac, args.seed)


if __name__ == "__main__":
    main()
