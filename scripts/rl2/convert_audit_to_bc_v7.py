#!/usr/bin/env python3
"""Convert audit-format npz (from rebuild_situations_from_logs.py) to bc_v7 training format.

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
  game_id            -> match_id       object

Validation checks:
  1. avail legality: y_action ∈ avail → 100%
  2. tile legality: discard y_tile ∈ [0,33] and hand[y_tile] > 0
  3. pass y_tile values
  4. sample count + action distribution (vs bc_v7 reference)
  5. game_id coverage (samples per game)
"""

import argparse
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


def convert(audit_path: str, out_path: str, report_path: str):
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
    out["match_id"] = audit["game_id"]  # object array

    # --- Validation checks ---
    y_action = out["y_action"]
    y_tile = out["y_tile"]
    avail = out["avail"]
    hand = out["hand"]
    match_id = out["match_id"]

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

    # For tiles in range, check hand count > 0
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
    rpt(f"  (reference bc_v7: y_tile=-1 (3335, 2.0%), y_tile=34 (160718, 98.0%))")
    rpt()

    # Check 4: sample count + action distribution
    rpt(f"[CHECK 4] sample count: {N}")
    if N < 5000:
        rpt(f"  WARNING: sample count < 5000, recommend merging with expert_bc_v7")
    rpt()
    rpt("  Action distribution:")
    action_counts = Counter(y_action.tolist())
    for a in range(6):
        cnt = action_counts.get(a, 0)
        pct = cnt / N * 100
        ref = REFERENCE_DIST.get(ACTION_NAMES[a], 0)
        rpt(f"    {ACTION_NAMES[a]:8s}: {cnt:7d} ({pct:5.1f}%)  [ref: {ref:.1f}%]")
    rpt()

    # Check 5: game_id coverage
    game_ids = match_id.tolist()
    game_counter = Counter(game_ids)
    n_games = len(game_counter)
    rpt(f"[CHECK 5] game_id coverage: {n_games} unique games")
    game_sizes = sorted(game_counter.values(), reverse=True)
    rpt(f"  Samples per game: min={min(game_sizes)}, max={max(game_sizes)}, "
        f"mean={np.mean(game_sizes):.1f}, median={np.median(game_sizes):.0f}")
    rpt(f"  Top 10 games by sample count:")
    for gid, cnt in game_counter.most_common(10):
        rpt(f"    {gid}: {cnt}")
    rpt(f"  Bottom 5 games by sample count:")
    for gid, cnt in game_counter.most_common()[:-6:-1]:
        rpt(f"    {gid}: {cnt}")
    rpt()

    # --- Save ---
    rpt(f"Saving to {out_path} ...")
    np.savez_compressed(out_path, **out)
    rpt(f"Saved {N} samples to {out_path}")
    rpt()

    # Print output structure
    rpt("Output fields:")
    verify = np.load(out_path, allow_pickle=True)
    for k in sorted(verify.keys()):
        a = verify[k]
        rpt(f"  {k}: {a.shape} {a.dtype}")
    rpt()

    if missing_dealer:
        rpt("NOTE: dealer was filled with 0 (not present in audit output)")
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
    parser = argparse.ArgumentParser(description="Convert audit npz to bc_v7 training format")
    parser.add_argument("--audit", required=True, help="Input audit npz path")
    parser.add_argument("--out", required=True, help="Output bc_v7 npz path")
    parser.add_argument("--report", required=True, help="Output report txt path")
    args = parser.parse_args()
    convert(args.audit, args.out, args.report)


if __name__ == "__main__":
    main()
