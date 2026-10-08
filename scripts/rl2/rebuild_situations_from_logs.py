#!/usr/bin/env python3
"""从线上对战日志 + auto_sessions 事件流重建完整 Situation，转成 bc_v7_base 可推理的 obs 向量。

匹配策略（action-verified matching）：
- draw 决策：用 choice 里的出牌（discard:X）在 replay 里找到对应的 tile_discarded 事件，
  局面 = 该事件之前的摸牌后状态
- peng 决策（非 pass）：在 replay 里找到我们的 peng 事件，
  局面 = 触发该 peng 的弃牌事件之后的状态
- pass 决策（response_peng/response_chi + choice=pass）：
  按顺序匹配到我们本可以响应但选择放弃的窗口

数据源：
- 日志：``logs/a_*.jsonl``（decision.made 记录我方每个决策点的选择）
- 事件流：``data/auto_sessions/<match>/events/<game_id>.json``（完整 4 座位起手牌 + 事件序列）

输出 npz 字段：
- ``obs_hand`` (N, 34) int8
- ``obs_discards`` (N, 4, 24) int8
- ``obs_melds`` (N, 4, 4, 3) int8
- ``obs_action_history`` (N, 200) int8
- ``obs_scores`` (N, 4) int32
- ``obs_god`` (N,) int8
- ``obs_wall_remaining`` (N,) int32
- ``obs_turn`` (N,) int8
- ``obs_phase`` (N,) int8  (0=draw, 1=response)
- ``obs_target`` (N,) int8
- ``obs_seat`` (N,) int8
- ``obs_dealer`` (N,) int8
- ``chosen_action_kind`` (N,) int8  (0=discard, 1=chi, 2=peng, 3=gang, 4=hu, 5=pass)
- ``chosen_tile`` (N,) int8  (0-33, -1 for non-tile actions)
- ``candidates_mask`` (N, 6) float32
- ``candidates_tile_mask`` (N, 34) float32
- ``game_final_score`` (N,) int32
- ``shanten`` (N,) int8
- ``game_id`` (N,) str
- ``round_no`` (N,) int8
- ``phase`` (N,) str
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project paths
_script_dir = Path(__file__).resolve().parent
_majiang_ai = _script_dir.parent.parent / "majiang_ai"
_majiang_rl2 = _majiang_ai.parent / "majiang_rl2"

sys.path.insert(0, str(_majiang_ai / "src"))
sys.path.insert(0, str(_majiang_rl2 / "src"))

from majiang.sim import replay
from majiang.rules import tiles
from majiang.rules.action import chi_combinations
from majiang.rules.situation import PHASE_DRAW, PHASE_RESPONSE_PENG, PHASE_RESPONSE_CHI

from nnrl2.obs import situation_to_obs, TILE_KINDS

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]


def parse_choice(choice_str: str) -> tuple[int, int]:
    """Parse choice string to (action_kind, tile)."""
    if choice_str == "pass":
        return (5, -1)
    if choice_str == "hu":
        return (4, -1)
    if choice_str.startswith("discard:"):
        tile_str = choice_str[len("discard:"):]
        try:
            return (0, tiles.parse(tile_str))
        except Exception:
            return (0, -1)
    if choice_str.startswith("peng:"):
        tile_str = choice_str[len("peng:"):]
        try:
            return (2, tiles.parse(tile_str))
        except Exception:
            return (2, -1)
    if choice_str.startswith("chi:"):
        tiles_str = choice_str[len("chi:"):]
        parts = tiles_str.split("+")
        if len(parts) >= 2:
            try:
                return (1, tiles.parse(parts[1]))
            except Exception:
                pass
        try:
            return (1, tiles.parse(parts[0]))
        except Exception:
            return (1, -1)
    if choice_str.startswith("gang"):
        match = re.search(r":(.+)$", choice_str)
        if match:
            try:
                return (3, tiles.parse(match.group(1)))
            except Exception:
                pass
        return (3, -1)
    return (5, -1)


def parse_candidates_mask(candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    kind_mask = np.zeros(6, dtype=np.float32)
    tile_mask = np.zeros(TILE_KINDS, dtype=np.float32)
    for cand in candidates:
        kind, tile = parse_choice(cand)
        kind_mask[kind] = 1.0
        if kind == 0 and 0 <= tile < TILE_KINDS:
            tile_mask[tile] = 1.0
    return kind_mask, tile_mask


def parse_shanten(reason: str) -> int:
    match = re.search(r"向听[=\s](\d+)", reason)
    return int(match.group(1)) if match else -1


def find_our_seat(payload: dict, user_id: str) -> int:
    for i, s in enumerate(payload.get("seats", [])):
        if s.get("user_id") == user_id:
            return i
    return -1


def get_final_score(payload: dict, our_seat: int) -> int:
    for block in reversed(payload.get("blocks", [])):
        for event in reversed(block.get("events", [])):
            if event.get("type") == "game_ended":
                fs = event.get("data", {}).get("final_scores", [])
                if 0 <= our_seat < len(fs):
                    return fs[our_seat]
    total = 0
    for r in payload.get("rounds", []):
        scores = r.get("scores", [])
        if 0 <= our_seat < len(scores):
            total += scores[our_seat]
    return total


def _could_respond(hand_counts, my_melds, tile, catch_play, god_discarder, our_seat):
    if catch_play and god_discarder != our_seat:
        return False
    if tile == tiles.GOD:
        return False
    if hand_counts[tile] >= 2:
        return True
    chi_count = sum(1 for m in my_melds if m.is_chi)
    if chi_count >= 2:
        return False
    return len(chi_combinations(hand_counts, tile)) > 0


def _build_sample(situation, our_seat, log_d, final_score, rn):
    """Build a single sample from situation + log decision."""
    obs = situation_to_obs(situation, our_seat)
    chosen_kind, chosen_tile = parse_choice(log_d["choice"])
    kind_mask, tile_mask = parse_candidates_mask(log_d.get("candidates", []))
    shanten = parse_shanten(log_d.get("reason", ""))
    return {
        "obs": obs,
        "chosen_action_kind": chosen_kind,
        "chosen_tile": chosen_tile,
        "candidates_mask": kind_mask,
        "candidates_tile_mask": tile_mask,
        "game_final_score": final_score,
        "shanten": shanten,
        "game_id": log_d["game_id"],
        "round_no": rn,
        "phase": log_d["phase"],
    }


def rebuild_game(log_decisions: list[dict], payload: dict, our_seat: int) -> tuple[list[dict], dict]:
    """Rebuild situations for one game using action-verified matching."""
    samples = []
    stats = {"matched": 0, "unmatched_log": 0, "unmatched_replay": 0}

    log_by_round = defaultdict(list)
    for d in log_decisions:
        log_by_round[d["round_no"]].append(d)

    final_score = get_final_score(payload, our_seat)

    for state, events in replay.iter_rounds(payload):
        rn = state.round_no
        round_decisions = log_by_round.get(rn, [])
        if not round_decisions:
            for event in events:
                replay.apply_event(state, event)
            continue

        # --- Pre-index replay events for this round ---
        # Our draws: [(event_idx, drawn_tile_str)]
        our_draws = []
        # Our discards: [(event_idx, discarded_tile_str)]
        our_discards = []
        # Our peng/chi/gang: [(event_idx, type, tile_str)]
        our_responses = []
        # Discards by others: [(event_idx, seat, tile_str)]
        other_discards = []

        for i, event in enumerate(events):
            etype = event["type"]
            seat = event.get("seat", -1)
            tile_str = event.get("tile", "")
            if seat == our_seat:
                if etype == "tile_drawn":
                    our_draws.append((i, tile_str))
                elif etype == "tile_discarded":
                    our_discards.append((i, tile_str))
                elif etype in ("peng", "chi", "gang"):
                    our_responses.append((i, etype, tile_str))
            elif etype == "tile_discarded" and tile_str:
                other_discards.append((i, seat, tile_str))

        # --- Match draw decisions ---
        draw_log = [d for d in round_decisions if d["phase"] == "draw"]
        # Each draw decision has choice=discard:X.
        # Match to our_discards by tile code, in order.
        # The situation is captured right before the discard (after the preceding draw).
        discard_ptr = 0
        for log_d in draw_log:
            _, chosen_tile = parse_choice(log_d["choice"])
            if chosen_tile < 0:
                stats["unmatched_log"] += 1
                continue
            chosen_code = tiles.to_code(chosen_tile)

            # Find the next matching discard in replay
            found = False
            while discard_ptr < len(our_discards):
                event_idx, tile_str = our_discards[discard_ptr]
                if tile_str == chosen_code:
                    found = True
                    break
                discard_ptr += 1

            if not found:
                stats["unmatched_log"] += 1
                continue

            # Replay state up to just before this discard event
            # We need to re-walk events from the beginning of the round
            # to build the state at the right point.
            # Optimization: instead of re-walking, we capture the state
            # at the preceding draw event.
            
            # Find the preceding draw
            preceding_draw_idx = None
            for di, (ei, _) in enumerate(our_draws):
                if ei < event_idx:
                    preceding_draw_idx = di
                else:
                    break

            # Replay events up to (but not including) the discard
            # and capture the situation
            situation = _replay_to_point(state, events, event_idx, our_seat, PHASE_DRAW)
            if situation is not None:
                sample = _build_sample(situation, our_seat, log_d, final_score, rn)
                samples.append(sample)
                stats["matched"] += 1
            else:
                stats["unmatched_log"] += 1

            discard_ptr += 1

        # --- Match response decisions ---
        response_log = [d for d in round_decisions if d["phase"] in ("response_peng", "response_chi")]
        
        # For non-pass responses (peng:X, chi:...), match to our response events
        # For pass responses, match to other_discards where we could respond
        
        response_ptr = 0  # pointer into our_responses
        discard_resp_ptr = 0  # pointer into other_discards
        
        for log_d in response_log:
            choice = log_d["choice"]
            phase = log_d["phase"]
            chosen_kind, chosen_tile = parse_choice(choice)

            if chosen_kind == 5:  # pass
                # Match to the next other_discard where we could respond
                found = False
                while discard_resp_ptr < len(other_discards):
                    event_idx, seat, tile_str = other_discards[discard_resp_ptr]
                    discard_resp_ptr += 1
                    try:
                        tile = tiles.parse(tile_str)
                        # Build state at this point to check if we could respond
                        sit = _replay_to_point(state, events, event_idx + 1, our_seat, phase,
                                               offered=tile)
                        if sit is not None:
                            hand = sit.hand
                            if _could_respond(hand.counts, sit.melds_for(our_seat),
                                            tile, sit.god.catch_play,
                                            sit.god.god_discarder_seat, our_seat):
                                sample = _build_sample(sit, our_seat, log_d, final_score, rn)
                                samples.append(sample)
                                stats["matched"] += 1
                                found = True
                                break
                    except Exception:
                        continue
                if not found:
                    stats["unmatched_log"] += 1

            else:
                # Non-pass response: match to our response event
                found = False
                while response_ptr < len(our_responses):
                    event_idx, etype, tile_str = our_responses[response_ptr]
                    # Check if this response matches the log
                    if chosen_kind == 2 and etype == "peng":  # peng
                        if chosen_tile >= 0 and tile_str == tiles.to_code(chosen_tile):
                            found = True
                            break
                    elif chosen_kind == 1 and etype == "chi":  # chi
                        found = True
                        break
                    elif chosen_kind == 3 and etype == "gang":  # gang
                        found = True
                        break
                    response_ptr += 1

                if found:
                    # Find the triggering discard
                    trigger_idx = None
                    for j in range(event_idx - 1, -1, -1):
                        if events[j]["type"] == "tile_discarded" and events[j].get("seat") != our_seat:
                            trigger_idx = j
                            break
                    
                    if trigger_idx is not None:
                        trigger_tile_str = events[trigger_idx].get("tile", "")
                        try:
                            trigger_tile = tiles.parse(trigger_tile_str)
                        except Exception:
                            trigger_tile = None
                        
                        sit = _replay_to_point(state, events, trigger_idx + 1, our_seat, phase,
                                               offered=trigger_tile)
                        if sit is not None:
                            sample = _build_sample(sit, our_seat, log_d, final_score, rn)
                            samples.append(sample)
                            stats["matched"] += 1
                        else:
                            stats["unmatched_log"] += 1
                    else:
                        stats["unmatched_log"] += 1
                    response_ptr += 1
                else:
                    stats["unmatched_log"] += 1

        # Apply all events to advance state for next round
        for event in events:
            replay.apply_event(state, event)

    return samples, stats


def _replay_to_point(state, events, target_idx, our_seat, phase, offered=None):
    """Replay events from the start of the round to just before target_idx,
    then return the situation.

    Note: This creates a fresh state for each call, which is O(n) per call.
    For production use, consider incremental state tracking.
    """
    # Create a fresh state for this round
    # We need to reconstruct from the round's start_hands
    # This is expensive but correct
    
    # For efficiency, we'll use a simpler approach:
    # Walk events and track state, stopping at target_idx
    # We need to create a fresh ReplayState from the round's initial conditions
    
    # Actually, we can't easily create a fresh state here because we don't have
    # the start_hands. Instead, we'll use a different approach:
    # Walk through events and track all state changes, then reconstruct.
    
    # For now, let's use a simpler but less efficient approach:
    # re-walk events from the round start using a fresh state.
    # This requires access to the round's start_hands, which we don't have here.
    
    # Alternative: since we're already walking events sequentially in the caller,
    # we can capture the state at the right point during the main walk.
    # This function is a placeholder for the approach we'll actually use.
    
    # The real implementation is in rebuild_game_with_state_tracking below.
    raise NotImplementedError("Use rebuild_game_with_state_tracking instead")


def rebuild_game_v2(log_decisions: list[dict], payload: dict, our_seat: int) -> tuple[list[dict], dict]:
    """Rebuild situations using incremental state tracking with action-verified matching.
    
    Walk through events once, tracking state. At each decision point,
    capture the situation and match to the corresponding log decision.
    """
    samples = []
    stats = {"matched": 0, "unmatched_log": 0, "unmatched_replay": 0}

    log_by_round = defaultdict(list)
    for d in log_decisions:
        log_by_round[d["round_no"]].append(d)

    final_score = get_final_score(payload, our_seat)

    for state, events in replay.iter_rounds(payload):
        rn = state.round_no
        round_decisions = log_by_round.get(rn, [])
        if not round_decisions:
            for event in events:
                replay.apply_event(state, event)
            continue

        # Split decisions by type
        draw_log = [d for d in round_decisions if d["phase"] == "draw"]
        response_log = [d for d in round_decisions if d["phase"] in ("response_peng", "response_chi")]

        # Pre-index: our discards in this round (event_idx, tile_code)
        our_discards = []
        for i, event in enumerate(events):
            if event["type"] == "tile_discarded" and event.get("seat") == our_seat:
                our_discards.append((i, event.get("tile", "")))

        # Pre-index: our responses (peng/chi/gang) (event_idx, type, tile_code)
        our_responses = []
        for i, event in enumerate(events):
            if event.get("seat") == our_seat and event["type"] in ("peng", "chi", "gang"):
                our_responses.append((i, event["type"], event.get("tile", "")))

        # Match draw decisions to discards by tile code
        # draw_match[i] = event_idx of the discard that matches draw_log[i]
        draw_match = {}
        discard_ptr = 0
        for di, log_d in enumerate(draw_log):
            _, chosen_tile = parse_choice(log_d["choice"])
            if chosen_tile < 0:
                continue
            chosen_code = tiles.to_code(chosen_tile)
            while discard_ptr < len(our_discards):
                event_idx, tile_str = our_discards[discard_ptr]
                if tile_str == chosen_code:
                    draw_match[di] = event_idx
                    discard_ptr += 1
                    break
                discard_ptr += 1

        # Match response decisions
        # For non-pass: match to our_responses by type and tile
        # For pass: match to the next other-player discard where we could respond
        response_match = {}  # ri -> (event_idx_of_triggering_discard, offered_tile)
        response_ptr = 0
        pass_discard_ptr = 0  # pointer into other-player discards (computed during walk)
        
        # First, match non-pass responses
        non_pass_responses = [(ri, d) for ri, d in enumerate(response_log) 
                             if parse_choice(d["choice"])[0] != 5]
        pass_responses = [(ri, d) for ri, d in enumerate(response_log) 
                         if parse_choice(d["choice"])[0] == 5]
        
        for ri, log_d in non_pass_responses:
            chosen_kind, chosen_tile = parse_choice(log_d["choice"])
            while response_ptr < len(our_responses):
                event_idx, etype, tile_str = our_responses[response_ptr]
                matched = False
                if chosen_kind == 2 and etype == "peng":
                    if chosen_tile >= 0 and tile_str == tiles.to_code(chosen_tile):
                        matched = True
                elif chosen_kind == 1 and etype == "chi":
                    matched = True
                elif chosen_kind == 3 and etype == "gang":
                    matched = True
                
                if matched:
                    # Find triggering discard
                    for j in range(event_idx - 1, -1, -1):
                        if events[j]["type"] == "tile_discarded" and events[j].get("seat") != our_seat:
                            trigger_tile_str = events[j].get("tile", "")
                            try:
                                trigger_tile = tiles.parse(trigger_tile_str)
                            except Exception:
                                trigger_tile = None
                            response_match[ri] = (j, trigger_tile)
                            break
                    response_ptr += 1
                    break
                response_ptr += 1

        # Now walk through events and capture situations at decision points
        draw_log_idx = 0
        response_log_idx = 0
        pass_response_idx = 0
        
        for i, event in enumerate(events):
            if not state.opened:
                replay.apply_event(state, event)
                continue

            etype = event["type"]
            seat = event.get("seat", -1)
            tile_str = event.get("tile", "")

            # Check if this is a discard event that matches a draw decision
            if etype == "tile_discarded" and seat == our_seat:
                # Check if this discard matches any unmatched draw decision
                for di, target_idx in draw_match.items():
                    if target_idx == i:
                        # Capture situation BEFORE this discard
                        # (state is already at the right point - after draw, before discard)
                        log_d = draw_log[di]
                        situation = state.situation_for(our_seat, phase=PHASE_DRAW)
                        sample = _build_sample(situation, our_seat, log_d, final_score, rn)
                        samples.append(sample)
                        stats["matched"] += 1
                        break

            # Check if this is a discard by another player that triggers a response decision
            if etype == "tile_discarded" and seat != our_seat and tile_str:
                # First apply the discard so the state reflects it
                replay.apply_event(state, event)
                
                try:
                    tile = tiles.parse(tile_str)
                except Exception:
                    continue
                
                # Check if any non-pass response matches this point
                for ri, (trigger_idx, offered_tile) in response_match.items():
                    if trigger_idx == i:
                        log_d = response_log[ri]
                        situation = state.situation_for(
                            our_seat,
                            phase=log_d["phase"],
                            offered=tile,
                            responding=[our_seat],
                        )
                        sample = _build_sample(situation, our_seat, log_d, final_score, rn)
                        samples.append(sample)
                        stats["matched"] += 1
                        break
                
                # Check if any pass response matches this point
                hand = state.seats[our_seat].hand
                melds = state.seats[our_seat].melds
                if _could_respond(hand, melds, tile, state.catch_play, state.god_discarder, our_seat):
                    if pass_response_idx < len(pass_responses):
                        ri, log_d = pass_responses[pass_response_idx]
                        situation = state.situation_for(
                            our_seat,
                            phase=log_d["phase"],
                            offered=tile,
                            responding=[our_seat],
                        )
                        sample = _build_sample(situation, our_seat, log_d, final_score, rn)
                        samples.append(sample)
                        stats["matched"] += 1
                        pass_response_idx += 1
                
                continue  # Already applied the event

            replay.apply_event(state, event)

        # Count unmatched
        matched_draws = sum(1 for di in draw_match if any(
            s["phase"] == "draw" and s["round_no"] == rn and 
            parse_choice(s.get("_choice", ""))[1] == parse_choice(draw_log[di]["choice"])[1]
            for s in samples
        ))
        # Simpler: count from stats
        # The unmatched counts are approximate
        stats["unmatched_log"] += len(draw_log) - len(draw_match)
        stats["unmatched_log"] += len(response_log) - len(response_match) - pass_response_idx

    return samples, stats


def process_file(log_path: str, auto_sessions_dir: str) -> tuple[list[dict], dict]:
    """Process one log file and return all samples."""
    match_id = os.path.basename(log_path).replace(".jsonl", "")

    log_decisions = []
    user_id = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARNING: skipping bad JSON line in {log_path}", file=sys.stderr)
                continue
            if e.get("event") == "runtime.start":
                user_id = e.get("user_id")
            elif e.get("event") == "decision.made":
                log_decisions.append(e)

    if not user_id:
        print(f"  WARNING: no runtime.start in {log_path}", file=sys.stderr)
        return [], {"matched": 0, "unmatched_log": 0, "unmatched_replay": 0}

    if not log_decisions:
        return [], {"matched": 0, "unmatched_log": 0, "unmatched_replay": 0}

    decisions_by_game = defaultdict(list)
    for d in log_decisions:
        decisions_by_game[d["game_id"]].append(d)
    for game_id in decisions_by_game:
        decisions_by_game[game_id].sort(key=lambda x: x.get("mono_ms", 0))

    all_samples = []
    total_stats = {"matched": 0, "unmatched_log": 0, "unmatched_replay": 0}
    events_dir = os.path.join(auto_sessions_dir, match_id, "events")

    for game_id, game_decisions in decisions_by_game.items():
        events_path = os.path.join(events_dir, f"{game_id}.json")
        if not os.path.exists(events_path):
            print(f"  WARNING: no events file for {game_id}", file=sys.stderr)
            continue

        with open(events_path) as f:
            payload = json.load(f)

        our_seat = find_our_seat(payload, user_id)
        if our_seat < 0:
            print(f"  WARNING: user_id {user_id} not found in seats for {game_id}, "
                  f"assuming seat=0", file=sys.stderr)
            our_seat = 0

        samples, stats = rebuild_game_v2(game_decisions, payload, our_seat)
        all_samples.extend(samples)
        for k in total_stats:
            total_stats[k] += stats[k]

    return all_samples, total_stats


def save_npz(samples: list[dict], out_path: str):
    if not samples:
        print("No samples to save")
        return

    obs_keys = ["hand", "discards", "melds", "action_history", "scores",
                "god", "wall_remaining", "turn", "phase", "target", "seat", "dealer"]

    arrays = {}
    for key in obs_keys:
        values = [s["obs"][key] for s in samples]
        arrays[f"obs_{key}"] = np.stack(values)

    arrays["chosen_action_kind"] = np.array([s["chosen_action_kind"] for s in samples], dtype=np.int8)
    arrays["chosen_tile"] = np.array([s["chosen_tile"] for s in samples], dtype=np.int8)
    arrays["candidates_mask"] = np.stack([s["candidates_mask"] for s in samples])
    arrays["candidates_tile_mask"] = np.stack([s["candidates_tile_mask"] for s in samples])
    arrays["game_final_score"] = np.array([s["game_final_score"] for s in samples], dtype=np.int32)
    arrays["shanten"] = np.array([s["shanten"] for s in samples], dtype=np.int8)
    arrays["round_no"] = np.array([s["round_no"] for s in samples], dtype=np.int8)
    arrays["game_id"] = np.array([s["game_id"] for s in samples], dtype=object)
    arrays["phase"] = np.array([s["phase"] for s in samples], dtype=object)

    np.savez_compressed(out_path, **arrays)
    print(f"Saved {len(samples)} samples to {out_path}")


def print_sample_summary(samples: list[dict], n: int = 5):
    print(f"\n{'='*80}")
    print(f"Sample summary (first {min(n, len(samples))} of {len(samples)}):")
    print(f"{'='*80}")

    for i, s in enumerate(samples[:n]):
        obs = s["obs"]
        hand_tiles = []
        for t in range(TILE_KINDS):
            if obs["hand"][t] > 0:
                hand_tiles.extend([tiles.to_code(t)] * obs["hand"][t])

        chosen_tile_code = tiles.to_code(s['chosen_tile']) if 0 <= s['chosen_tile'] < TILE_KINDS else 'N/A'
        action_name = ACTION_NAMES[s['chosen_action_kind']] if 0 <= s['chosen_action_kind'] < 6 else '?'

        print(f"\n--- Sample {i} ---")
        print(f"  game_id: {s['game_id']}, round: {s['round_no']}")
        print(f"  phase: {s['phase']}")
        print(f"  hand ({len(hand_tiles)} tiles): {sorted(hand_tiles)}")
        print(f"  seat: {obs['seat']}, dealer: {obs['dealer']}")
        print(f"  wall_remaining: {obs['wall_remaining']}")
        print(f"  scores: {obs['scores']}")
        print(f"  chosen: {action_name} {chosen_tile_code}")
        print(f"  candidates_mask: {s['candidates_mask']}")
        print(f"  shanten: {s['shanten']}")
        print(f"  final_score: {s['game_final_score']}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild situations from logs")
    parser.add_argument("--logs-dir", default="/home/wuwenjie01/majiang_ai/logs")
    parser.add_argument("--auto-sessions-dir", default="/home/wuwenjie01/majiang_ai/data/auto_sessions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="/tmp/test_rebuild.npz")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_files = sorted(Path(args.logs_dir).glob("a_*.jsonl"))
    if args.limit:
        log_files = log_files[:args.limit]

    print(f"Processing {len(log_files)} log files...")

    all_samples = []
    total_stats = {"matched": 0, "unmatched_log": 0, "unmatched_replay": 0}

    for i, log_path in enumerate(log_files):
        if args.verbose:
            print(f"  [{i+1}/{len(log_files)}] {log_path.name}...")
        samples, stats = process_file(str(log_path), args.auto_sessions_dir)
        all_samples.extend(samples)
        for k in total_stats:
            total_stats[k] += stats[k]
        if args.verbose:
            print(f"    -> {len(samples)} samples (matched={stats['matched']}, "
                  f"unmatched_log={stats['unmatched_log']}, unmatched_replay={stats['unmatched_replay']})")

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Matching stats: matched={total_stats['matched']}, "
          f"unmatched_log={total_stats['unmatched_log']}, "
          f"unmatched_replay={total_stats['unmatched_replay']}")
    total_log = total_stats['matched'] + total_stats['unmatched_log']
    if total_log > 0:
        coverage = total_stats['matched'] / total_log * 100
        print(f"Coverage: {coverage:.1f}% of log decisions matched")

    save_npz(all_samples, args.out)
    print_sample_summary(all_samples, n=5)

    print(f"\n{'='*80}")
    print("Output fields and shapes:")
    print(f"{'='*80}")
    with np.load(args.out, allow_pickle=True) as data:
        for key in sorted(data.keys()):
            arr = data[key]
            print(f"  {key}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
