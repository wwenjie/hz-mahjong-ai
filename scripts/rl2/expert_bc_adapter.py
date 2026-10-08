#!/usr/bin/env python
"""expert BC 数据适配层：D 线 npz → v7 训练格式（dict obs + y_action/y_tile/avail）。

D 线 npz 键：hand/discards/melds/action_history/scores/dealer/god_in_hand/god_seen/
              phase/seat/target/turn/wall_remaining + y
v7 模型期望：上述全部 + god（财神牌型 id=33 常数）+ y_action（6 类）+ y_tile + avail（6 维 mask）

标签约定：
- discard npz: y = tile id (0-33) → y_action=0, y_tile=y
- response npz: y = 0=pass, 1=chi, 2=peng, 3=gang → y_action=y==0?5:y, y_tile=-1

avail 推导（响应样本，对齐 rules/action.py，白板不可凑碰/杠/吃）：
- pass 恒可行；peng: hand[target]>=2；gang: hand[target]>=3
- chi: target<27 且手牌有字面搭子（三种模式）
- 丢弃「只有 pass 可行」的 trivial pass 样本（v7 训练零梯度）

用法::

    python scripts/expert_bc_adapter.py \
        --src /home/wuwenjie01/majiang_rl/data/expert_bc \
        --out /home/wuwenjie01/majiang_rl2/data/expert_bc_v7 \
        --drop-trivial-pass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TILE_KINDS = 34
GOD = 33
ACTION_DISCARD, ACTION_CHI, ACTION_PENG, ACTION_GANG, ACTION_HU, ACTION_PASS = 0, 1, 2, 3, 4, 5


def avail_for_response(hand: np.ndarray, target: int) -> np.ndarray:
    """响应阶段 6 维 avail（白板不可凑碰/杠/吃）。"""
    avail = np.zeros(6, dtype=np.float32)
    avail[ACTION_PASS] = 1.0
    if target < 0 or target == GOD:
        return avail
    cnt = int(hand[target])
    if cnt >= 2:
        avail[ACTION_PENG] = 1.0
    if cnt >= 3:
        avail[ACTION_GANG] = 1.0
    if target < 27:
        r = target % 9
        patterns = []
        if r >= 2:
            patterns.append((target - 2, target - 1))
        if 1 <= r <= 7:
            patterns.append((target - 1, target + 1))
        if r <= 6:
            patterns.append((target + 1, target + 2))
        for a, b in patterns:
            if hand[a] >= 1 and hand[b] >= 1:
                avail[ACTION_CHI] = 1.0
                break
    return avail


def convert(src_dir: Path, out_dir: Path, drop_trivial_pass: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_obs: dict[str, list[np.ndarray]] = {}
    all_y_action: list[int] = []
    all_y_tile: list[int] = []
    all_avail: list[np.ndarray] = []
    stats = {"discard_in": 0, "response_in": 0, "dropped_trivial_pass": 0}

    # --- discard 样本 ---
    d = np.load(src_dir / "expert_discard.npz")
    n = len(d["y"])
    stats["discard_in"] = n
    obs_keys = [k for k in d.keys() if k != "y"]
    for k in obs_keys:
        all_obs.setdefault(k, []).append(d[k])
    # god 字段：财神固定为白板（33）
    god_arr = np.full(n, GOD, dtype=np.int8)
    all_obs.setdefault("god", []).append(god_arr)
    all_y_action.extend([ACTION_DISCARD] * n)
    all_y_tile.extend(d["y"].tolist())
    for _ in range(n):
        a = np.zeros(6, dtype=np.float32)
        a[ACTION_DISCARD] = 1.0
        all_avail.append(a)

    # --- response 样本 ---
    r = np.load(src_dir / "expert_response.npz")
    nr = len(r["y"])
    stats["response_in"] = nr
    kept_idx = []
    for i in range(nr):
        y_resp = int(r["y"][i])  # 0=pass, 1=chi, 2=peng, 3=gang
        y_action = ACTION_PASS if y_resp == 0 else y_resp  # chi=1,peng=2,gang=3 恰好对齐
        target = int(r["target"][i])
        avail = avail_for_response(r["hand"][i], target)
        if drop_trivial_pass and y_action == ACTION_PASS and avail.sum() <= 1.0:
            stats["dropped_trivial_pass"] += 1
            continue
        kept_idx.append(i)
        all_y_action.append(y_action)
        all_y_tile.append(-1)
        all_avail.append(avail)

    kept_idx = np.array(kept_idx, dtype=np.int64)
    for k in [k for k in r.keys() if k != "y"]:
        all_obs.setdefault(k, []).append(r[k][kept_idx])
    god_r = np.full(len(kept_idx), GOD, dtype=np.int8)
    all_obs.setdefault("god", []).append(god_r)

    # --- 合并 + 落盘 ---
    merged: dict[str, np.ndarray] = {}
    for k, parts in all_obs.items():
        merged[k] = np.concatenate(parts, axis=0)
    merged["y_action"] = np.array(all_y_action, dtype=np.int64)
    merged["y_tile"] = np.array(all_y_tile, dtype=np.int64)
    merged["avail"] = np.stack(all_avail)

    total = len(merged["y_action"])
    out_path = out_dir / "expert_bc_v7.npz"
    np.savez_compressed(out_path, **merged)

    stats["total_out"] = total
    stats["out_path"] = str(out_path)
    with open(out_dir / "adapter_meta.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/wuwenjie01/majiang_rl/data/expert_bc")
    ap.add_argument("--out", default="/home/wuwenjie01/majiang_rl2/data/expert_bc_v7")
    ap.add_argument("--drop-trivial-pass", action="store_true", default=True)
    args = ap.parse_args(argv)

    stats = convert(Path(args.src), Path(args.out), args.drop_trivial_pass)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
