#!/usr/bin/env python3
"""P3 M2：克隆模型训练与保真度评估（A 的三条约束全部落实）。

评估设计：
- **同房留出**（主指标，A 约束 2）：每房内按样本顺序（=场次时间序）前 80% 训练、
  后 20% 测试 → 测「克隆同一批对手的未见决策」的一致率
- **跨房留出**（对照）：整房 80/20 → 测对全新对手的泛化上界损失
- 基线：全局最频牌（5.5%）、均匀随机（≈1/14=7.1%）
- 指标：top-1 一致率（主）、top-3、log-loss
- 分层：按房内对手胜率分强/弱两半（A 方案里的按强度分层）

用法：nice -n 15 uv run python verify/clone_train.py
"""
import collections
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, top_k_accuracy_score

OUR_UID = "u_a7f7c67bb14a"


def room_strength(rooms):
    """每房对手（非我们座位）的局胜率 → 强/弱分层。"""
    import glob
    wins = collections.Counter()
    total = collections.Counter()
    for room in rooms:
        for p in glob.glob(f"data/auto_sessions/{room}/events/*.json"):
            try:
                doc = json.load(open(p, encoding="utf-8"))
            except OSError:
                continue
            if doc.get("status") != "finished":
                continue
            seats = doc.get("seats") or []
            our = next((i for i, s in enumerate(seats)
                        if s.get("user_id") == OUR_UID), None)
            if our is None:
                continue
            seen = set()
            for b in doc["blocks"]:
                for e in b.get("events") or []:
                    if e["type"] != "round_ended":
                        continue
                    d = e.get("data") or {}
                    rn = d.get("round_no")
                    if rn in seen:
                        continue
                    seen.add(rn)
                    for s in range(4):
                        if s == our:
                            continue
                        total[room] += 1
                        if not d.get("draw") and e.get("seat") == s:
                            wins[room] += 1
    rate = {r: wins[r] / total[r] for r in rooms if total[r]}
    med = np.median(list(rate.values()))
    return {r: ("strong" if rate[r] >= med else "weak") for r in rate}


def evaluate(model, x, y, label, classes):
    proba = model.predict_proba(x)
    top1 = top_k_accuracy_score(y, proba, k=1, labels=classes)
    top3 = top_k_accuracy_score(y, proba, k=3, labels=classes)
    ll = log_loss(y, proba, labels=classes)
    print(f"  [{label}] top-1={top1:.1%} top-3={top3:.1%} logloss={ll:.3f} (n={len(y)})")
    return top1


def main():
    t0 = time.perf_counter()
    d = np.load("verify/out/clone_samples.npz", allow_pickle=True)
    x, y, meta = d["x"], d["y"].astype(int), d["meta"]
    rooms = [str(r) for r in d["rooms"]]
    print(f"样本 {len(y)} 特征 {x.shape[1]} 房 {len(rooms)}")

    room_ids = meta[:, 0]
    # 同房留出：房内按样本序（=时间序）前 80% 训练
    train_mask = np.zeros(len(y), dtype=bool)
    for rid in np.unique(room_ids):
        idx = np.flatnonzero(room_ids == rid)
        cut = int(len(idx) * 0.8)
        train_mask[idx[:cut]] = True
    same_test = ~train_mask

    # 跨房留出：整房 80/20
    rng = np.random.RandomState(20260927)
    uniq = np.unique(room_ids)
    rng.shuffle(uniq)
    train_rooms = set(uniq[: int(len(uniq) * 0.8)])
    cross_train = np.array([r in train_rooms for r in room_ids])
    cross_test = ~cross_train

    # 房强度分层
    tier = room_strength(rooms)
    tier_of_sample = np.array([tier.get(rooms[r], "weak") for r in room_ids])

    print(f"训练 {train_mask.sum()} / 同房测试 {same_test.sum()} / "
          f"跨房测试 {cross_test.sum()}  准备耗时 {time.perf_counter()-t0:.0f}s")

    classes = np.unique(y)
    model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    t1 = time.perf_counter()
    model.fit(x[train_mask], y[train_mask])
    print(f"训练耗时 {time.perf_counter()-t1:.0f}s")

    print(f"基线: 全局最频={np.bincount(y).max()/len(y):.1%} 均匀随机≈7.1%")
    evaluate(model, x[same_test], y[same_test], "同房留出(主指标)", classes)
    evaluate(model, x[cross_test], y[cross_test], "跨房留出(泛化)", classes)
    for tname in ("strong", "weak"):
        m = same_test & (tier_of_sample == tname)
        if m.sum():
            evaluate(model, x[m], y[m], f"同房留出·{tname}对手", classes)


if __name__ == "__main__":
    main()
