#!/usr/bin/env python
"""构建 bc_v7 piao 微调 v2 的分层采样索引（修复 v1 的 discard 淹没问题）。

构成：
  - piao 样本 (1,285) 过采样 5x = 6,425
  - 非 piao 对照按 chosen_action_kind 分层：
      discard ~3,000 / pass ~2,000 / chi 2,000 / peng 2,000 / gang 全量 1,986
  - 听牌点 (shanten=-1) 非 piao discard ~2,000（保护听牌场景）
保存: data/finetune_indices_v2.npy
"""
import numpy as np

rng = np.random.RandomState(42)
d = np.load("data/audit_v56_with_model.npz", allow_pickle=True)
chosen = d["chosen_action_kind"].astype(np.int64)
shanten = d["shanten"]
n_total = len(chosen)

piao_idx = np.load("data/piao_indices.npy")
piao_mask = np.zeros(n_total, dtype=bool)
piao_mask[piao_idx] = True
non_piao = np.where(~piao_mask)[0]

ACTION_NAMES = ["discard", "chi", "peng", "gang", "hu", "pass"]

# 1) piao 过采样 5x
piao_oversampled = np.tile(piao_idx, 5)

# 2) 分层对照（非 piao）
def sample(kind_id, n):
    pool = non_piao[chosen[non_piao] == kind_id]
    take = min(n, len(pool))
    return rng.choice(pool, size=take, replace=False)

s_discard = sample(0, 3000)
s_pass    = sample(5, 2000)
s_chi     = sample(1, 2000)
s_peng    = sample(2, 2000)
s_gang    = sample(3, 10**9)   # 全量 1,986

# 3) 听牌点非 piao discard 保护
tp_pool = non_piao[(shanten[non_piao] == -1) & (chosen[non_piao] == 0)]
s_tingpai = rng.choice(tp_pool, size=min(2000, len(tp_pool)), replace=False)

parts = {
    "piao_x5": piao_oversampled,
    "discard": s_discard,
    "pass": s_pass,
    "chi": s_chi,
    "peng": s_peng,
    "gang": s_gang,
    "tingpai_discard": s_tingpai,
}
all_idx = np.concatenate(list(parts.values()))
rng.shuffle(all_idx)

n_piao = len(piao_oversampled)
print(f"总规模: {len(all_idx)}")
for k, v in parts.items():
    print(f"  {k:>16s}: {len(v)}")
print(f"piao 占比: {n_piao/len(all_idx)*100:.1f}%  (上限 35%)")
kinds = np.bincount(chosen[all_idx], minlength=6)
print("动作分布:", {ACTION_NAMES[i]: int(kinds[i]) for i in range(6)})
print("听牌点样本数:", int((shanten[all_idx] == -1).sum()))

np.save("data/finetune_indices_v2.npy", all_idx)
print("已保存 data/finetune_indices_v2.npy")
