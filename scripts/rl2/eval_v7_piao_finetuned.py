#!/usr/bin/env python
"""bc_v7_piao_finetuned 全量审计集评估脚本。

用法: .venv/bin/python -u scripts/eval_v7_piao_finetuned.py
"""
import sys, numpy as np, torch, time, json
sys.path.insert(0, 'src')
from nnrl2.model_v7 import MahjongTransformerV7

device = 'cuda'

# === 加载微调模型 ===
ckpt = torch.load('runs/bc_v7_piao_finetuned.pt', map_location=device, weights_only=False)
model = MahjongTransformerV7(d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'微调模型加载完成 (epoch={ckpt.get("epoch")})', flush=True)

# === 加载数据（一次性全量到 GPU）===
d = np.load('data/audit_v56_with_model.npz', allow_pickle=True)
piao_idx = np.load('data/piao_indices.npy')
n_total = len(d['chosen_action_kind'])
print(f'总样本: {n_total}', flush=True)

OBS_KEY_MAP = {
    'obs_hand': 'hand', 'obs_discards': 'discards', 'obs_melds': 'melds',
    'obs_action_history': 'action_history', 'obs_scores': 'scores',
    'obs_god': 'god', 'obs_wall_remaining': 'wall_remaining',
    'obs_turn': 'turn', 'obs_phase': 'phase', 'obs_target': 'target',
    'obs_seat': 'seat', 'obs_dealer': 'dealer',
}

print('加载全部 obs 到 GPU...', flush=True)
t0 = time.perf_counter()
all_obs = {}
for npz_key, model_key in OBS_KEY_MAP.items():
    data = d[npz_key]
    dtype = np.float32 if data.dtype in (np.float32, np.float64) else np.int64
    all_obs[model_key] = torch.from_numpy(data.astype(dtype)).to(device)
print(f'obs 加载完成: {time.perf_counter()-t0:.1f}s, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

chosen = d['chosen_action_kind'].astype(np.int64)
shanten = d['shanten']
base_top1 = d['model_top1_action'].astype(np.int64)

# === 分批推理（数据已在 GPU）===
batch_size = 4096
all_pred = []
t0 = time.perf_counter()

for start in range(0, n_total, batch_size):
    end = min(start + batch_size, n_total)
    obs_batch = {k: v[start:end] for k, v in all_obs.items()}
    with torch.no_grad():
        a_probs, _ = model.predict_full(obs_batch)
        pred = a_probs.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())

    done = end
    if (start // batch_size) % 50 == 0:
        print(f'  推理: {done}/{n_total} ({time.perf_counter()-t0:.1f}s)', flush=True)

all_pred = np.concatenate(all_pred)
print(f'推理完成: {time.perf_counter()-t0:.1f}s', flush=True)

# 释放 GPU
del all_obs
torch.cuda.empty_cache()

# === 指标计算 ===
overall_ft = float((all_pred == chosen).mean() * 100)
overall_base = float((base_top1 == chosen).mean() * 100)

tp_mask = shanten == -1
tp_base = float((base_top1[tp_mask] == chosen[tp_mask]).mean() * 100)
tp_ft = float((all_pred[tp_mask] == chosen[tp_mask]).mean() * 100)

piao_mask = np.zeros(n_total, dtype=bool)
piao_mask[piao_idx] = True
piao_base = float((base_top1[piao_mask] == chosen[piao_mask]).mean() * 100)
piao_ft = float((all_pred[piao_mask] == chosen[piao_mask]).mean() * 100)
piao_base_hu = int((base_top1[piao_mask] == 4).sum())
piao_ft_hu = int((all_pred[piao_mask] == 4).sum())
piao_total = int(piao_mask.sum())
piao_engine_discard = int((chosen[piao_mask] == 0).sum())

ACTION_NAMES = ['discard', 'chi', 'peng', 'gang', 'hu', 'pass']
per_action = {}
for act_id, name in enumerate(ACTION_NAMES):
    mask = chosen == act_id
    if mask.sum() == 0:
        continue
    per_action[name] = {
        'n': int(mask.sum()),
        'base_acc': float((base_top1[mask] == chosen[mask]).mean() * 100),
        'ft_acc': float((all_pred[mask] == chosen[mask]).mean() * 100),
    }

report = {
    'overall': {'base': round(overall_base, 2), 'finetuned': round(overall_ft, 2), 'delta': round(overall_ft - overall_base, 2)},
    'tingpai': {'n': int(tp_mask.sum()), 'base': round(tp_base, 2), 'finetuned': round(tp_ft, 2), 'delta': round(tp_ft - tp_base, 2)},
    'piao': {
        'n': piao_total, 'base': round(piao_base, 2), 'finetuned': round(piao_ft, 2), 'delta': round(piao_ft - piao_base, 2),
        'base_pred_hu': piao_base_hu, 'ft_pred_hu': piao_ft_hu,
        'base_pred_hu_pct': round(piao_base_hu / piao_total * 100, 1),
        'ft_pred_hu_pct': round(piao_ft_hu / piao_total * 100, 1),
        'engine_discard': piao_engine_discard,
    },
    'per_action': {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in per_action.items()},
}

print()
print('========== 对比报告 ==========')
print(f'总体一致率:   base={overall_base:.2f}%  ft={overall_ft:.2f}%  delta={overall_ft-overall_base:+.2f}pp')
print(f'听牌点一致率: base={tp_base:.2f}%  ft={tp_ft:.2f}%  delta={tp_ft-tp_base:+.2f}pp  (n={int(tp_mask.sum())})')
print(f'Piao一致率:   base={piao_base:.2f}%  ft={piao_ft:.2f}%  delta={piao_ft-piao_base:+.2f}pp  (n={piao_total})')
print(f'Piao预测hu:   base={piao_base_hu}/{piao_total} ({piao_base_hu/piao_total*100:.1f}%)  ft={piao_ft_hu}/{piao_total} ({piao_ft_hu/piao_total*100:.1f}%)')
print(f'引擎选discard: {piao_engine_discard}/{piao_total}')
print()
for name, v in per_action.items():
    print(f'  {name:>8s}: n={v["n"]:>7d}  base={v["base_acc"]:.2f}%  ft={v["ft_acc"]:.2f}%  delta={v["ft_acc"]-v["base_acc"]:+.2f}pp')

import os
os.makedirs('outputs', exist_ok=True)
with open('outputs/v7_piao_finetune_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'\n报告已保存: outputs/v7_piao_finetune_report.json')
