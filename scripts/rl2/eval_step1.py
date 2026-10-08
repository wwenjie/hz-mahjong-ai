#!/usr/bin/env python
"""Step 1: 推理并保存预测到 npy。Step 2 单独算指标。"""
import sys, numpy as np, torch, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from nnrl2.model_v7 import MahjongTransformerV7

ck = torch.load('runs/bc_v7_piao_finetuned.pt', map_location='cuda', weights_only=False)
m = MahjongTransformerV7().cuda()
m.load_state_dict(ck['model_state_dict'])
m.eval()
print('model ok', flush=True)

d = np.load('data/audit_v56_with_model.npz', allow_pickle=True)
pi = np.load('data/piao_indices.npy')
sh = d['shanten']
ti = np.where(sh == -1)[0]
rng = np.random.RandomState(42)
ns = np.setdiff1d(np.where(sh != -1)[0], pi)
ri = rng.choice(ns, min(5000, len(ns)), replace=False)
ei = np.unique(np.concatenate([pi, ti, ri]))
print(f'eval n={len(ei)}', flush=True)

KM = {'obs_hand':'hand','obs_discards':'discards','obs_melds':'melds',
    'obs_action_history':'action_history','obs_scores':'scores',
    'obs_god':'god','obs_wall_remaining':'wall_remaining',
    'obs_turn':'turn','obs_phase':'phase','obs_target':'target',
    'obs_seat':'seat','obs_dealer':'dealer'}
o = {}
for nk, mk in KM.items():
    a = d[nk][ei]
    dt = np.float32 if a.dtype in (np.float32, np.float64) else np.int64
    o[mk] = torch.from_numpy(a.astype(dt)).cuda()
print('data on gpu', flush=True)

with torch.no_grad():
    ap, _ = m.predict_full(o)
    pr = ap.argmax(1).cpu().numpy()
print('inference done', flush=True)

np.save('outputs/eval_pred.npy', pr)
np.save('outputs/eval_idx.npy', ei)
print('saved', flush=True)
