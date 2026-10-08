#!/usr/bin/env python
"""子集快速评估：piao 点 + 听牌点 + 随机对照。秒级完成。"""
import sys, numpy as np, torch, time, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from nnrl2.model_v7 import MahjongTransformerV7

device = 'cuda'
ckpt = torch.load('runs/bc_v7_piao_finetuned.pt', map_location=device, weights_only=False)
model = MahjongTransformerV7(d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'模型加载完成 epoch={ckpt.get("epoch")}', flush=True)

d = np.load('data/audit_v56_with_model.npz', allow_pickle=True)
piao_idx = np.load('data/piao_indices.npy')
n_total = len(d['chosen_action_kind'])
shanten = d['shanten']
chosen = d['chosen_action_kind'].astype(np.int64)
base_top1 = d['model_top1_action'].astype(np.int64)

# 构造子集索引
tp_idx = np.where(shanten == -1)[0]
rng = np.random.RandomState(42)
# 随机对照：从非 piao 非听牌中抽 5000 条
non_special = np.where((shanten != -1))[0]
non_special = np.setdiff1d(non_special, piao_idx)
rand_idx = rng.choice(non_special, size=min(5000, len(non_special)), replace=False)

# 合并所有需要评估的索引
eval_idx = np.unique(np.concatenate([piao_idx, tp_idx, rand_idx]))
print(f'评估子集: {len(eval_idx)} 条 (piao={len(piao_idx)}, tingpai={len(tp_idx)}, random={len(rand_idx)})', flush=True)

KMAP = {'obs_hand':'hand','obs_discards':'discards','obs_melds':'melds',
    'obs_action_history':'action_history','obs_scores':'scores',
    'obs_god':'god','obs_wall_remaining':'wall_remaining',
    'obs_turn':'turn','obs_phase':'phase','obs_target':'target',
    'obs_seat':'seat','obs_dealer':'dealer'}

# 加载子集到 GPU
t0 = time.perf_counter()
obs = {}
for nk, mk in KMAP.items():
    a = d[nk][eval_idx]
    dt = np.float32 if a.dtype in (np.float32, np.float64) else np.int64
    obs[mk] = torch.from_numpy(a.astype(dt)).to(device)
print(f'GPU加载: {time.perf_counter()-t0:.1f}s mem={torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

# 推理（小 batch，避免一次性 2 万条撑爆 attention）
BS = 1024
pred_chunks = []
with torch.no_grad():
    for s in range(0, len(eval_idx), BS):
        e = min(s + BS, len(eval_idx))
        ob = {k: v[s:e] for k, v in obs.items()}
        ap, _ = model.predict_full(ob)
        pred_chunks.append(ap.argmax(dim=1).cpu().numpy())
pred = np.concatenate(pred_chunks)
print(f'推理完成', flush=True)

# 映射回原始索引
idx_to_pos = {v: i for i, v in enumerate(eval_idx)}

# === 指标 ===
chosen_sub = chosen[eval_idx]
base_sub = base_top1[eval_idx]

# 总体子集
o_ft = float((pred == chosen_sub).mean() * 100)
o_bs = float((base_sub == chosen_sub).mean() * 100)

# piao 点
piao_pos = np.array([idx_to_pos[i] for i in piao_idx])
p_bs = float((base_sub[piao_pos] == chosen_sub[piao_pos]).mean() * 100)
p_ft = float((pred[piao_pos] == chosen_sub[piao_pos]).mean() * 100)
p_bs_hu = int((base_sub[piao_pos] == 4).sum())
p_ft_hu = int((pred[piao_pos] == 4).sum())
p_n = len(piao_idx)
p_ed = int((chosen_sub[piao_pos] == 0).sum())

# 听牌点
tp_pos = np.array([idx_to_pos[i] for i in tp_idx])
tp_bs = float((base_sub[tp_pos] == chosen_sub[tp_pos]).mean() * 100)
tp_ft = float((pred[tp_pos] == chosen_sub[tp_pos]).mean() * 100)
tp_n = len(tp_idx)

# 各动作类型（子集内）
AN = ['discard','chi','peng','gang','hu','pass']
pa = {}
for ai, nm in enumerate(AN):
    m = chosen_sub == ai
    if m.sum() == 0: continue
    pa[nm] = {'n':int(m.sum()),
              'base':round(float((base_sub[m]==chosen_sub[m]).mean()*100),2),
              'ft':round(float((pred[m]==chosen_sub[m]).mean()*100),2)}

rpt = {
  'subset_size': int(len(eval_idx)),
  'overall_subset': {'base':round(o_bs,2),'ft':round(o_ft,2),'delta':round(o_ft-o_bs,2)},
  'tingpai': {'n':tp_n,'base':round(tp_bs,2),'ft':round(tp_ft,2),'delta':round(tp_ft-tp_bs,2)},
  'piao': {'n':p_n,'base':round(p_bs,2),'ft':round(p_ft,2),'delta':round(p_ft-p_bs,2),
    'base_pred_hu':p_bs_hu,'ft_pred_hu':p_ft_hu,
    'base_pred_hu_pct':round(p_bs_hu/p_n*100,1),'ft_pred_hu_pct':round(p_ft_hu/p_n*100,1),
    'engine_discard':p_ed},
  'per_action': pa,
}

print(f'\n=== 对比报告 ===')
print(f'子集总体:  base={o_bs:.2f}%  ft={o_ft:.2f}%  d={o_ft-o_bs:+.2f}pp  (n={len(eval_idx)})')
print(f'听牌点:    base={tp_bs:.2f}%  ft={tp_ft:.2f}%  d={tp_ft-tp_bs:+.2f}pp  (n={tp_n})')
print(f'Piao点:    base={p_bs:.2f}%  ft={p_ft:.2f}%  d={p_ft-p_bs:+.2f}pp  (n={p_n})')
print(f'Piao预测hu: base={p_bs_hu}/{p_n}({p_bs_hu/p_n*100:.1f}%)  ft={p_ft_hu}/{p_n}({p_ft_hu/p_n*100:.1f}%)')
print(f'引擎discard: {p_ed}/{p_n}')
for nm,v in pa.items():
    print(f'  {nm:>8s}: n={v["n"]:>7d}  base={v["base"]:.2f}%  ft={v["ft"]:.2f}%  d={v["ft"]-v["base"]:+.2f}pp')

os.makedirs('outputs', exist_ok=True)
with open('outputs/v7_piao_finetune_report.json','w') as f:
    json.dump(rpt, f, indent=2, ensure_ascii=False)
print(f'\n已保存: outputs/v7_piao_finetune_report.json')
