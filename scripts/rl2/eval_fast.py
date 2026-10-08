#!/usr/bin/env python
"""快速全量评估：预加载全部 obs 到 GPU，纯 GPU 分批推理。"""
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
n = len(d['chosen_action_kind'])
print(f'总样本: {n}', flush=True)

KMAP = {'obs_hand':'hand','obs_discards':'discards','obs_melds':'melds',
    'obs_action_history':'action_history','obs_scores':'scores',
    'obs_god':'god','obs_wall_remaining':'wall_remaining',
    'obs_turn':'turn','obs_phase':'phase','obs_target':'target',
    'obs_seat':'seat','obs_dealer':'dealer'}

t0 = time.perf_counter()
obs = {}
for nk, mk in KMAP.items():
    a = d[nk]
    dt = np.float32 if a.dtype in (np.float32, np.float64) else np.int64
    obs[mk] = torch.from_numpy(a.astype(dt)).to(device)
print(f'GPU加载: {time.perf_counter()-t0:.1f}s mem={torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

chosen = d['chosen_action_kind'].astype(np.int64)
shanten = d['shanten']
base_top1 = d['model_top1_action'].astype(np.int64)

# 纯 GPU 推理
BS = 8192
preds = []
t0 = time.perf_counter()
with torch.no_grad():
    for s in range(0, n, BS):
        e = min(s+BS, n)
        b = {k: v[s:e] for k, v in obs.items()}
        ap, _ = model.predict_full(b)
        preds.append(ap.argmax(dim=1).cpu().numpy())
pred = np.concatenate(preds)
print(f'推理: {time.perf_counter()-t0:.1f}s', flush=True)
del obs; torch.cuda.empty_cache()

# 指标
o_ft = float((pred==chosen).mean()*100)
o_bs = float((base_top1==chosen).mean()*100)
tp = shanten==-1
tp_bs = float((base_top1[tp]==chosen[tp]).mean()*100)
tp_ft = float((pred[tp]==chosen[tp]).mean()*100)
pm = np.zeros(n, dtype=bool); pm[piao_idx]=True
p_bs = float((base_top1[pm]==chosen[pm]).mean()*100)
p_ft = float((pred[pm]==chosen[pm]).mean()*100)
p_bs_hu = int((base_top1[pm]==4).sum())
p_ft_hu = int((pred[pm]==4).sum())
p_n = int(pm.sum())
p_ed = int((chosen[pm]==0).sum())

AN = ['discard','chi','peng','gang','hu','pass']
pa = {}
for ai, nm in enumerate(AN):
    m = chosen==ai
    if m.sum()==0: continue
    pa[nm] = {'n':int(m.sum()),'base':round(float((base_top1[m]==chosen[m]).mean()*100),2),
              'ft':round(float((pred[m]==chosen[m]).mean()*100),2)}

rpt = {'overall':{'base':round(o_bs,2),'ft':round(o_ft,2),'delta':round(o_ft-o_bs,2)},
 'tingpai':{'n':int(tp.sum()),'base':round(tp_bs,2),'ft':round(tp_ft,2),'delta':round(tp_ft-tp_bs,2)},
 'piao':{'n':p_n,'base':round(p_bs,2),'ft':round(p_ft,2),'delta':round(p_ft-p_bs,2),
   'base_pred_hu':p_bs_hu,'ft_pred_hu':p_ft_hu,'engine_discard':p_ed},
 'per_action':pa}

print(f'\n=== 对比报告 ===')
print(f'总体:   base={o_bs:.2f}%  ft={o_ft:.2f}%  d={o_ft-o_bs:+.2f}pp')
print(f'听牌:   base={tp_bs:.2f}%  ft={tp_ft:.2f}%  d={tp_ft-tp_bs:+.2f}pp  (n={int(tp.sum())})')
print(f'Piao:   base={p_bs:.2f}%  ft={p_ft:.2f}%  d={p_ft-p_bs:+.2f}pp  (n={p_n})')
print(f'Piao_hu: base={p_bs_hu}/{p_n}({p_bs_hu/p_n*100:.1f}%)  ft={p_ft_hu}/{p_n}({p_ft_hu/p_n*100:.1f}%)')
print(f'引擎discard: {p_ed}/{p_n}')
for nm,v in pa.items():
    print(f'  {nm:>8s}: n={v["n"]:>7d}  base={v["base"]:.2f}%  ft={v["ft"]:.2f}%  d={v["ft"]-v["base"]:+.2f}pp')

os.makedirs('outputs', exist_ok=True)
with open('outputs/v7_piao_finetune_report.json','w') as f:
    json.dump(rpt, f, indent=2, ensure_ascii=False)
print(f'\n已保存: outputs/v7_piao_finetune_report.json')
