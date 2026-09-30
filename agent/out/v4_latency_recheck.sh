#!/usr/bin/env bash
# agent-c 独立复核 A 的 v4 真机耗时（THREAD 2026-09-30 02:30 ★③）
# 口径：logs/*.jsonl 的 decision.made，phase=draw，按 decider 签名分流
set -euo pipefail
cd /home/wuwenjie01/majiang_ai
.venv/bin/python - <<'PY'
import json,glob,collections,statistics
buckets=collections.defaultdict(list)
nlogs=0
for f in sorted(glob.glob('logs/*.jsonl')):
    nlogs+=1
    for line in open(f,encoding='utf-8'):
        try: e=json.loads(line)
        except Exception: continue
        if e.get('event')!='decision.made': continue
        if e.get('phase')!='draw': continue
        d=e.get('decider') or ''
        if 'shape-value=True' in d: k='v4'
        elif d=='heuristic[wait-aware-tenpai=True]': k='v3'
        else: continue
        v=e.get('elapsed_ms'); b=e.get('budget_ms')
        if not isinstance(v,(int,float)): continue
        buckets[k].append((v,b))

def q(xs,p):
    xs=sorted(xs); 
    if not xs: return float('nan')
    import math
    i=(len(xs)-1)*p
    lo=int(math.floor(i)); hi=int(math.ceil(i))
    return xs[lo]+(xs[hi]-xs[lo])*(i-lo)

print(f'logs scanned: {nlogs}')
for k in ('v3','v4'):
    xs=[v for v,_ in buckets[k]]; bs={b for _,b in buckets[k]}
    if not xs: print(k,'EMPTY'); continue
    over=sum(1 for v,b in buckets[k] if b and v>b)
    print(f'--- {k}  n={len(xs)}  budgets={sorted(b for b in bs if b)}')
    print(f'    p50={q(xs,.50):.1f} p90={q(xs,.90):.1f} p99={q(xs,.99):.1f} max={max(xs):.1f} over_budget={over}')
PY
