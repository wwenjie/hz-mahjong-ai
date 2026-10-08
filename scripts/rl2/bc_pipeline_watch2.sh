#!/bin/bash
# 全流程看护 v2：等全部 train 分片 -> 校验 -> 合并 -> 生成 valid -> 终检
# 变更 vs v1：7h 截止；分片数目标 9（补充第 9 片以越过 200k）；valid 用 25 matches（>=5k 样本）
set -u
cd /home/wuwenjie01/majiang_rl2
STATUS=logs/bc_pipeline_status2.txt
PY=/home/wuwenjie01/majiang_ai/.venv/bin/python
export PYTHONPATH=src:/home/wuwenjie01/majiang_ai/src
: > "$STATUS"

say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

say "watcher v2 started (pid $$), target_parts=9"

DEADLINE=$(( $(date +%s) + 25200 ))   # 最多等 7 小时
retried=0
while true; do
    n=$(ls data/bc_v5_parts/train_part*.npz 2>/dev/null | wc -l)
    p=$(pgrep -cf '[g]en_bc_data_v5.py')
    say "parts=$n running=$p"
    [ "$n" -ge 9 ] && break
    if [ "$p" -eq 0 ]; then
        if [ "$retried" -ge 2 ]; then
            say "FAIL: no generator procs and parts=$n after retries"
            exit 1
        fi
        retried=$((retried+1))
        say "relaunching shard runner (retry=$retried)"
        setsid bash scripts/gen_bc_v5_shards.sh >> logs/shards_main.log 2>&1 < /dev/null &
        sleep 20
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        say "FAIL: deadline exceeded, parts=$n"
        exit 1
    fi
    sleep 60
done

say "all parts present: $(ls data/bc_v5_parts/train_part*.npz | wc -l)"

# 校验每个分片
"$PY" - <<'EOF' >> "$STATUS" 2>&1
import numpy as np, glob, sys
total = 0
for f in sorted(glob.glob('data/bc_v5_parts/train_part*.npz')):
    d = np.load(f)
    n = len(d['y'])
    total += n
    print(f'{f}: {n} samples')
print(f'TOTAL: {total}')
sys.exit(0 if total > 0 else 1)
EOF
rc=$?
[ $rc -ne 0 ] && { say "FAIL: shard validation rc=$rc"; exit 1; }

# 合并所有分片
"$PY" - <<'EOF' >> "$STATUS" 2>&1
import numpy as np, glob, sys
files = sorted(glob.glob('data/bc_v5_parts/train_part*.npz'))
ds = [np.load(f) for f in files]
keys = list(ds[0].files)
merged = {k: np.concatenate([d[k] for d in ds]) for k in keys}
np.savez_compressed('data/bc_v5_train.npz', **merged)
n = len(merged['y'])
print(f'Merged {len(files)} shards -> data/bc_v5_train.npz ({n} samples)')
sys.exit(0)
EOF
rc=$?
[ $rc -ne 0 ] && { say "FAIL: merge rc=$rc"; exit 1; }

# valid 集（25 matches，确保 >=5000 样本）
if [ ! -s data/bc_v5_valid.npz ]; then
    nice -n 19 "$PY" scripts/gen_bc_data_v5.py \
        --matches 25 --rounds 8 --seed 771014 --out data/bc_v5_valid.npz >> "$STATUS" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { say "FAIL: valid gen rc=$rc"; exit 1; }
else
    say "valid exists, skip gen"
fi

# 终检
"$PY" - <<'EOF' >> "$STATUS" 2>&1
import numpy as np
ok = True
for name, need in [('bc_v5_train.npz', 200000), ('bc_v5_valid.npz', 5000)]:
    d = np.load(f'data/{name}')
    n = len(d['y'])
    print(f'{name}: {n} samples (need >= {need}) -> {"OK" if n >= need else "SHORT"}')
    ok = ok and n >= need
raise SystemExit(0 if ok else 2)
EOF
rc=$?
if [ $rc -eq 0 ]; then say "DONE rc=0"; else say "DONE rc=$rc (acceptance short)"; fi
exit $rc
