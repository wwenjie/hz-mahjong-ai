#!/usr/bin/env bash
# ⓐ 同一性检验（耐久版）：_rank_discards 对 w/wo wait_aware_tenpai 是否逐点相同。
# 独立脚本避免命令行自匹配；-u 无缓冲落文件；外层重冲以抗环境回收。
set +e
cd /home/wuwenjie01/majiang_ai || exit 1
LOG=agent/out/probe-a-identity.log
: >"$LOG"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "--- attempt $attempt $(date -Is) ---" >>"$LOG"
  nice -n 19 .venv/bin/python -u -c "
import sys; sys.path.insert(0,'agent/verify')
from recompute_search_v3 import check_rank_discards_identity
check_rank_discards_identity()
print('PROBE_A_DONE', flush=True)
" >>"$LOG" 2>&1
  rc=$?
  if grep -q PROBE_A_DONE "$LOG"; then echo "完成 rc=$rc $(date -Is)" >>"$LOG"; break; fi
  echo "attempt $attempt rc=$rc 未成，10s 后续跑" >>"$LOG"
  sleep 10
done
