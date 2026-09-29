#!/bin/sh
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "verify_b_review_numbers.py" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "F4 seen" agent/out/verify-b-review.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
