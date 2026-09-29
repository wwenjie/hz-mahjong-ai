#!/bin/sh
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "python -m pytest" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -qE "passed|failed|error" agent/out/pytest-full.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
