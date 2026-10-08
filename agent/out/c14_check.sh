#!/bin/sh
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "pytest -q" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "passed" agent/out/pytest-c14.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
