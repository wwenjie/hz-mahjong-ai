#!/bin/sh
cd /home/wuwenjie01/majiang_ai || exit 1
if pgrep -f "shape_value_mechanism_probe.py" >/dev/null 2>&1; then run=1; else run=0; fi
if grep -q "按副露数分层" agent/out/shape-value-mechanism.log 2>/dev/null; then res=1; else res=0; fi
echo "$run $res"
