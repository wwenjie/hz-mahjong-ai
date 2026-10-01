#!/usr/bin/env bash
# C30 只读状态检查：给调度器触发器用。输出单行 JSON。
# 本脚本命令行不含被匹配的模式（c30_runner / convert_by_remaining 只出现在文件内容里），避免自匹配。
cd /home/wuwenjie01/majiang_ai
LOG=agent/out/c30-full-runner.log
if grep -q "^PROBE_DONE$" "$LOG" 2>/dev/null; then
  echo '{"done":true,"alive":true,"chunks":-1}'
  exit 0
fi
alive=false
if pgrep -f "c30_runner" >/dev/null 2>&1 || pgrep -f "convert_by_remaining" >/dev/null 2>&1; then
  alive=true
fi
nchunks=$(ls agent/out/c30-chunks/cs250-n*/chunk-*.json 2>/dev/null | wc -l)
echo "{\"done\":false,\"alive\":$alive,\"chunks\":$nchunks}"
