#!/usr/bin/env bash
# 把本地仓库同步到 AutoDL 远端（tar 流式，不落盘中转）。
# 用法: bash scripts/sync_to_remote.sh [--with-data] [--with-agent-out]
set -euo pipefail

HOST=connect.nma1.seetacloud.com
PORT=53838
USER=root
REMOTE=/root/autodl-tmp/majiang_ai
LOCAL=/home/wuwenjie01/majiang_ai
PW_FILE=/tmp/.sshpw

WITH_DATA=0; WITH_OUT=0
for a in "$@"; do
  case "$a" in
    --with-data) WITH_DATA=1;;
    --with-agent-out) WITH_OUT=1;;
  esac
done

# askpass（密码不入 argv / history）
cat > /tmp/askpass.sh <<'EOF'
#!/bin/bash
cat /tmp/.sshpw
EOF
chmod 700 /tmp/askpass.sh
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0

EX=""
EX="$EX --exclude=./.venv --exclude=./.git --exclude=./logs --exclude=./runs"
EX="$EX --exclude=./.pytest_cache --exclude=__pycache__ --exclude=node_modules --exclude=*.pyc"
[ "$WITH_DATA" = 1 ] || EX="$EX --exclude=./data"
[ "$WITH_OUT" = 1 ]  || EX="$EX --exclude=./agent/out"

echo "[sync] 打包并推送 → $REMOTE （data=$WITH_DATA agent-out=$WITH_OUT）"
tar -cf - $EX -C "$LOCAL" . \
| setsid -w ssh -p "$PORT" -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/tmp/known_hosts_new \
    -o PreferredAuthentications=password -o PubkeyAuthentication=no \
    "$USER@$HOST" \
    "mkdir -p $REMOTE && tar -xf - -C $REMOTE && echo EXTRACT_OK && du -sh $REMOTE"
echo "[sync] 完成"
