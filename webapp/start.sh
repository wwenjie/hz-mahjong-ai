#!/usr/bin/env bash
# 一键启动杭州麻将网页对局（首次会自动安装依赖并构建前端）。
#
# 用法：
#   bash webapp/start.sh              # 默认端口 8848，8 局，你坐庄（座位 0）
#   bash webapp/start.sh --port 9000  # 换端口
#   bash webapp/start.sh --rounds 4 --seed 42
set -euo pipefail

cd "$(dirname "$0")"

# 1) 前端依赖与构建（幂等：已存在则跳过）
if [ ! -d node_modules ]; then
  echo "[start] 安装前端依赖 npm install ..."
  npm install --no-audit --no-fund
fi
if [ ! -f dist/index.html ]; then
  echo "[start] 构建前端 npm run build ..."
  npm run build
fi

# 2) 启动后端（复用仓库 .venv；纯标准库，无需额外依赖）
PY="../.venv/bin/python"
[ -x "$PY" ] || PY="python3"
echo "[start] 启动服务 ..."
exec "$PY" server.py "$@"
