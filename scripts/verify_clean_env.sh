#!/usr/bin/env bash
# 8.3 干净环境验收：从零证明运行时代码零第三方依赖、模型求值不依赖 numpy/sklearn。
#
# 用法：bash scripts/verify_clean_env.sh
# 退出码 0 = 全部通过。全程只动 /tmp 下的临时 venv，不碰本仓库环境。
set -euo pipefail

VENV=/tmp/majiang-clean-venv
GOLDEN_VALUE="16.8095439097"      # models/value_model.json 对全零向量的预测（开发环境算出）
GOLDEN_OPPONENT="0.0766896547"    # models/opponent_model.json 同上

echo "== 1. 建干净 venv：$VENV =="
rm -rf "$VENV"
if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$VENV"
else
    # 本机 python3.12 由 uv 托管时用 uv 建环境（同样是不带任何包的干净 venv）
    uv venv --python 3.12 "$VENV"
fi
PY="$VENV/bin/python"

echo "== 2. 仅安装项目自身（无 dev 依赖） =="
"$VENV/bin/pip" install -q -e .

echo "== 3. numpy/sklearn 不应存在 =="
if "$PY" -c "import numpy" 2>/dev/null; then echo "FAIL: numpy 泄漏进了干净环境"; exit 1; fi
if "$PY" -c "import sklearn" 2>/dev/null; then echo "FAIL: sklearn 泄漏进了干净环境"; exit 1; fi
echo "OK: numpy/sklearn 均不可导入"

echo "== 4. CLI 可用 =="
"$PY" -m majiang --help >/dev/null
"$VENV/bin/majiang" --help >/dev/null
echo "OK: python -m majiang 与 majiang 入口均正常"

echo "== 5. 模型手写树遍历求值（无科学计算库） =="
"$PY" - "$GOLDEN_VALUE" "$GOLDEN_OPPONENT" <<'EOF'
import sys
from majiang.strategy.gbdt import TreeEnsemble

golden_value, golden_opponent = float(sys.argv[1]), float(sys.argv[2])
value = TreeEnsemble.load("models/value_model.json")
opponent = TreeEnsemble.load("models/opponent_model.json")
v = value.predict([0.0] * 29)
o = opponent.predict([0.0] * 29)
assert abs(v - golden_value) < 1e-9, f"value 模型漂移: {v} != {golden_value}"
assert abs(o - golden_opponent) < 1e-9, f"opponent 模型漂移: {o} != {golden_opponent}"
print(f"OK: value={v:.10f} opponent={o:.10f} 与黄金值一致")
EOF

echo "== 6. 本地模拟一场 8 局（离线） =="
"$PY" - <<'EOF'
from majiang.runtime.decider import FirstLegalDecider
from majiang.sim.batch import run_match

result = run_match([FirstLegalDecider() for _ in range(4)], rounds=8, seed=7)
assert result.rounds == 8, f"局数不对: {result.rounds}"
total = sum(s.total_score for s in result.seats)
assert total == 0, f"零和被打破: {total}"
print(f"OK: 8 局完成，流局 {result.flows}，零和校验通过")
EOF

echo
echo "== 全部通过：干净环境下可安装、可运行、模型求值一致 =="
