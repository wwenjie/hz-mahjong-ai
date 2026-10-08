#!/usr/bin/env python3
"""纯 Python 前向：读 `research/artifacts/*.json`，只用标准库算推理。

为什么必须有它：`src/majiang/**` 的零第三方依赖是硬约束（`pyproject.toml` 的 `dependencies = []`），
参赛运行路径不可能装 torch。导出模式照 `src/majiang/strategy/gbdt.py`——模型是「数据 + 手写前向」，
不是「加载框架」。

约束：
- 只 import 标准库（`--audit` 自我检查）
- 确定性：纯算术，无随机、无网络

用法::

    uv run python research/pure_forward.py --artifact research/artifacts/nn-value-v1.json \\
        --input 0.5,0.5,...        # 逗号分隔的 29 维特征
    uv run python research/pure_forward.py --audit
"""
import argparse
import ast
import json
import math
import os
import sys


def audit():
    path = os.path.abspath(__file__)
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    allowed = {"argparse", "ast", "json", "math", "os", "sys"}
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed:
                bad.append(node.module or "")
    print(f"审计 {path}")
    if bad:
        print("  非标准库导入: " + ", ".join(bad))
        return 1
    print("  仅标准库导入；无网络、无随机 —— 确定性且零依赖")
    return 0


def load_layers(weights):
    """把 torch 的 state_dict 键（net.<i>.weight/bias）排成有序层。"""
    layers = {}
    for key, value in weights.items():
        head, _, tail = key.rpartition(".")
        index = head.split(".")[-1]
        if not index.isdigit():
            raise SystemExit(f"无法解析的权重键：{key}")
        layers.setdefault(int(index), {})[tail] = value
    return [(idx, layers[idx]["weight"], layers[idx]["bias"]) for idx in sorted(layers)]


def forward(artifact, features):
    mean = artifact["normalization"]["mean"]
    std = artifact["normalization"]["std"]
    if len(features) != len(mean):
        raise SystemExit(f"特征维度不符：输入 {len(features)}，模型要求 {len(mean)}")
    x = [(v - m) / (s if s else 1.0) for v, m, s in zip(features, mean, std)]

    layers = load_layers(artifact["weights"])
    for pos, (_idx, weight, bias) in enumerate(layers):
        out = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(weight, bias)]
        x = out if pos == len(layers) - 1 else [v if v > 0 else 0.0 for v in out]
    return x[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact")
    ap.add_argument("--input", help="逗号分隔的特征向量")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    if args.audit:
        sys.exit(audit())
    if not args.artifact or args.input is None:
        ap.error("需要 --artifact 与 --input（或使用 --audit）")

    with open(args.artifact, encoding="utf-8") as f:
        artifact = json.load(f)
    features = [float(v) for v in args.input.split(",")]
    first = forward(artifact, features)
    second = forward(artifact, features)
    print(f"模型 {artifact['name']}（{artifact.get('kind')}）")
    print(f"  推理值 {first!r}")
    print(f"  两次一致：{first == second}")
    print(f"  是否有限：{math.isfinite(first)}")


if __name__ == "__main__":
    main()
