---
name: mahjong-train-model
description: 生成离线数据、训练模型，并把产物导出为纯 Python 可推理格式，最后与基线对拍。需要训练或评估新模型（含 RL/NN 研究线）时使用。
---

# 训练模型

## 何时用

- 要训练价值模型 / 对手模型 / 尝试新的网络结构或 RL 方法。
- 要评估一个训练产物是否值得进策略。

## 现有（树模型）链路

```bash
uv run python tools/gen_value_data.py --rounds 4000 --out data/value_train.npz --seed 20260924
uv run python tools/train_value.py --train 'data/value_train.part*.npz' \
    --valid 'data/value_valid.part*.npz' --out models/value_model.json --trees 200 --depth 3 --lr 0.05
uv run python tools/gen_opponent_data.py --rounds 1500 --out data/opponent_train.npz --seed 20260924
uv run python tools/train_opponent.py --out models/opponent_model.json
```

导出模式照 `src/majiang/strategy/gbdt.py`：模型 JSON = 树结构 + 阈值，推理是纯 Python 遍历。

## RL/NN 研究线（产物落 `research/`）

```bash
# 训练依赖只在 dev 组：numpy / scikit-learn / torch(CUDA)
ls research/            # 代码、训练记录、导出产物
```

## 硬规则

1. **必须显式给随机种子**（`--seed`）。省略就拒绝执行——随机性会让结论失去可比性。
2. **运行路径零第三方依赖**：`src/majiang/**` 不许 import numpy / torch / scikit-learn。
   产物必须是**纯 Python 可推理**（确定性、无网络、无第三方库）。
3. **特征只来自公开信息**：自身手牌、场上已打出牌、各家副露、牌墙剩余、财神状态。
   **对手手牌只能作离线标签，不得进入任何推理输入路径。**
4. **必须与基线对拍**：同一批样本、噪声底口径用 `verify/noise_floor.py`。
   **未超过噪声底 = 「无显著增益」**，不许切换默认决策器，产物归档备查。
5. **显存预算**：单次训练不超过本机 8GB；支持检查点续跑；训练与巡检不要同时全速跑。
6. **训练记录必须含**：数据指纹、超参、随机种子、依赖版本、GPU 型号。缺一项不算完成。
7. 模型缺失不得导致启动失败——相关档位必须回退启发式并打印警告。

## 验收

- 干净环境对同一输入连续推理两次结果一致，且未加载第三方库。
- 对拍报告含样本量、噪声底、结论；未超噪声底时默认决策器不变。
