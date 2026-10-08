# majiang_rl2 — Transformer BC + PPO+KL 训练线

## 立项依据（硬证据）

| 证据 | 来源 | 结论 |
|---|---|---|
| MLP 整段替换惨败 | majiang_rl/notes/REPORT-6.6-nnrl.md | 名次分 -3.4, t=-6.7，跨种子一致为负 |
| BC 克隆教师零增益 | 同上 | 学生最优 = 教师恒等 (0.9462)，无信息增量 |
| REINFORCE 出牌重排负 | majiang_rl/notes/REPORT-RL-20m.md | 名次分 -0.99/-1.58，双种子显著为负 |
| 近邻重排无可学信号 | majiang_rl/notes/REPORT-RL-probe.md | top1 vs top2 t=+0.21，可分辨率仅 22.4% |
| Mahjax BC→PPO+KL 有效 | arXiv:2605.20577 + GitHub README | 平均名次 < 2.5 vs BC baseline |
| Mortal 用名次辅助头 | Mortal config.example.toml | next_rank_weight=0.2, pts=[3,1,-1,-3] |

## 架构决定（锁定）

1. **网络**：Transformer encoder + MLP heads（Mahjax 路线，非 CNN）
2. **观测**：dict-style token 序列（手牌/弃牌/副露/动作历史），非 2D 通道
3. **动作空间**：34 类出牌 softmax + 响应布尔头（吃/碰/杠/胡/过）
4. **训练 pipeline**：启发式 BC → PPO + KL(π‖π_BC) 锚
5. **教师**：主仓 `heuristic`（默认档 v2）
6. **自对弈**：主仓 `sim/batch.py`（规则引擎不重写）
7. **推理**：训练用 torch/GPU；导出产物纯 Python（model.json + pure_forward）

## 硬约束

- 主仓 `/home/wuwenjie01/majiang_ai` **只读**（代码层强制）
- `nice -n 15`，同一时刻只跑一个重活
- 自对弈胜率**不作验收**（§14.3 场地偏差已知）
- 验收：与 v3/v4 对拍，名次分 |t|≥1.96 且同号
- 显存 ≤ 8.5GB（RTX 5060 Ti）

## 三道门

1. **BC 门**：held-out 一致率 ≥ 0.80（低于此说明表征不对，停）
2. **PPO 门**：训练曲线上升 + 不崩（w3_norm 有界）
3. **A/B 门**：对 heuristic 名次分 t ≥ +1.96（才改默认决策器）

## 目录

```
src/nnrl2/     代码（模型、数据、训练、评估）
scripts/       可执行脚本
notes/         计划、调研、报告
runs/          训练产物（gitignore）
records/       对拍记录（gitignore）
data/          数据集（gitignore）
```
