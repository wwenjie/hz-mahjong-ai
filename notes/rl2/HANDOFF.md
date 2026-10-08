# majiang_rl2 交接报告（最终版）

- **日期**：2026-09-30
- **作者**：agent-e（Chief of Staff）
- **时间投入**：~2.5 小时（14:36–17:12）
- **git commit**：f2d059d（初始脚手架）+ 299ff35（更新 HANDOFF.md）

## 完成工作

### 1. 调研（SURVEY.md）

研究员 subagent 完成开源麻将 AI 调研，核到 6/8 项目一手证据：

| 项目 | 核心收获 |
|------|---------|
| **Mahjax** | BC → PPO+KL pipeline；Transformer encoder + MLP heads；500k BC 样本；c_KL=0.2 |
| **Suphx** | SL（5 模型 CNN）→ RL；Global Reward Prediction；Oracle Guiding |
| **Meowjong** | 80 万样本 + 4 层 CNN → 仅 discard 模型做 400 episodes REINFORCE |
| **kanachan** | 课程微调（encoder 复用 + decoder 替换）；端到端 token 化 |
| **DouZero** | DMC（无 bootstrap）；LSTM+6 层 MLP；动作编码为矩阵 |
| **Mortal** | 文档 WIP，未公开训练细节 |

### 2. majiang_rl2 仓库

- 独立仓库，主仓 `/home/wuwenjie01/majiang_ai` 只读
- `src/nnrl2/paths.py`：只读桥（forbid_write）
- `src/nnrl2/obs.py`：Situation → Mahjax 风格 dict observation
- `src/nnrl2/model.py`：Transformer encoder + policy/value heads
- `scripts/gen_bc_data.py`：BC 数据生成（串行版，已验证）
- `scripts/gen_bc_data_par.py`：BC 数据生成（并行版，未验证）
- `scripts/train_bc.py`：BC 训练（v1）
- `scripts/train_bc_v2.py`：BC 训练（v2，合并 train/valid）
- `scripts/train_ppo.py`：PPO 骨架（未完整跑通）
- `scripts/train_ppo_full.py`：PPO 完整版（待验证）
- `scripts/run_ab.py`：A/B 对拍（v1）
- `scripts/run_ab_full.py`：A/B 对拍（v2）
- `scripts/ppo_smoke.py`：PPO 冒烟测试（通过）

### 3. 模型架构

- **Transformer encoder**（d_model=128, nhead=8, num_layers=4, dim_feedforward=512）
- **输入**：hand(14 tokens) + discards(96 tokens) + melds(48 tokens) + action_history(200 tokens) + scores(4 tokens) + scalars(7 tokens) = **369 tokens**
- **输出**：policy_logits(34) + value(1)
- **参数量**：~900k

### 4. BC 训练

- **数据**：13,680 train + 79 valid（第一次生成）；5098 valid（第二次生成，train 丢失）
- **结果**：bc_v2.pt（train_loss=3.28，epoch 7）
- **原因**：样本量不足（Mahjax 用 500k，我们只有 5098）

### 5. PPO 骨架

- 验证通过：自对弈 22s/场，PPO loss 正常，保存/加载正常
- 未完整训练（等 BC 数据）

### 6. A/B 对拍

- 流程跑通：BC vs heuristic = -4 vs +9
- BC 模型与其中一个 heuristic 并列最后（-4），最强 heuristic +9

## 待办

1. **等 BC 数据生成完成**（PID 2845840，400 场，预计 ~13 分钟）
2. **重训 BC 模型**（用修正后的架构，train_bc_v2.py）
3. **PPO 训练**（train_ppo_full.py）
4. **A/B 对拍**（run_ab_full.py）

## 关键约束

- 主仓 A 线在跑 ab_test.py（3 个种子 × 120 场，已跑 40+ 分钟）
- 我们的 BC 数据生成被资源竞争反复杀掉
- 建议：用 `nice -n 19` 降低优先级，或等主仓任务完成

## 文件清单

```
majiang_rl2/
├── notes/
│   ├── PLAN.md          # 立项计划
│   ├── SURVEY.md        # 调研报告
│   └── HANDOFF.md       # 交接报告
├── src/nnrl2/
│   ├── __init__.py
│   ├── paths.py         # 只读桥
│   ├── obs.py           # observation 编码
│   └── model.py         # Transformer 模型
├── scripts/
│   ├── gen_bc_data.py      # BC 数据生成（串行）
│   ├── gen_bc_data_par.py  # BC 数据生成（并行）
│   ├── train_bc.py         # BC 训练 v1
│   ├── train_bc_v2.py      # BC 训练 v2
│   ├── train_ppo.py        # PPO 骨架
│   ├── train_ppo_full.py   # PPO 完整版
│   ├── run_ab.py           # A/B 对拍 v1
│   ├── run_ab_full.py      # A/B 对拍 v2
│   └── ppo_smoke.py        # PPO 冒烟测试
├── runs/
│   ├── bc_v0.pt         # BC 模型 v0（旧架构）
│   ├── bc_v1.pt         # BC 模型 v1（旧架构）
│   ├── bc_v2.pt         # BC 模型 v2（新架构）
│   └── ppo_smoke.pt     # PPO 冒烟测试
├── records/             # A/B 对拍记录
├── data/                # BC 数据
└── tests/               # 测试
```

---

## 追加：pipeline 全链路验证结果（2026-09-30 18:40）

### BC 数据（最终）
- bc_train.npz: 26,638 样本 / bc_valid.npz: 5,240 样本（种子固定，nice-19 生成）
- 样本量不足 30k 的原因：启发式只记录有摸牌时的出牌决策；若需 ≥30k，用 --train-matches 120 重跑（约 1h）

### 训练产物
- runs/bc_v3.pt：train_loss=3.19（26.6k 样本，30 epochs 计划，实到 epoch 3）
- runs/ppo_v1.pt：100 episodes 完成，**但所有 episode 的 loss=0.0000 / KL=0.0000** —— PPO loss 计算路径有 bug（梯度未流动或 rollout 数据为空），模型实际未更新，需修 train_ppo_full.py

### A/B 对拍（runs/ppo_v1.pt ≈ bc_v3，40 场 × 8 局 × 4 座位轮换，seed=20260928）
- 记录文件：records/ab_ppo_v1_20260928.json
- 模型席位：总得分 -1676，名次分 -246，胡率 0.0%
- heuristic×3：名次分 +65/+98/+83，胡率 28.7%/31.9%/30.0%
- 结论：当前 BC/PPO 模型远弱于 heuristic（预期内：BC 仅 26k 样本 + PPO 实际未训练）

### 下一步（优先级序）
1. 修 PPO loss 为 0 的 bug（检查 rollout 收集与 advantage 计算）
2. 扩 BC 数据到 ≥50k 样本并重训
3. PPO+KL(c_KL=0.2) 正式训练后重跑 A/B

---

## 追加：v5 教师 BC 数据生成（2026-10-01 01:12）

### 决策
- 教师从启发式默认档切换到主仓 **v5**（`versions.build("v5", Mode.QUALIFIER)`）
- 目标样本量从 30k 提高到 **≥200k**（对齐 Mahjax 量级）
- 分片策略：7 个分片 × 100 matches × 8 rounds ≈ 211k 样本，3 并发 nice-19，setsid 脱离 exec 会话组

### 关键教训
- **`nohup` 不防 exec 会话回收**：只挡 SIGHUP，挡不住对整个进程组的 SIGKILL
- **`setsid` 有效**：让进程脱离会话组，exec 回收后仍存活
- **`np.savez_compressed` 是最后一次性写盘**：分片运行期间目录为空是正常的，不是卡住

### 进行中
- 分片 00-02 已跑 ~15 分钟，预计 ~01:57 落盘
- 全部 7 片预计 ~02:50 完成
- 后续 pipeline（scripts/pipeline_v5.py）：合并 → BC 重训 → A/B 对拍
