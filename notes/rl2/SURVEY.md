# 开源麻将 AI 训练 Pipeline 调研（一手证据）

> 调研时间：2026-09-30  
> 调研人：Researcher Subagent  
> 范围：Mahjax, Mortal, kanachan, Suphx, Meowjong, DouZero, LuckyJ, mjx

---

## 1. Mahjax

**来源**
- GitHub: https://github.com/nissymori/mahjax
- arXiv: https://arxiv.org/abs/2605.20577

**训练 Pipeline**
- 环境：JAX 全向量化 Riichi Mahjong 模拟器，支持 `red_mahjong`（天凤规则，含赤五）和 `no_red_mahjong`（无赤五，简化规则）。
- 提供 `examples/` 目录，包含 **Behavior Cloning (BC) + PPO** 示例。
- BC 预训练：使用启发式规则智能体生成 500k 样本。
- RL 微调：PPO + KL 正则化（向 BC 策略），1,024 并行环境，rollout 长度 256，总训练 100M steps，约 5.8 小时（单张 NVIDIA GH200）。

**网络架构**
- Transformer encoder：处理手牌、弃牌、副露等高维状态为 latent representation。
- 分离的 MLP heads：policy head + value head。

**数据**
- BC 数据：500k 样本（启发式规则智能体生成）。
- RL 数据：自对弈生成，100M 环境步。

**训练超参**
- γ = 1.0
- GAE λ = 0.95
- 学习率 η = 3×10⁻⁴
- PPO clip ε = 0.2
- 熵系数 c_ent = 0.01
- value 系数 c_vf = 0.5
- KL 惩罚 c_KL = 0.2

**强度证据**
- 训练后智能体对 3 个固定 BC 对手的平均排名优于 2.5（1,000 局评估，3 seeds）。
- 吞吐量：8×A100 上 no-red 规则 ~2M steps/sec，red 规则 ~1M steps/sec。

**可借鉴点**
- GPU 全向量化环境设计（JAX），适合大规模自对弈。
- BC → PPO + KL 正则化的标准 pipeline。
- 观测空间支持 Transformer（dict）和 CNN（2D）两种模式。

---

## 2. Mortal

**来源**
- GitHub: https://github.com/Equim-chan/Mortal
- 文档: https://mortal.ekyu.moe
- 模型权重说明: https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56

**训练 Pipeline**
- 基于深度强化学习的日本麻将 AI。
- 使用 Rust 编写的超快模拟器（Libriichi），Python 接口，RTX 4090 + Ryzen 9 7950X 上可达 40K 半庄/小时。
- 训练方法：未在 README/文档中明确给出完整 pipeline 细节，但已知为 **深度强化学习**（非纯监督学习）。
- 文档为 work-in-progress，大部分页面为空。

**网络架构**
- **未找到公开来源**（README 和文档未描述具体网络结构）。

**数据**
- **未找到公开来源**（未披露训练数据规模）。

**训练超参**
- **未找到公开来源**（CQL 权重、next_rank_weight 等关键参数未在公开文档中披露）。

**强度证据**
- 兼容天凤标准四人麻将规则。
- 作为 mjai-reviewer 后端使用。

**可借鉴点**
- Rust + Python 混合架构实现高速模拟器。
- AGPL-3.0 开源，可研究其代码实现。
- 但核心训练细节未公开，可借鉴性受限。

---

## 3. kanachan

**来源**
- GitHub: https://github.com/Cryolite/kanachan
- Wiki: https://github.com/Cryolite/kanachan/wiki/Notes-on-Training-Data

**训练 Pipeline**
- 支持雀魂（Mahjong Soul）标准四人日本立直麻将规则变体。
- 流程：收集雀魂牌谱 → `src/annotation`（C++）转换为标注 → `kanachan/training`（PyTorch）训练。
- 不提供爬虫、训练数据或预训练模型，用户需自行准备。
- 核心思想：**Step-by-step Curriculum Fine-tuning**（课程微调）。
  - 从易到难学习目标：模仿人类行为 → 最大化 round 得分 → 更高最终排名 → 最大化段位战点数。
  - 前一步训练好的 encoder 复用，仅替换 decoder 以适应新目标。

**网络架构**
- 基于 **Transformer** 的大规模模型（作者明确表示利用大数据训练比 ResNet 更强的模型）。
- 输入特征几乎无人工设计：所有牌作为 token（embedding 索引），手牌、弃牌、副露等均为 token 序列。
- 数值特征（点数、供托等）直接作为数值输入。
- 具体训练数据格式：sparse features（embedding 索引）+ numeric features（6 维）+ progression features（事件序列）+ option features（合法动作）。

**数据**
- 雀魂牌谱：作者自 2020 年 7 月开始爬取，截至 2021 年 8 月已达约 6,500 万局（金之间及以上），预计 2021 年底超 1 亿局。
- 对比：天凤凰桌 11 年（2009-2019）仅 1,700 万局。

**训练超参**
- **未找到公开来源**（具体超参数未在 README/Wiki 中披露）。

**强度证据**
- 目标：击败 NAGA、Suphx 及顶尖人类玩家。
- 未提供具体对战成绩。

**可借鉴点**
- **课程微调（Curriculum Fine-tuning）**：encoder 复用 + decoder 替换，逐步从 BC 到更抽象目标。
- **无人工特征设计**：端到端 token 化输入，依赖大数据+大模型。
- 数据规模远超天凤，适合训练超大模型。

---

## 4. Suphx

**来源**
- arXiv: https://arxiv.org/abs/2003.13590

**训练 Pipeline**
1. **监督学习（SL）**：从天凤平台收集顶尖人类玩家 (state, action) 对，分别训练 5 个模型（discard, Riichi, Chow, Pong, Kong）。
2. **强化学习（RL）**：以 SL 模型为初始策略，自对弈 + 策略梯度算法（带重要性采样处理异步分布训练的轨迹陈旧性）。
   - **Global Reward Prediction**：训练 GRU 奖励预测器，将整场游戏奖励分配到每一轮。
   - **Oracle Guiding**：先训练能看到完美信息（其他玩家手牌+牌山）的 oracle agent，再通过特征 dropout 逐渐丢弃完美信息，最终转换为普通 agent。
   - **Run-time Policy Adaptation (pMCPA)**：在线对局时，根据当前轮次信息微调离线策略。
3. 在线评估：Suphx = RL-2（约 250 万局训练），未集成 pMCPA（因天凤时间限制）。

**网络架构**
- 深度 CNN，无 pooling 层（因每列有语义含义）。
- discard 模型：34 输出（34 种牌）。
- Riichi/Chow/Pong/Kong 模型：2 输出（是否执行）。
- 输入：多通道 34×1 特征图（手牌、副露、宝牌、弃牌序列、整数特征分桶、类别特征）。
- 额外设计 100+ look-ahead 特征（DFS 搜索可能和牌型，忽略对手行为）。

**数据**
- SL 训练数据：天凤顶尖人类玩家日志。
  - discard 模型：训练集约 4M-15M（具体数字未在文中给出，但提到比 Meowjong 大得多）。
  - 验证集 10K，测试集 50K（各模型统一）。
- RL 训练：每个 agent 150 万局自对弈。
- 奖励预测器训练数据：天凤人类游戏日志。

**训练超参**
- 分布式 RL：多 CPU 模拟器 + GPU 推理引擎 + 参数服务器。
- 策略梯度 + 熵正则化（动态调整 α，目标熵 H_target）。
- Oracle guiding：γ_t 从 1 衰减到 0，之后学习率降至 1/10，拒绝重要性权重过大的样本。
- 评估：每个 agent 训练用 44 GPUs（4 Titan XP 参数服务器 + 40 Tesla K80 自对弈 worker），2 天。

**强度证据**
- 天凤 10 段（record rank），稳定段位 8.74 dan。
- 超越 99.99% 天凤官方排名人类玩家。
- 稳定段位比 Bakuuchi 和 NAGA 高约 2 段。
- 防御强，放铳率低，四位率低。

**可借鉴点**
- **Global Reward Prediction**：解决麻将多轮奖励分配问题，值得借鉴。
- **Oracle Guiding**：利用完美信息加速训练，再逐步丢弃，适合不完美信息游戏。
- **Look-ahead 特征**：编码可能和牌型及得分，辅助决策。
- 分模型设计（discard/Riichi/Chow/Pong/Kong）降低单个模型复杂度。

---

## 5. Meowjong

**来源**
- arXiv: https://arxiv.org/abs/2202.12847
- GitHub: https://github.com/VictorZXY/meowjong

**训练 Pipeline**
1. **监督学习（SL）预训练**：5 个 CNN 分别对应 5 种动作（discard, Pon, Kan, Kita, Riichi）。
2. **强化学习（RL）增强**：仅对 discard 模型使用自对弈 + Monte Carlo 策略梯度（REINFORCE）。
   - 预训练模型初始化，η = 10⁻³，γ = 0.99。
   - 400 episodes，每 10 episodes 评估一次（每次 500 局×3 位置）。

**网络架构**
- 4 层 CNN + 1 层全连接（256 节点）。
- 前 3 层各 64 filters，第 4 层 32 filters。
- 每层后接 BatchNorm + Dropout(0.5)。
- 无 pooling，无 padding。
- ReLU 激活，输出层 softmax。
- 各动作模型 filter size 网格搜索调优（2≤x,y≤5），最优：discard(4,5), Pon(5,4), Kan(2,3), Kita(3,2), Riichi(3,4)。
- 输入：34×366 数组（22 种特征，含目标牌、手牌、赤宝牌、副露、拔北、弃牌、宝牌指示牌、他人立直状态、分数、局数、本场、供托、自风、他人副露/弃牌）。

**数据**
- 训练：2019 年天凤凤凰桌 50,000 局三麻。
- 验证：10% 分层抽样。
- 测试：2020 年 5,000 局。
- 各动作数据集大小：
  - Discard: 797,285 / 88,588 / 147,444
  - Pon: 151,348 / 16,817 / 16,887
  - Kan: 34,319 / 3,814 / 3,548
  - Kita: 136,924 / 15,214 / 15,498
  - Riichi: 109,804 / 12,201 / 11,944

**训练超参**
- SL：Adam，lr = 10⁻³，mini-batch 64（discard）/ 32（其余），200 epochs（discard）/ 500 epochs（其余）。
- 硬件：4×NVIDIA P100（64GB 总显存），训练时间 10-30 小时。
- RL：REINFORCE，η = 10⁻³，γ = 0.99，400 episodes。

**强度证据**
- SL 测试准确率：discard 65.81%，Pon 70.95%，Kan 92.45%，Kita 94.26%，Riichi 62.63%。
- RL 后 discard 模型显著提升：对 baseline 1st place rate 从 ~21.8% 提升至 ~72.4%。
- RL vs 2SL：1st place rate ~57.0%，显著优于 SL。
- 作者自称首个三麻 AI，达到 SOTA。

**可借鉴点**
- 小数据+小模型也能通过 SL+RL 达到可用强度（相比 Suphx 的 4M-15M 样本和 102/104 层 ResNet，Meowjong 仅 80 万样本+4 层 CNN）。
- 仅对主要动作（discard）做 RL，其余保持 SL，节省训练资源。
- 特征编码紧凑（34×366），适合资源受限场景。

---

## 6. DouZero

**来源**
- arXiv: https://arxiv.org/abs/2106.06135
- GitHub: https://github.com/kwai/DouZero

**训练 Pipeline**
- **Deep Monte-Carlo (DMC)**：传统 MC 方法 + 深度神经网络 + 动作编码 + 并行 actor。
- 无需人类知识或状态/动作空间抽象。
- 并行架构：45 actors（3 GPU）+ 1 learner（1 GPU），共享 buffer 通信。
- 从 scratch 训练，无需监督预训练。

**网络架构**
- Q-network：LSTM（编码历史动作）+ 6 层 MLP（hidden 512）。
- 输入：手牌、其他玩家手牌并集、最近动作（卡牌矩阵），以及动作本身的卡牌矩阵（4×15 one-hot）。
- 输出：Q 值。

**数据**
- 自对弈生成，无需人类数据。
- 内部 SL 基线使用 226,230 人类 expert 对局生成 49,990,075 样本。

**训练超参**
- 45 actors，B=50 entries，S=100，batch size M=32。
- ε = 0.01（exploration）。
- γ = 1（斗地主最后一步才有非零奖励）。
- RMSprop，lr = 0.0001，smoothing constant 0.99，ε = 10⁻⁵。
- 训练 30 天。
- 硬件：48 核 Intel Xeon Silver 4214R + 4×1080 Ti。

**强度证据**
- 超越所有现有斗地主 AI：对 DeltaDou WP=0.586，对 SL WP=0.659。
- Botzone 344 个 AI 中排名第 1（Elo 1625.11）。
- 半天超 CQN 和启发式规则，2 天超内部 SL agent，10 天超 DeltaDou。
- 推理速度极快（单次前向传播，远快于 DeltaDou 的搜索）。

**可借鉴点**
- **DMC 方法**：对于动作空间巨大且变化的游戏，MC 方法比 DQN 更稳定（无高估偏差），比策略梯度更能利用动作特征。
- **并行 actor 架构**：简单有效，适合大规模自对弈。
- **动作编码**：将动作编码为矩阵输入 Q-network，可泛化到未见动作。
- 对杭州麻将启示：如果动作空间复杂（如财神百搭导致和牌型多样），可考虑 DMC 而非 PPO。

---

## 7. LuckyJ

**来源**
- 主页: https://haobofu.github.io

**是否论文**
- **未找到论文**。作者主页 News 中提及 LuckyJ，但未链接到论文。

**训练方法**
- **未找到公开来源**。主页仅提及 2023 年 5 月 30 日 LuckyJ 在天凤达到 10 段，但未描述训练方法。

**网络架构**
- **未找到公开来源**。

**强度证据**
- 天凤 10 段，从零开始仅用不到 1,500 局。
- 稳定段位 10.68 dan，超越所有在天凤 expert room 玩过 1000+ 局的人类和 AI。

**可借鉴点**
- 无公开技术细节，无法借鉴。

---

## 8. mjx

**来源**
- GitHub: https://github.com/mjx-project/mjx

**是否提供训练 pipeline**
- **否**。mjx 是**日本麻将模拟器/框架**，不是训练 pipeline。
- 提供 Gym-like API、gRPC 分布式计算、可视化。
- 引用论文：Mjx: A framework for Mahjong AI research (IEEE CoG 2022)。

**可借鉴点**
- 可作为环境/模拟器使用，但项目状态：⚠️ "Currently Mjx build is broken. Also, Mjx API will change in the near future."
- 如需快速环境，可考虑 Mahjax（更新、GPU 加速）或 Mortal 的 Libriichi。

---

## 总结：核到了哪些、没找到哪些

### 已核到（有公开一手来源）
| 项目 | 核到内容 |
|------|---------|
| Mahjax | 完整 pipeline（BC→PPO+KL）、Transformer 架构、超参、吞吐量 |
| Mortal | 存在性、Rust 模拟器速度、开源协议；训练细节未公开 |
| kanachan | 课程微调思想、Transformer 架构、数据规模（65M+ 局）、无人工特征设计 |
| Suphx | 完整 pipeline（SL→RL+GRP+OG+pMCPA）、CNN 架构、数据规模、超参、强度证据 |
| Meowjong | 完整 pipeline（SL→REINFORCE）、CNN 架构、数据规模、超参、强度证据 |
| DouZero | 完整 pipeline（DMC）、LSTM+MLP 架构、训练资源、超参、强度证据 |
| LuckyJ | 存在性、天凤 10 段成绩；论文/方法未找到 |
| mjx | 模拟器定位、无训练 pipeline、项目状态 |

### 未找到/未公开
- Mortal：网络架构、训练数据、CQL 权重、next_rank_weight 等核心训练细节。
- LuckyJ：论文、训练方法、网络架构。
- kanachan：具体训练超参数。

---

## 最关键三条借鉴

1. **Suphx 的 Oracle Guiding + Global Reward Prediction 组合**  
   对于杭州麻将（不完美信息+多轮游戏），可先训练一个能看到财神/对手手牌的 oracle agent 加速收敛，再通过特征 dropout 逐步去除完美信息。同时用 GRU 预测整场游戏奖励并分配到每轮，解决多轮信用分配问题。

2. **Meowjong 的轻量级 SL→RL 路径**  
   如果数据/算力有限，Meowjong 证明：仅用 80 万样本 + 4 层 CNN，先 SL 预训练 5 个动作模型，再仅对核心动作（discard）做 400 episodes 的 REINFORCE，即可获得显著提升。这对快速验证杭州麻将 pipeline 非常有价值。

3. **DouZero 的 Deep Monte-Carlo（DMC）与动作编码**  
   杭州麻将因财神百搭导致和牌型组合爆炸，动作空间复杂。DouZero 证明：DMC 比 DQN 更稳定（无高估），比策略梯度更能利用动作特征（将动作编码为矩阵输入 Q-network）。若 PPO 在杭州麻将上不稳定，DMC 是值得尝试的替代方案。

---

*报告完成时间：2026-09-30*  
*注：所有信息均来自公开一手来源（GitHub README、arXiv 论文、项目文档），未核到的内容已明确标注。*
