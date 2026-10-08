# 开源麻将 AI 训练方法调研：为什么 BC 打不过规则 bot 教师

> 调研日期：2026-10-04
> 调研人：agent-e-researcher
> 问题背景：`bc_v7_base`（v5 教师数据，30 epochs，~1M 参数）对 v5 教师实测胜率仅 **19.2%**（v5 教师自身 26.7%）；所有 PPO 变体更差（16.9%–7.2%）。

---

## TL;DR（结论先行）

1. **BC 模仿一个弱教师，不可能逼近教师本身，更不可能超过教师。** 这不是数据量或 epoch 数不够的问题——是 BC 的结构性上限。Mortal 作者与 MahJAX 作者的路线一致：**BC/SL 只用作 RL 的初始化**，最终强度全部来自大规模自对弈 RL（Suphx 用 44 GPU × 2 天 × 1.5M 局自对弈；Mortal 用 40K 半庄/小时的 Rust 模拟器跑 RL）。
2. **开源项目没有一个靠"模仿规则 bot"取得成功。** Suphx 的 SL 教师是**天凤顶级人类玩家**（不是规则 bot），SL 之后仍然需要 RL 才能达到 10 段；MahJAX 用规则 bot 做 BC（500k 样本）只是为了让 PPO 训练稳定，最终性能完全由 PPO 阶段贡献。
3. **我们当前配置与开源项目差距是数量级的：**
   - 参数：~1M vs Mortal 的 **ResNet 192ch × 40 blocks（粗估 20M+ 参数）**
   - 数据：v5 教师（弱规则 bot）数据 vs Suphx 的天凤顶级人类数据 / Mortal 的自对弈数据
   - 训练：30 epochs BC vs Suphx 44 GPU 2 天 RL / Mortal 数月持续 RL
4. **建议：** 停止在弱教师数据上堆 epoch。两条可行路线：(a) 用更强教师（人类高分谱或现有更强 AI）重新 BC 作为 RL 初始化；(b) 直接走 MahJAX/Mortal 路线——BC 小规模初始化后立刻切大规模自对弈 RL。如果算力有限，优先把网络加深加宽（至少 192ch × 20+ blocks），并用 MahJAX 式的 GPU 向量化模拟器提升自对弈吞吐。

---

## 一、各项目训练规模 / 数据量 / 最终性能对比表

| 项目 | 架构 | 参数量（估） | 训练数据 / 初始化 | RL 训练规模 | 训练硬件 / 时长 | 最终性能 | 关键证据来源 |
|---|---|---|---|---|---|---|---|
| **Suphx**（微软） | 5 个独立 CNN 模型（discard/riichi/chow/pong/kong），34×1 通道卷积，无 pooling | 未公开精确数；discard 模型含 100+ look-ahead 特征通道，ResNet 式深层 CNN | SL：天凤顶级人类玩家日志（5 个模型分开训练）。Discard 测试准确率 **76.7%**，Riichi 85.7%，Chow 95.0%，Pong 91.9%，Kong 94.0% | 每个 RL 变体训练 **150 万局自对弈**；Suphx 最终版 = RL-2 训练约 **250 万局** | **44 GPU**（4× Titan XP 参数服务器 + 40× Tesla K80 自对弈 worker）× **2 天**（每个 RL 变体）；评估用 20× K80 × 2 天跑 100 万局 | 天凤 **10 段**（最高纪录段位），稳定段位超过 99.99% 人类玩家；SL→RL-basic→RL-1(+全局奖励预测)→RL-2(+oracle guiding) 逐步提升 | [arXiv:2003.13590](https://arxiv.org/abs/2003.13590) §4.1–4.2, §5 |
| **Mortal**（Equim-chan，开源） | 单一 ResNet：`conv_channels=192, num_blocks=40`，1D 卷积（34 列）+ CBAM Channel Attention + BatchNorm + Mish；输入 obs v4 = **1012 通道 × 34**；输出 action space = **37**；另有 GRU 全局奖励预测器（GRP, hidden 64×2 层） | 粗估：**20M+**（192ch × 40 blocks 的 1D-ResNet，每个 block 两个 3×1 卷积 + CA 模块） | 未使用人类数据 BC。直接自对弈 RL（离线 DQN 变体 + CQL + aux 任务 + KL/Boltzmann 探索，boltzmann_epsilon=0.005, temp=0.05） | Rust 模拟器 libriichi 达 **40K 半庄/小时**（RTX 4090 + R9 7950X，batch=2000）；训练从 2021-04 起持续多年，多版本迭代（v1.0 → v3.1 → v4.x） | 个人项目，单 GPU 训练；评估用 **100 万局** duplicate mahjong（1v3，同种子 4 局轮换） | 对 akochan（强规则 bot）：Mortal v3.1 平均顺位 **2.477** vs akochan 2.568（rank pt +1.89 vs -5.67，90/45/0/-135 分布）；胜率 21.2% vs 19.5%，放铳率 11.3% vs 13.0%。自评"与初版 Suphx 强度相当" | [mortal.ekyu.moe/perf/strength](https://mortal.ekyu.moe/perf/strength.html)、[config.example.toml](https://github.com/Equim-chan/Mortal/blob/main/mortal/config.example.toml)、[model.py](https://github.com/Equim-chan/Mortal/blob/main/mortal/model.py) |
| **MahJAX**（NTT/东大等，2026 新工作） | Transformer encoder + 独立 policy/value MLP 头；观测为 dict 结构（手牌/舍牌/副露 token 化） | 未公开精确数（Transformer encoder，示例规模） | BC：**50 万样本**，教师是**启发式规则 bot**（仅用于初始化，稳定 PPO 训练） | PPO + KL 正则（向 BC 策略）：1024 并行环境 × rollout 256 步，**1 亿环境步**；γ=1.0, λ=0.95, lr=3e-4, clip=0.2, ent=0.01, vf=0.5, KL=0.2 | 单张 **GH200 GPU × 5.8 小时**（1 亿步）；模拟器峰值 **200 万步/秒**（8×A100, no-red）/ **100 万步/秒**（red） | PPO 微调后对 3 个固定 BC 对手平均顺位**显著优于 2.5**（1000 局评估），证明 BC→RL 路线有效；论文明确说"SOTA 非本文目标" | [arXiv:2605.20577](https://arxiv.org/abs/2605.20577) §IV-B |
| **本项目 bc_v7_base** | 未知（~1M 参数） | **~1M** | BC：v5 规则 bot 教师数据，30 epochs | PPO 变体均劣于 BC（16.9%–7.2% vs BC 19.2%） | 未知 | 对 v5 教师胜率 **19.2%**（教师自身 26.7%，四人随机基准 25%） | 内部实测 |

---

## 二、差距分析：bc_v7_base vs 开源项目

### 2.1 为什么 BC 打不过规则 bot 教师——这是预期行为

**BC 的天花板 = 教师水平减去模仿误差。** 四个独立证据：

1. **MahJAX 的直接对照实验**（与本项目设置几乎完全相同）：用规则 bot 生成 50 万样本做 BC，得到 BC 策略作为基线。然后 PPO + KL 微调后，对 3 个固定 BC 策略的平均顺位**显著优于 2.5**。注意：他们从未报告"BC 策略打规则 bot 教师"能赢——BC 只是 PPO 的起点。
2. **Suphx 的 SL→RL 阶梯**：SL 模型（学人类顶级玩家，discard 准确率 76.7%）只是起点；RL-basic 就"显著优于 SL"，RL-2 再优于 RL-1。即使用**人类顶级玩家**做教师，SL 也远不够。
3. **Mortal 完全跳过人类数据**：自对弈 RL 从零达到"初版 Suphx 强度"，靠的不是模仿，是 40K 半庄/小时 × 长时间训练。
4. **四人麻将的胜率结构**：四人游戏随机基准胜率是 25%。v5 教师自身只有 26.7%（仅比随机好 1.7pt），说明 v5 教师本身强度接近随机。BC 学一个接近随机的教师，再叠加模仿误差（30 epochs、1M 参数的小模型），得到 19.2% 完全符合"BC ≤ 教师"的规律。

### 2.2 量化差距

| 维度 | bc_v7_base | MahJAX BC 基线 | Mortal | Suphx |
|---|---|---|---|---|
| 参数量 | ~1M | Transformer（示例规模，估数 M） | ~20M+（192ch×40blocks） | 深层 CNN ×5 模型 |
| 教师强度 | v5 规则 bot（胜率 26.7%，≈随机+1.7pt） | 启发式规则 bot | 无教师（自对弈） | 天凤顶级人类（7 段+） |
| BC 数据量 | 未知 | 50 万样本 | 0 | 每模型数十万~百万级 state-action 对 |
| RL 自对弈局数 | PPO 变体（规模未知，但更差） | 1 亿环境步 | 持续多年、数百万局级 | 150 万局/变体，最终 250 万局 |
| 模拟器吞吐 | 未知 | 100 万步/秒（8×A100） | 40K 半庄/小时（单卡） | 40×K80 集群 |
| 训练硬件 | 未知 | 1×GH200，5.8 小时 | 单 GPU，长期 | 44 GPU × 2 天 |

**关键差距排序（按影响）：**

1. **教师质量**（致命）：v5 教师 26.7% 胜率意味着它只比随机好一点。BC 的上限被钉死在教师之下。Suphx 用 7 段+人类；MahJAX 的规则 bot 教师只用于"让 PPO 别一开始就崩"。
2. **模型容量**：1M 参数对麻将这种 10^48 隐藏状态的游戏严重欠拟合。Mortal 用 192 通道 × 40 残差块（约 20M+ 参数），Suphx 用更深的 CNN + 100 多个 look-ahead 特征。
3. **缺少有效 RL**：我们的 PPO 变体全部比 BC 差，说明 PPO 配置/奖励/熵正则有问题（Suphx 论文明言"RL 对策略熵非常敏感：熵太小则自对弈不提升，熵太大则不稳定"，并专门设计了全局奖励预测 + oracle guiding + 熵正则）。MahJAX 的经验是 PPO **必须带向 BC 策略的 KL 正则**才稳定。
4. **数据量**：如果教师不变，堆更多同教师数据没用（上限不变）。

### 2.3 "需要远超教师的数据量才能逼近教师吗？"——不是

数据量解决的是**模仿误差**，不能突破**教师上限**。Suphx 的 discard 模型用人类顶级数据达到 76.7% top-1 准确率——即使 100% 准确，也只是复刻人类。真正的提升全部来自 RL 阶段（150 万局自对弈，44 GPU × 2 天）。**结论：不是数据量问题，是必须 RL 才能超越教师。**

---

## 三、具体建议（按优先级/可行性排序）

### 建议 0（先做，成本最低）：修复 PPO——为什么我们的 PPO 比 BC 还差

开源项目的 PPO 能稳定超越 BC，我们的更差，说明实现有问题。对照 MahJAX/Suphx 检查：

- **加 KL 正则向 BC 策略**（MahJAX：c_KL=0.2）。这防止 PPO 早期策略崩溃——最可能的根因。
- **熵正则调参**（Suphx：熵太小不提升，太大不稳定；需要扫描）。
- **奖励塑形**：Suphx 不用单局得分直接做奖励，而是训练一个**全局奖励预测器**（GRP，GRU 64×2）把终局奖励分配到每一局。直接用局分做奖励在麻将里是出了名的方差大。
- **CQL / 离线 RL**：Mortal 用 CQL（min_q_weight=5）+ aux 任务（next_rank_weight=0.2）。

### 建议 1（短期，1–2 周）：扩大模型容量 + 继续 BC 作为 RL 初始化

- 把网络从 ~1M 参数扩到 **至少 10–20M**：参考 Mortal 的 192 通道 × 40 blocks（或先用 128ch × 20 blocks 折中）。观测编码参考 Mortal 的 ~1000 通道 × 34 列 1D 卷积方案。
- 保留 bc_v7 数据做初始化，但**把 epoch 从 30 降到刚好收敛**（BC 只是初始化，过拟合教师反而限制 RL 探索）。

### 建议 2（中期，核心突破）：上大规模自对弈 RL

- **模拟器吞吐是瓶颈**。两条路：
  - 若现有模拟器够快（>1K 局/小时）：直接用，参考 Mortal 的单卡路线。
  - 若不够：考虑接入/参考 **MahJAX**（JAX 向量化，单 GPU 即达数十万步/秒；8×A100 达 100–200 万步/秒）或 **libriichi**（Rust，40K 半庄/小时）。
- **RL 规模目标**：Suphx 每个变体 150 万局。用 MahJAX 的吞吐，1 亿环境步在单张 GH200 上 5.8 小时——这个量级是我们应该瞄准的最低标准。
- **RL 算法**：从 MahJAX 的配方开始（PPO + KL→BC + 熵 0.01 + GAE λ=0.95 + clip 0.2），再逐步加 Suphx 的全局奖励预测。

### 建议 3（如果算力实在不够）：换教师，而不是堆数据

- 如果能拿到**天凤/雀魂高段人类牌谱**（Suphx 路线），BC 上限立刻从"26.7% 胜率的弱 bot"跳到"7 段人类"。这是最省算力的提升方式。
- 或用现有最强开源模型（如 Mortal 在线版/akochan）生成教师数据——注意 Mortal 权重未公开，但 akochan 可得。

### 不建议的方向

- **继续在 v5 教师数据上加 epoch / 加数据**：上限已锁死，30 epochs 之后再加只是过拟合。
- **期望 PPO 在没有 KL 锚定/奖励塑形的情况下自己修复**：我们所有 PPO 变体更差，说明配置层面有 bug，不是训练时长问题。

---

## 四、关键引用与证据

| 来源 | 关键事实 | 链接 |
|---|---|---|
| Suphx 论文 §4.1 | SL 测试准确率：discard 76.7%, riichi 85.7%, chow 95.0%, pong 91.9%, kong 94.0%；SL 数据来自天凤顶级人类 | https://arxiv.org/abs/2003.13590 |
| Suphx 论文 §4.2 | 每个 RL 变体 150 万局自对弈；44 GPU（4 Titan XP + 40 K80）× 2 天；评估 20×K80×2 天 / 100 万局 | 同上 |
| Suphx 论文 §3.1 | RL 对熵敏感，需熵正则；全局奖励预测 + oracle guiding | 同上 |
| Suphx 论文 §5 | Suphx = RL-2 × 250 万局；天凤 10 段，超 99.99% 人类 | 同上 |
| Mortal strength 页 | Mortal v3.1 vs akochan：平均顺位 2.477 vs 2.568；胜率 21.2% vs 19.5%；评估 100 万局 duplicate | https://mortal.ekyu.moe/perf/strength.html |
| Mortal config | ResNet 192ch × 40 blocks；batch 512；CQL min_q_weight=5；GRP aux 0.2；boltzmann 探索 | https://github.com/Equim-chan/Mortal/blob/main/mortal/config.example.toml |
| Mortal model.py / consts.rs | obs v4 = 1012ch × 34；ACTION_SPACE=37；GRU GRP 64×2 | https://github.com/Equim-chan/Mortal |
| Mortal 作者 gist | "近一年从零造出，超 akochan 及所有可得 baseline，约等于初版 Suphx" | https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56 |
| MahJAX 论文 §IV-B | BC 用规则 bot 50 万样本初始化；PPO+KL(0.2) 微调 1 亿步 / 单 GH200 × 5.8h / 1024 env × 256 步；对 3 个 BC 对手显著优于 2.5 | https://arxiv.org/abs/2605.20577 |
| MahJAX 论文 §IV-A | 模拟器 200 万步/秒（no-red, 8×A100）、100 万步/秒（red） | 同上 |

---

## 五、不确定性与局限

1. **Suphx 的 SL 具体样本数在 Table 3 的表格图片中**，本次未能从 PDF 提取到精确数字（环境无 pdftotext/Python 受限）；已用文中明确给出的测试准确率代替。
2. **Mortal 的总训练算力/时长未公开**（个人项目，作者自述"花费远超预期的时间和金钱"）；只有模拟器吞吐（40K 半庄/小时）和评估规模（100 万局）是确切数字。
3. **MahJAX 是 2026 年 5 月的新论文**，其 RL 实验是概念验证性质（single-round 模式），非完整半庄 SOTA；但其 BC→PPO 配方和吞吐数字对本项目最直接相关。
4. **本项目 bc_v7_base 的数据量、PPO 超参、训练时长未知**，对比表中标"未知"的项需要内部补齐后才能做更精确的差距量化。
5. **未能找到 nttbest/mahjax 这个 GitHub 路径**（404）；实际的 MahJAX 仓库是 `nissymori/mahjax`（论文一作 Nishimori 的账号），任务描述中的路径可能有误。

---

## 六、一句话回答四个关键问题

1. **BC 模仿规则 bot 的天花板是多少？** 严格低于教师。我们的 19.2% vs 教师 26.7% 就是这个规律的体现。
2. **需要远超教师的数据量才能逼近教师吗？** 不需要——数据量只影响模仿误差，不影响上限；30 epochs 已足够收敛，问题不在 epoch/数据量。
3. **必须 RL 才能超越吗？** 是。三个开源项目（Suphx/Mortal/MahJAX）无一例外：BC/SL 只做初始化，全部强度来自大规模自对弈 RL。
4. **下一步做什么？** (a) 修 PPO（加 KL→BC 锚定 + 熵调参）；(b) 扩模型到 10–20M 参数；(c) 提升模拟器吞吐至 10 万+ 步/秒量级后跑 ≥1 亿环境步的自对弈 RL；(d) 若算力不足，换更强教师（人类高段谱或 akochan）重新 BC。
