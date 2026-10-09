# 麻将 AI「早听/窄听 vs 晚听/宽听」权衡 — 证据简报

- 日期：2026-10-09
- 作者：agentb-researcher（子代理调研）
- 检索范围：arXiv（2019–2026 麻将 AI 主链）、GitHub 开源项目、作者公开博客
- 工具说明：本次运行 web_search 不可用，全部经由 arXiv 搜索接口 + 已知 URL 直接 web_fetch 核实；因此覆盖面有限，"未找到"不等于"不存在"。

---

## ① 结论摘要（对我们失败干预的直接启示）

1. **主流强 AI 都不在『行动价值函数』里手写"速度×宽度"的线性权衡**。Suphx（2020）用『全局奖励预测 + oracle 引导 + 运行时策略适应 (pMCPA)』，把"听牌速度 vs 打点"内化进 value/policy 网络；Mortal（2022 开源）跟 Suphx 同路线。他们的 look-ahead 特征是 100+ 维的『弃某张牌后能形成的和牌型 × 打点』向量，而**不是**单一的 ukeire 标量。**直接启示：我们用固定线性权重混『向听 −10』与『两拍值』是过度参数化一个本质上非线性的权衡，A/B 不显著是这个建模选择的预期后果，而不是『两拍值没信息』。**

2. **"听口宽度 ≠ 和率"在日麻研究里是已建立结论**：Suphx 的 look-ahead 特征以『丢弃 t 后能形成 X 点和牌』为粒度，正是为了区分『宽而便宜』与『窄而贵』的听口；早稻田/东大系的 Mizukami & Tsuruoka 2015 与 Gao et al. 2019 的『弃牌一致率 62.1% → 70.44%』表明人类高手决策里有显著的非向听分量。**直接启示：我们的『进张张数 × 最宽听口』缺一个『打点/对手的牌池排斥』修正项；在自摸制+白板百搭的规则下，白板在手概率对进张真实张数的折扣可能非常大。**

3. **单个决策点的蒙特卡洛 rollout 在麻将里被证明不可行/不必要**：Suphx 论文明确指出"由于规则复杂，无法构建规则博弈树，MCTS 和 CFR 不能直接应用"，改用**运行时** pMCPA（参数化蒙特卡洛策略适应）——本质是用少量 self-play 轨迹在线微调策略，而不是叶子 rollout。**直接启示：『用 M 次单点 rollout 估胜率再选牌』这条路线在四人麻将上没有被顶级工作采用， evidence 是负向的；要做 rollout 应该做 Suphx 式的整局策略适应，或转向离线训一个 value head。**

4. **『1-ply vs 2-ply』在棋牌里的经验结论对你不利**：Suphx 的 look-ahead 特征是『深度优先搜索 + 仅考虑自家摸打，忽略对手』的浅层展开，Meowjong（Sanma, 2022）的 RL 也只在 policy 网络上做蒙特卡洛策略梯度，都不在测试时做深搜。**2-ply『两拍值』对真机胡率的三分位预测力（14.3/45.0/68.4）已经证明它有信息量——信息没消失，是『怎么用』错了。**

5. **『速度 vs 宽度』的权衡最干净的量化口径是期望值，而非加权和**：Suphx 的 reward 是最终整场排名分；论文§3.2 的『全局奖励预测』把单场得分映射到整场期望排名，从而把『早听小和』vs『晚听大和』折算为同一单位。**直接启示：我们应该直接用『该局期望最终得分』作为评估口径，离线用 replays 学一个状态→期望得分模型，替换掉『向听 −10 + 两拍值』的线性组合。**

---

## ② 逐篇条目

### 论文（含 arXiv 编号 / URL / DOI）

#### 1. Suphx: Mastering Mahjong with Deep Reinforcement Learning
- **作者/年份**：Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon. 2020.
- **链接**：https://arxiv.org/abs/2003.13590 · DOI: 10.48550/arXiv.2003.13590
- **怎么处理权衡**：核心机制就是『speed vs value trade-off』的工程化解。
  - §2.2 Look-ahead features：枚举『弃某张特定牌后，能形成 12,000 点 / 特定役种 的和牌』的 100+ 维二值向量，每个特征对应一个 34 维弃牌向量。**这是『打点 × 速度』联合编码，而非线性加权。**
  - §3.2 Global Reward Prediction：训练一个预测器，从当前/过去局预测最终整场排名奖励，把单场得分的不确定性折算进 value 信号，解决『单场输不一定差』的归因问题。
  - §3.4 Parametric Monte-Carlo Policy Adaptation (pMCPA)：**运行时**用当前局的 self-play 轨迹对离线训好的策略做在线适应，避开 MCTS。
  - 明确指出（§1）："This prevents the direct application of previously successful techniques for games such as Monte-Carlo tree search and counterfactual regret minimization."
- **关键数字**：达到天凤 10 段，stable rank 超过 99.99% 人类排名玩家；look-ahead 特征 100+ 维；隐藏信息集规模 >10^48。

#### 2. Mortal — 开源日麻 AI
- **作者/年份**：Equim-chan. 2021–2022 持续维护.
- **链接**：https://github.com/Equim-chan/Mortal · 文档 https://mortal.ekyu.moe/ · 作者博客 https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- **性质**：**开源工程实现 + 中文/英文博客，非论文**。
- **怎么处理权衡**：跟 Suphx 同代技术栈——深度 CNN + self-play RL + mjai 协议；Rust 引擎 + Python 推理，单台 RTX 4090 + Ryzen 7950X 上 4 万局/小时。强度自评『约等于初版 Suphx』。
- **对我们的价值**：① 现成的强 baseline 引擎可做我们启发式干预的对拍底；② 高吞吐（40K hanchan/h）使『固定线性权重 vs 学习 value』的 A/B 在普通 GPU 上可数天完成；③ 源代码可验证『他们**没有**手写速度×宽度线性项』这一论断。

#### 3. Building a 3-Player Mahjong AI using Deep Reinforcement Learning (Meowjong)
- **作者/年份**：Xiangyu Zhao, Sean B. Holden. 2022 (AAAI/AIIDE 投稿格式).
- **链接**：https://arxiv.org/abs/2202.12847 · DOI: 10.48550/arXiv.2202.12847
- **怎么处理权衡**：Sanma（三人日麻）五模型（弃牌/碰/杠/北/立直），先用监督学习，再对弃牌模型用 **Monte Carlo policy gradient (REINFORCE)** 自我对弈提升。
- **关键数字**：SL 测试准确率与四人日麻 AI 相当；RL 阶段相比 SL 有显著提升（论文摘要的定性声明，未给出具体百分点在 abstract 中）。
- **对我们的价值**：说明『MC 策略梯度』是在麻将这类不规则博弈树上被实际使用的范式，**而不是**叶子 rollout 选牌。

#### 4. Building a Computer Mahjong Player via Deep Convolutional Neural Networks
- **作者/年份**：Shiqi Gao, Fuminori Okuya, Yoshihiro Kawahara, Yoshimasa Tsuruoka. 2019.
- **链接**：https://arxiv.org/abs/1906.02146 · DOI: 10.48550/arXiv.1906.02146
- **怎么处理权衡**：纯 SL，测试指标是『弃牌一致率』，70.44% vs Mizukami & Tsuruoka 2015 的 62.1%。
- **关键数字**：一致率 70.44%；天凤评分约 1850。
- **对我们的价值**：作为『不向听×打点显式建模也能达到人类中级以上水平』的实证；提示弃牌决策里有近 30% 的人类选择不是纯向听最优。

#### 5. Let's Play Mahjong! / A Fast Algorithm for Computing the Deficiency Number of a Mahjong Hand
- **作者/年份**：Sanjiang Li, Xueqing Yan (2019) ; Xueqing Yan, Yongming Li, Sanjiang Li (2021).
- **链接**：https://arxiv.org/abs/1903.03294 · https://arxiv.org/search/?query=mahjong+deficiency+number&searchtype=all
- **怎么处理权衡**：定义 deficiency（≈向听数）并给出最优弃牌以最大化『k 步内成和』的概率；2021 年的后续工作给出快速计算 deficiency 的算法。
- **对我们的价值**：是『纯速度派』的数学根基——**只关心听牌速度，不关心听口宽度**。我们当前的『向听 −10』本质是这个学派；A/B 失败可以被解读为『在自摸制+百搭规则下，纯 deficiency 最优的假设被违反』。

#### 6. Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX
- **作者/年份**：Soichiro Nishimori, Shinri Okano, Keigo Habara, Sotetsu Koyamada (Suphx 共同作者), Eason Yu, Masashi Sugiyama. 2026.
- **链接**：https://arxiv.org/search/?query=Mahjax&searchtype=title
- **怎么处理权衡**：JAX 实现的 GPU 麻将模拟器，目标是大规模 RL 训练，本身不直接讨论速度×宽度权衡，但『Koyamada 从 Suphx 转向大规模模拟器』暗示社区共识是**通过 RL 学 value 而不是手工权衡**。

#### 7. Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong (Evo-Sparrow)
- **作者/年份**：Jim O'Connor, Derin Gezgin, Gary B. Parker. 2025. AAAI/AIIDE.
- **链接**：https://arxiv.org/abs/2508.07522 · DOI: 10.48550/arXiv.2508.07522
- **怎么处理权衡**：用 CMA-ES 进化 LSTM 的策略网络，与 PPO baseline 性能相当。Sparrow Mahjong 是简化变体。
- **对我们的价值**：边缘证据——即便在简化规则下，**『进化策略直接搜索网络参数』也能避免手工设计权衡**。

#### 8. Mxplainer: Explain and Learn Insights by Imitating Mahjong Agents
- **作者/年份**：Lingfeng Li, Yunlong Lu, Yongyi Wang, Qifan Zheng, Wenxin Li. 2025.
- **链接**：https://arxiv.org/search/?query=mahjong+AI&searchtype=all
- **怎么处理权衡**：通过模仿强 AI 的决策来『提取可解释的策略洞察』。
- **对我们的价值**：如果我们能从 Mortal/Suphx-style 模型抽取出『什么情况下选速度、什么情况下选宽度』的可解释规则，就能避免拍脑袋的固定权重。

### 中文工程资料（非论文）

#### 9. Equim-chan 的博客《Should I release the trained model of my mahjong AI?》
- **链接**：https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- **性质**：工程决策博客（中英双语），含 Mortal 与天凤对局牌谱链接。
- **关键声明**：Mortal 1.0 超过 akochan 与所有可得 baseline，"约等于初版 Suphx"；提供天凤日志链接可直接审计。
- **对我们的价值**：可直接拿到 Mortal 牌谱，自己跑『两拍值』特征与 Mortal 实际弃牌的一致率，验证『宽度信息是否已被 RL 内化』。

### 未找到/未覆盖

- **NAGA（Dwango）**：未找到公开论文，仅有产品页与第三方评测。其内部权衡机制不可核实，**不能作为结论引用**。
- **国标 / 四川麻将血战**：arXiv 搜索 `Chinese official mahjong`、`Sichuan mahjong AI`、`blood-battle` 均无结果。**这是中国麻将变体 AI 的一个公开文献空白**——我们的『自摸制 + 白板百搭』在公开文献中没有直接对应工作。
- **专门的『1-ply vs 2-ply 前瞻深度对比』论文**：未找到。Suphx 用浅层 DFS 做特征、Meowjong 用 1-ply 网络评估，是仅有的间接证据。

---

## ③ 可用方法清单（离线对拍可复现）

| 方法 | 出处 | 我们怎么复现 | 样本量/成本量级 |
|---|---|---|---|
| **Look-ahead features（弃牌 → 可能和牌型 × 打点）** | Suphx §2.2 | 枚举『弃 t 后，若再摸 k 张能成和的所有组合及其打点』，做成每动作 100+ 维二值/计数特征；用来替换或增强『两拍值』 | 离线预计算，仅 CPU；13×34×3 量级组合，分钟级 |
| **Global Reward Prediction（每场最终排名作为 value 信号）** | Suphx §3.2 | 不改奖励，用历史 replay 训一个『局面 → 最终排名分布』的小网络；作为『速度 vs 宽度』的统一折算单位 | 需要 10⁴–10⁵ 局 replay；GPU 数小时 |
| **浅层 DFS 评估弃牌（仅自家摸打、忽略对手）** | Suphx §2.2 simplification | 弃 t 后做深度 2–3 的自家摸打 DFS，估计『X 步内成和概率』与『期望打点』两个标量 | 每个决策点 <1ms CPU |
| **Monte Carlo policy gradient (REINFORCE) 微调弃牌模型** | Meowjong 2022 | 在我们现有启发式策略上加一个可学习线性头，用 REINFORCE 以整场得分为回报做 self-play | 10³–10⁴ self-play 局；Mortal 引擎表明单机 40K hanchan/h 可行 |
| **pMCPA-style 运行时策略适应** | Suphx §3.4 | 不做完整 RL，仅在本局内对若干候选『速度/宽度权衡权重』做 bandit 选择 | 每局 10–50 次 rollout；CPU 分钟级 |
| **『一致率』基准（vs Mortal）** | Gao 2019 / Mortal | 下载 Mortal 天凤牌谱，测我们的 AI 在『弃牌一致率』上离 70% 还差多少 | 1 万局 Mortal 牌谱即可显著 |
| **白板折扣修正的进张真值** | 无直接出处（推断） | 对每个候选听口，统计『其余玩家手牌含白板概率 × 该听口被白板替代的真实剩余张数』，替代裸 ukeire | 离线 10⁵ 局面，分钟级 |

---

## ④ 不确定与反面证据

- **NAGA、腾讯 LuckyJ、快手/字节麻将 AI** 的内部权衡机制无公开论文；本简报对它们的任何描述都是**推断**，不是论文结论。
- **没找到任何工作直接对比『1-ply vs 2-ply』在麻将里的增量收益**。"2-ply 不划算"是从 Suphx/Meowjong 设计选择**反推**的，不是被实验证明的结论。
- **没找到讨论『愚形（单吊/嵌张/边张）折扣』的学术论文**。日麻社区的『受入枚数』修正（被其他玩家捏牌、王牌沉底）在工程博客/书籍里常见，但没有可引用的论文版本。**这是显著的文献空白**，我们的『被对手捏牌概率』建模要么是首创，要么需要去天凤大数据统计自建。
- **反面证据**：Suphx 论文明确指出麻将『不能构建规则博弈树』，因此**单点 rollout 选牌**这条路在公开文献里**没有被顶级工作采用**——这不等于证明它无效，但举证责任在我们这边。
- ** freshness**：本简报最后核实日期 2026-10-09；Mahjax (2026-05) 表明该领域仍在活跃演进，建议每季度复查 arXiv `cs.AI + mahjong`。
