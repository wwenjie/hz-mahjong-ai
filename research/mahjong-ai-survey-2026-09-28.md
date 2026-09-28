# 麻将 AI 公开文献/开源调研（2026-09-28，agent-c）

**起因**：用户 23:19 转述 A 的判断——v3 已到瓶颈，让我们搜文献/博客找优化头绪。

**方法说明（重要，先声明取证方式）**：
- 本机 `web_search` 工具**已禁用（无 provider）**，故全部证据**直接取自一手 API**：
  arXiv API、GitHub REST API、Crossref REST API、arXiv HTML 全文、GitHub raw README / cn.bing.com。
- **本文件是两份独立取证的合并版**：一份来自本线 23:2x 的直接取证（已提交 `605a982`），
  一份来自用户派出的隔离子代理（`mahjong-ai-lit-survey`）的独立取证（23:2x 完成）。
  两份**不冲突**，本版把两份的来源并成一张表、把候选实验并成一份 Top-5，**不丢任何已验证来源**。
- **只写不编**：每条给 URL/ID；找不到的写「未找到来源」。数字均来自摘要原文，并区分**来源自述**与**我方估计**。
- 已知覆盖缺口：Semantic Scholar API 全程 429、DuckDuckGo 触发验证码 ⇒ 本报告**不覆盖**
  非 arXiv/Crossref/GitHub/Bing 索引内的中文期刊与学位论文。

---

## 0. 一句话结论（给决策用）

我们当前是「启发式期望值 + 精确向听/进张」，且**只有进攻**（听口宽度/牌效）。
文献里**唯一与我们约束相容、且我们明显缺的一块是「只用公开信息的风险/防守模型（押し引き）」**；
其次是 **Suphx 的「全局奖励预测」思想（按整局名次/分数调姿态，而非每局期望值）**，
以及 **把「打点/番型」并入出牌目标（Suphx 的 look-ahead 特征做的事）**——
这三条都能**不引入第三方库**地做成启发式/查表。
其余（CNN/RL/CFR/PIMC/GPU 模拟器）大多要求 torch/JAX 或联网推理，**违反运行路径零依赖硬约束，
只能进「禁入」表或降级为离线研究**。MCTS/CFR 在麻将上**不能直接套用**（见 P1 原文）。

---

## 1. 来源表

### 1.1 论文（arXiv，均经 arXiv API 取到标题/摘要/年份）

| # | 标题 | URL/ID | 年 | 类型 | 方法要点 | 报告的增益口径+样本量 | 能否搬到我方 |
|---|---|---|---|---|---|---|---|
| P1 | Suphx: Mastering Mahjong with Deep RL | https://arxiv.org/abs/2003.13590 | 2020 | 论文 | 深度 RL + **全局奖励预测**、**oracle guiding**（教师可见隐藏信息、逐步退火）、**运行时策略自适应(pMCPA)**；5 个模型分管 打/吃/碰/杠/立直；CNN 吃 34×C 通道 + **look-ahead 特征**（100+ 个、每个 34 维，编码「打某张后能做成多大点数」的概率与番数） | SL 准确率：弃牌 **76.7%**、立直 85.7%、吃 95.0%、碰 91.9%、杠 94.0%（Table 3，验证 10K / 测试 50K）；RL 消融 RL-basic < RL-1(+GRP) < RL-2(+oracle)（Fig.8，1M 局离线评估）；每 RL agent 1.5M 局、**44 GPU×2 天**；天凤特上房 5000+ 局 → 记录段位 10 段、稳定段位 **8.74** 段；**「四着率极低是稳定段位高的关键」** | **部分**：思想可搬（全局奖励/姿态、look-ahead 打点通道、运行时自适应）；网络与规模不可搬 |
| P2 | Building a 3-Player Mahjong AI using Deep RL（Meowjong） | https://arxiv.org/abs/2202.12847 | 2022 | 论文 | 定义紧凑 **2 维特征结构**；按 5 种动作各预训 1 个 CNN，弃牌模型再用 **MC 策略梯度自对弈**增强 | 监督精度与 4 人麻 AI 相当，RL 后再显著提升；**无人类对照 A/B、摘要无数字** | **部分**：2 维多通道编码可借鉴；CNN 不可搬 |
| P3 | Building a Computer Mahjong Player via Deep CNN | https://arxiv.org/abs/1906.02146 | 2019 | 论文 | 新的不完全信息**数据模型** + 专用 CNN；以**弃牌一致率(agreement rate)** 评估 | 弃牌一致率 **70.44%** vs 前人 **62.1%**（引 Mizukami & Tsuruoka 2015）；天凤 rating ≈ **1850**（来源自述，未核实） | **不能**（CNN + 训练框架）；表征思想可参考 |
| P4 | A Fast Algorithm for Computing the Deficiency Number | https://arxiv.org/abs/2108.06832 | 2021 | 论文 | 向听数快速算法，比基线**通常快 100×**，且**尊重「已知可用牌」** | 100× 加速（算法复杂度，非打法强度；来源自述，未独立复现） | **部分**：我们已有精确 shanten；profiling 显示是热点才借鉴 |
| P5 | Let's Play Mahjong! | https://arxiv.org/abs/1903.03294 | 2019 | 论文 | 定义 deficiency，给**每个 k≥1 上最大化「k 次换牌内成和概率」的最优弃牌策略** | 数学/策略分析（无数值 A/B） | **部分**：把「贪心进张」换成「k 步内到听概率」是文献认证的更强目标 |
| P6 | Method for Constructing AI Player with Abstraction to MDPs in Multiplayer Mahjong（Kurita & Hoki） | https://arxiv.org/abs/1904.07491（期刊版 https://doi.org/10.1109/tg.2020.3036471） | 2019 / 2021 | 论文 | 用**多个 MDP 抽象**构造搜索树；两种「从 MDP 推断原局面状态价值」的方法 | 摘要只给「评估了有效性 vs 当时最强 AI」，**无数字** | **部分**：抽象+搜索；实现需自建 |
| P7 | CFR-p: CFR with Hierarchical Policy Abstraction（二人麻将） | https://arxiv.org/abs/2307.12087 | 2023 | 论文 | 分层策略抽象后上 CFR；**仅两人麻将** | 摘要**无任何定量增益** | **不能**（四人 + 实时预算；且 P1 明确否定直接套用） |
| P8 | Perfect Information Monte Carlo with Postponing Reasoning（EPIMC） | https://arxiv.org/abs/2408.02380 | 2024 | 论文（CoG 2024） | determinization（PIMC）类；**推迟完美信息求解**以缓解 **strategy fusion**；理论 + 多游戏实证 | 「在被 strategy fusion 严重影响的游戏上有可观提升」，具体数值未取到 | **部分**：做 PIMC 前必读的**陷阱清单**（strategy fusion / non-locality） |
| P9 | The αμ Search Algorithm for the Game of Bridge | https://arxiv.org/abs/1911.07960 | 2019 | 论文 | anytime 启发式搜索，**假设对手掌握完美信息**；针对 PIMC 的 strategy fusion / non-locality | 摘要无数字 | **部分**：同上，算法思想 |
| P10 | Mahjax: GPU-Accelerated Mahjong Simulator（JAX） | https://arxiv.org/abs/2605.20577 | 2026 | 论文 | 全向量化 JAX 麻将环境，GPU 并行 rollout；tabula-rasa RL 可提名次 | 8×A100 上 **200 万/100 万步/秒**（无赤/有赤） | **不能作运行时**；仅离线（本机 8GB、无 JAX） |
| P11 | Abstraction Agent | https://arxiv.org/abs/2609.04303 | 2026 | 论文 | 用 **LLM** 从规则自然语言「零样本」发现战略特征 → 打分 → 筛特征 → k-means 分桶；迁移到日麻 | HUNL turn endgame 上**降低可被剥削度最多 62%**；Riichi Mahjong 上「发现的特征对应公认战略概念」（**只验证概念对应，无麻将强度增益**） | **不能/存疑**：构造期依赖 LLM（违反无网络精神）；仅作特征发现方法论 |
| P12 | A Novel Reward Shaping Function for Single-Player Mahjong | https://arxiv.org/abs/2305.04145 | 2023 | 论文 | ShangTing 奖励塑形 + 前向搜索；新 bonus 对协同型牌型给更高相对价值 | 单人对局平均 35 步和牌（10,000 局）；**1v1 模拟 1000 局平均赢 1.37** | **不能**：单人/1v1，非 4 人变体 |
| P13 | Securing Equal Share: Learning Multiplayer Symmetric Games | https://arxiv.org/abs/2406.04201 | 2024 | 论文 | 多人对称常和博弈（含 Mahjong）的 **equal-share** 目标；no-regret 算法 + 下界 | 理论 + 实验：证明 two-player 均衡概念在多人下**既不唯一也不不可剥削** | **部分**：概念提醒——多人不能照搬 2 人零和判据 |
| P14 | Mxplainer: Explain and Learn Insights by Imitating Mahjong Agents | https://arxiv.org/abs/2506.14246（期刊 https://doi.org/10.3390/a18120738） | 2025 | 论文 | **参数化搜索算法可等价转成 NN**，用来学黑箱 agent 参数 | top-3 动作预测：人类 >**92%**、AI >**90%**，优于决策树 **34.8%**；**无打法增益** | **部分**：诊断/解释工具；「参数化搜索≈NN」有方法学启发 |
| P15 | Evolutionary Optimization of DL Agents for Sparrow Mahjong（Evo-Sparrow） | https://arxiv.org/abs/2508.07522 | 2025 | 论文 | 用 **CMA-ES 优化 LSTM**（免梯度） | 胜随机与规则 agent，与 PPO 相当；**Sparrow 变体，非标准麻将** | **不能**（LSTM + 非本变体） |
| P16 | Monte Carlo Tree Search: A Review of Recent Modifications | https://arxiv.org/abs/2103.04931 | 2021/2023 | 论文（综述） | MCTS 改造与混合方法综述（99 页） | 综述，无单点增益 | **能**（参考） |

### 1.2 论文（非 arXiv，经 Crossref 核实存在）

| # | 标题 | URL/DOI | 年 | 类型 | 方法要点 | 报告增益口径+样本量 | 能否搬到我方 |
|---|---|---|---|---|---|---|---|
| C1 | Building a computer Mahjong player based on Monte Carlo simulation and opponent models（Mizukami & Tsuruoka） | https://doi.org/10.1109/CIG.2015.7317929 | 2015 | 论文（IEEE CIG） | **蒙特卡洛模拟 + 对手模型**；被引 32 | **摘要缺失（Crossref 无 abstract）⇒ 数字未找到来源** | **部分/谨慎**：对手模型 + MC 路线一手出处；MCTS 不在实时预算内 |
| C2 | MJ-DLVAT: A Deep Learning Value Assessment Technique for Mahjong（Ogami, Amano, Tsuruoka） | https://doi.org/10.1109/CoG60054.2024.10645601 | 2024 | 论文（IEEE CoG） | 麻将**局面价值评估**；参考文献含 Zinkevich 无偏估计、White 价值分析工具、「用误差率估牌手表现」(Kume 2019) | 摘要未取到；被引 0（新） | **部分**：**测量学**——用价值/误差率做对局评估，而非只看均值 |
| C3 | Bloody Mahjong playing strategy based on deep learning + XGBoost | https://doi.org/10.1049/cit2.12031 | 2021/2022 | 论文（CAAI Trans. Intel. Tech.） | **DenseNet 提特征 + XGBoost 出牌**；针对中国地方变体「血战麻将」（换三张、胡须缺一门、胡后继续） | 高维稀疏下融合优于单模型；**训练轮数少时准确率仍达 83%**（来源自述） | **部分**：中国地方变体 + 「特征+树模型」先例；训练侧用第三方库（离线） |
| C4 | Towards a Competitive 3-Player Mahjong AI using Deep RL | https://doi.org/10.1109/CoG51982.2022.9893576 | 2022 | 论文（IEEE CoG） | 3 人麻将 DRL（**arXiv 版即 P2**） | 同 P2 | **部分**：同 P2 |
| C5 | An Empirical Evaluation of Applying Deep RL to Taiwanese Mahjong Programs | https://doi.org/10.1109/ICASI57738.2023.10179593 | 2023 | 论文（IEEE ICASI） | 台湾麻将 DRL 实证 | 摘要未取到 | **部分**：中文变体 DRL 先例 |

### 1.3 开源项目 / 工程（★为 GitHub stars 快照，会变动）

| # | 项目 | URL | ★ | 语言 / 许可 | 架构与可借鉴点 | 能否搬到我方 |
|---|---|---|---|---|---|---|
| G1 | Mortal（凡夫） | https://github.com/Equim-chan/Mortal | 1571 | Rust / **AGPL-3.0** | 日麻最强开源 AI 之一；**权重需经 Discord 领取**（README 明示） | **不能直接搬**（Rust+DRL；AGPL 传染；运行时零依赖）。可作**离线对照基线** |
| G2 | kanachan | https://github.com/Cryolite/kanachan | 345 | Python / 无 | **无人工特征**（牌作 token+embedding，手牌/弃牌/副露作集合/序列）；**课程微调**（复用 encoder、只换 decoder）；README 给数据量：**雀魂金之间约 6500 万局 vs 天凤凤凰 11 年 1700 万局**；「模型早期已自发学到 dora/断幺九/现物/筋/流局满贯」 | **不能**运行时；**能**作「表征该多激进」的对照与数据量级参照 |
| G3 | mjx | https://github.com/mjx-project/mjx（论文 https://doi.org/10.1109/CoG51982.2022.9893712） | 210 | C++ / MIT | 日麻模拟器，**比 mjai 快 100×**、gym-like API、gRPC 分布式 | **不能**运行时；离线评测/自对弈成熟参照 |
| G4 | mahjong-helper | https://github.com/EndlessCheng/mahjong-helper | **2349** | Go / MIT | 定位即「**牌效 + 防守 + 记牌**」——工程界把**防守当与牌效并列的一等公民** | **部分**：防守项建模的启发式参照（只读公开描述） |
| G5 | akochan | https://github.com/critter-mj/akochan | — | C++ | 日麻非 DL 路线 AI | **不能**运行时；离线对照 |
| G6 | Akagi | https://github.com/shinkuan/Akagi | 1064 | Rust / Apache-2.0（+ Commons Clause） | **MITM 抓雀魂 WebSocket → 转 mjai → 送 Mortal 出动作**；README 明示封号风险 | **不能**（违反 §3：不碰平台写操作、不做协议中间人） |
| G7 | MahjongCopilot / MajsoulAI / MahjongMaster_AI | https://github.com/latorc/MahjongCopilot (1078, Python, GPL-3.0), https://github.com/moxcomic/MajsoulAI, https://github.com/halfofsever/MahjongMaster_AI | — | 多为 Mortal 封装 + 协议/画面桥 | 雀魂实时 AI 指导 | **不能**（同上；非算法贡献） |
| G8 | Chinese-Standard-Mahjong（北大 AILab） | https://github.com/ailab-pku/Chinese-Standard-Mahjong | 136 | C++ | 国标麻将联赛框架；配套 **PyMahjongGB**（番种计算器 https://github.com/ailab-pku/PyMahjongGB） | **部分**：番种/役型计算与「按番计分」工程做法；非杭州规则 |
| G9 | botzone-mahjong-environment | https://github.com/ccr-cheng/botzone-mahjong-environment | — | Python | Botzone 平台国标麻将 Python 环境 | **部分**：国标纯 Python 环境参照 |
| G10 | MahjongRepository/mahjong | https://github.com/MahjongRepository/mahjong | — | Python | 向听、手牌代价（hand cost）、和了判定等基础库 | **部分**：向听/代价实现对照 |
| G11 | esrrhs/majiang_algorithm | https://github.com/esrrhs/majiang_algorithm | 481 | Java / MIT | 毫秒级和牌/听牌计算 + AI 弃牌 | **部分**：算法对照 |
| G12 | mjai-reviewer | https://github.com/Equim-chan/mjai-reviewer | 1247 | Rust / Apache-2.0 | mjai 协议牌谱复盘 | **不能**运行时；离线判读参照 |
| G13 | rlcard | https://github.com/datamllab/rlcard（论文 https://arxiv.org/abs/1910.04376） | 3558 | Python / MIT | 卡牌 RL 工具箱，含简化 Mahjong 环境（**简化变体，非杭州/日麻**） | **部分**：环境/算法积木；规则不对口 |

> 检索路径：`api.github.com/search/repositories?q=mahjong+ai`（459 个结果）、
> `q=mahjong+dataset+OR+replay`（10007 个结果）、`q=MahjongGB+OR+"Chinese+Standard+Mahjong"`（50 个结果）。

### 1.4 博客 / 二手来源（仅定位用，不作为技术结论）

| # | 标题 | URL | 年 | 类型 | 要点 | 能否搬到我方 |
|---|---|---|---|---|---|---|
| B1 | 微软超级麻将 AI Suphx | https://www.microsoft.com/en-us/research/articles/mahjong-ai-suphx/ | 2019 | 官方博客 | Suphx 项目简介（与 P1 同源） | 仅背景 |
| B2 | [伏羲讲堂] 微软麻将 AI Suphx 解读 | https://zhuanlan.zhihu.com/p/139764115 | 2022 | 博客 | 中文解读；「44 张 GPU 耗时 2 天」与 P1 一致 | 仅背景 |
| B3 | NAGA 相关中文社区帖（NGA / 贴吧 / 攻略站） | https://ngabbs.com/read.php?tid=39353003 等 | — | 论坛 | 只有使用体验与「贵不贵」，**无技术架构、无一手评测口径** | 不可引用 |

---

## 2. Top-5 候选实验（按「预期增益 × 成本」排序）

> 判据沿用本仓纪律：`|t|≥1.96 且符号一致`；机制量优先；**四座位旋转 + 同批牌 + 两种子 + 按房分层**。
> 算力成本按我方口径估：一个 120 场 job 在 CPU 竞争下约 115–180 分钟（见 `notes/THREAD.md` 2026-09-28 22:20 A 条目）。
> **效应量凡标「估计」者均为我方估计，不是文献数字。**

### T1（首选）**只用公开信息的「放铳风险 / 押し引き」启发式**
- **依据**：C1（**MC + 对手模型**是一手路线）；C1 参考文献含 Wagatsuma 2014「用 SVR 估弃牌危险度」（IPSJ SIG GI）；G4（★2349 项目把防守当一等公民）；P1（Suphx **极低四着率**是其强项来源）。
- **可检验假设**：在「我方已听牌或一向听、但某候选牌是对手高危舍牌」的局面，用**仅公开信息**
  （各家已舍牌、副露、牌河、剩余牌数、庄位、财神状态）估「被铳风险」，在「进攻收益」与「放铳损失」间做阈值/线性权衡，能降**被铳率**并改善**整局名次分**。
- **最少样本量**：名次分尺度 **≥120 配对场 × 2 种子**（四座位旋转）；被铳率是频率量、功效更高，可先看它。
- **预期效应量（估计）**：放铳率 −0.3 ~ −1.0 个百分点；名次分 +0.05 ~ +0.25（小）。
- **算力成本（估计）**：CPU 1–2 小时（自对弈），不占 GPU。
- **kill_criteria**：① 两种子被铳率不降 → 放弃；② 名次分不同向 → 放弃；③ 胡率同步下降且名次分仍不显著 → 只是变保守 → 放弃。
- **硬约束**：✅ 合规——**只从公开牌河/副露推断**，**绝不含任何读取/推断对手手牌的路径**（§3 红线）。
- **注意（A 22:20 机制发现）**：爆头率与胜负**反向**，**不追爆头**；本项目标是**降放铳、保名次**。

### T2 **整局姿态 + 打点（Suphx「全局奖励预测」+ look-ahead 的启发式版）**
- **依据**：P1（全局奖励预测按**整局名次**调姿态；look-ahead 特征显式编码「打某张后能做成多大点数」的概率与番数）；C3（地方变体直接用树模型学出牌）。
- **可检验假设**：① 引入「整局累计分数 → 姿态」（领先保守、落后激进）能改善**整局名次分**；
  ② 在 ukeire 次排序之上用「期望番数/期望得分」做**同进张下的次级排序**，增益**集中在财神多（≥2 张）的局面**。
- **最少样本量**：**≥120 配对场 × 2 种子**；姿态项很可能只在「末局领先」子样本更明显（做子样本诊断）。
- **预期效应量（估计）**：名次分 +0.10 ~ +0.30（噪声底附近，需两种子同向）；打点的机制指标（同进张候选的期望番数）应明显。
- **算力成本（估计）**：CPU 2–6 小时（纯标准库）。
- **kill_criteria**：① 两种子名次分不同向 → 杀；② 只在无意义子样本上有效 → 杀；③ 打点项若伴随「均番下降/爆头率下降」，按 A 22:20 判读（别追爆头）。
- **硬约束**：✅ 合规（只用自身分数与公开局面）。

### T3 **表征升级：多通道编码 + 财神/番型通道（离线训练，纯 Python 导出）**
- **依据**：P1（34×C 多通道 + look-ahead 通道）；P3（弃牌一致率 70.44% 的表征收益）；P2（2D 张量）；G2（kanachan 无人工特征路线）；本线自述「瓶颈在表征」「MLP 只在无财神局面有点增益」。
- **可检验假设**：① 把特征从聚合 scalar 换成 Suphx/Meowjong 式**多通道 one-hot / 2 维结构**，离线留出集**排序指标**提升；
  ② MLP 只在无财神局面有效是因为**表征缺 财神/番型/得分 通道**；补上后同一族简单模型在**有财神局面**也超启发式。
- **最少样本量**：离线留出集（沿用 T1/T2 既有配对口径）；离线数据 ≥50 万决策点；A/B 阶段每臂 ≤2000 配对局 / 种子 × 2 种子。
- **预期效应量（估计）**：比「换模型家族」（已证 ~0）更可能有增益；只有**离线留出指标先动**才值得做自对弈 A/B。
- **算力成本（估计）**：离线训练 2–10 小时（dev 组用 numpy/sklearn/torch）；导出纯 Python 后 CPU < 1 小时。
- **kill_criteria**：① 留出集排序指标（有财神层）不超基线 → 杀；② 导出后纯 Python 输出与训练框架不一致 → 杀（**先用已知答案小样本验仪表**）；③ 模型缺失不能回退启发式 → 违反 §5 工程约束 → 修好再谈。
- **硬约束**：训练侧可用 numpy/torch（dev 组、离线），**产物必须纯 Python 可推理、确定性、无网络**；推理输入仅公开信息。

### T4 **向听/进张算法剖析 + k 步到听（P4 / P5）**
- **依据**：P4（快速 deficiency，**尊重可见牌**，快约 100×）；P5（**k 步内到听概率**的最优弃牌策略）。
- **可检验假设**：① 若真机决策延迟受限，更快的 deficiency 算法可**在同样时限内扩候选面/加深搜索**；
  ② 当前 `ukeire` 是 1 步代理；用「打完仍能在 k 步内到听的**概率**」做排序，在**非财神与财神两层**都提升到听速度。
- **最少样本量**：**先 profiling（不是 A/B）**，确认 shanten 是热点后才做 A/B：每臂 ≤2000 配对局 / 种子 × 2 种子；
  机制量（到听摸序、进 1 向听步数）自带更高功效，可先看机制门。
- **预期效应量（估计）**：到听摸序 −0.05 ~ −0.20 步（我方 v4 里 tenpai-wait-6 的机制量就是 6.15→5.87）；名次分 +0.10 ~ +0.30，但**很可能与 v3 增量重复**（v4 已证「向听 1 候选面放大」不成立）⇒ 必须先做机制隔离。
- **算力成本（估计）**：profiling 近乎零；k 步版 CPU 3–8 小时（注意 v2 已带 0.6 秒墙钟上限，k 步要控预算）。
- **kill_criteria**：① profiling 显示 shanten 非热点 → 不做 k 步；② 相对 **v3**（不是相对 heuristic）增量不显著 → 杀；③ 趋势递减（像 v4 的 0.329→0.160→0.050）→ 杀；④ 到听摸序无变化 → 杀。
- **硬约束**：✅ 合规（纯算法；只用公开信息）。

### T5 **有界 determinization / PIMC 期望搜索（离线研究优先）**
- **依据**：C1（MC 模拟 + 对手模型）；P8 + P9（determinization 的**已知陷阱** strategy fusion / non-locality 及缓解法）；**P1 明确**「麻将出牌顺序不规则，无法建常规博弈树，MCTS/CFR 不能直接套用」。
- **可检验假设**：在**听牌 / 向听 1** 的少量关键决策点上，用「从已知牌分布**采样**对手手牌 + 牌墙 → 快速 rollout」估候选期望得分，能改进 ukeire 排序（尤其能发现「进张多但实际和率低」的陷阱）。
- **最少样本量**：先做**机制隔离**（固定局面上比较 rollout 排序 vs ukeire 排序的 regret），样本 ~200–500 个决策点；机制门通过后再每臂 ≤2000 配对局 / 种子 × 2 种子。
- **预期效应量（估计）**：regret-to-best-wait 下降 5–20%（我方 v3 现 regret≈0.0%，故这一路更可能改善**打点/安全**而非听口宽度）；名次分很可能落在噪声内。
- **算力成本（估计）**：**高**，CPU 10–40 小时/扫描（每决策点 N 次 rollout；纯 Python 无 numpy 时更慢）。
- **kill_criteria**：① 机制门（rollout 排序相对 ukeire 的 regret）无改善 → 杀，不做 A/B；② A/B 不显著 → 杀；③ 出现 strategy fusion 症状 → 记录并回退到 P8/P9 缓解法；④ **耗时超墙钟预算**（我方有 0.6 秒级上限先例）→ 不可上线 → 杀。
- **硬约束**：✅ 合规——采样对手手牌是**从公共信息分布采样**，不是读取真实手牌；**绝不能用真实牌墙**（见 §3 X1）。

### 暂缓项（不进 Top-5）
- **离线用 Mahjax / mjx 做大规模自对弈数据生成**：GPU 并行 rollout 能低成本产出数据供 T3 训练，但
  **本机 8GB、无 JAX 环境，A100 级数字不可得** ⇒ **暂缓**；仅离线，**不得**进入运行路径。
- **CFR / P7 / P11 等强依赖第三方框架或联网的方向**：见 §3 禁入表。

---

## 3. 「禁入」表（违反硬约束 ⇔ 列为禁入，不得实施）

| # | 方案 | 来源 | 违反的硬约束 | 说明 |
|---|---|---|---|---|
| X1 | 用日志恢复的**真实牌墙**做 determinization / 搜索（如 kanachan 的 paishan） | https://github.com/Cryolite/kanachan | ③ 推理输入仅限公开信息 | 恢复牌墙＝读全局隐藏信息。**只能**用「从已知牌分布采样」的假牌墙（T5）。kanachan README 自注该产物**只能**用于测试 |
| X2 | 运行路径 import 深度学习/科学计算框架（torch/jax/numpy） | P1/P2/P10/G1/G2/P10 | ① 零第三方依赖；② 纯 Python 可推理 | `src/majiang/**` 只许标准库。把 Suphx/Meowjong/Mortal 权重直接塞进运行路径的方案一律禁 |
| X3 | Suphx / Meowjong / Mortal 的**神经网络推理**直接上线 | https://arxiv.org/abs/2003.13590 等 | ① ② | 需 torch/ONNX 运行时；除非导出为纯 Python 且确定性（成本极高，未见先例） |
| X4 | 运行时调用外部推理服务（NAGA 在线、Mortal 服务、**LLM**） | P11；https://ngabbs.com/read.php?tid=39353003 | ① ② ④ 无网络 / 确定性 / 官方未声明服务 | 推理必须离线确定；LLM 做工时特征/决策＝联网 + 不确定 + 未声明 |
| X5 | 把 **oracle agent**（可见对手手牌与牌墙）当运行时决策器 | https://arxiv.org/abs/2003.13590 | ③ 推理输入仅限公开信息 | oracle guiding **只在训练期**当教师（原文 §3.3），最终部署策略只用可观测信息；唯一允许写法是「训练期 oracle 蒸馏，导出物只吃公开特征」 |
| X6 | 用对手真实手牌训练/评估**运行路径**（标签泄露进线上特征） | 通用 | ③ | 对手手牌**只允许**在离线数据生成当标签；导出物与推理路径不得含任何读对手手牌路径 |
| X7 | 用 Mortal 代码（AGPL-3.0）/ Akagi（+Commons Clause）**复制或改写到本仓** | G1 / G6 | 许可证传染 + 零依赖 | AGPL 会污染仓库；仅可**读文档取思想**，不可抄代码 |
| X8 | 协议中间人 / 抓包 / 自动对局脚本（Akagi 式） | https://github.com/shinkuan/Akagi | 我方 §3 运营红线 | 不碰平台写操作、不抢令牌配额、不干预平台进程 |
| X9 | CFR / MCTS 直接做实时决策 | P7 / P16 | 实时时限 + 四人复杂度 | P1 §1 明确指出四人麻将无法建常规博弈树；P7 只做二人 |
| X10 | 依赖 gRPC / JAX / C++ 扩展做**运行时**推理 | G3 / P10 | ① ② | 只可**离线**评测/自对弈，且不得成为运行路径一环 |
| X11 | 直接打包/分发第三方权重 | G1 | ② ④ + 许可 | Mortal 权重经 Discord 领取，README 明示；AGPL + Commons Clause |

### 灰区（未必禁，但必须限定写法）
- **G-a 训练期用隐藏信息**：oracle guiding / 离线数据里的对手手牌标签 / 危险牌模型标签——
  这些**离线**使用不违反「推理输入仅限公开信息」。**限定**：① 导出物只接受公开特征；② 推理期无隐藏输入；
  ③ 报告须写明「训练用了什么标签、推理只用什么」；④ 能通过一次只读审计。
- **G-b 采样对手手牌**：determinization 采样的是**分布**不是真实手牌 ⇒ 合规；**不能**用真实牌墙采样（X1）。
- **G-c 课程微调 / encoder 复用**（G2）：架构技巧本身合规；实现若用 torch，只能活在离线训练侧。

---

## 4. 「无法核实的断言」清单（宁可少报，不得编造）

1. **杭州麻将（财神百搭、爆头、财飘）的 AI 论文**：arXiv（`all:mahjong` 27 条、`ti:"mahjong"` 15 条、`all:"riichi mahjong"` 3 条、`all:"japanese mahjong"` 2 条、`all:"chinese standard mahjong"` 0 条）+ Bing + GitHub 均**未找到来源**。⇒ 不做任何外推。
2. **Mortal 的强度数字**：只有 README 描述与社区口碑，**未找到受控 A/B 或公开基准** ⇒ **未找到来源**。
3. **C1（Mizukami 2015）的增益数字**：Crossref 无 abstract、未抓全文 ⇒ **数字未找到来源**。
4. **「押し引き带来 X% 提升」的具体数字**：公开项目未给受控量化 ⇒ **未找到来源**；本报告对该方向的价值判断基于「社区把防守当一等公民」+ 我方确无此层，属**推断**，需自测。
5. **Suphx 各技术的单独消融**：摘要/正文只给 RL-basic/RL-1/RL-2 定性排序与 Figure 8 图，**具体分位数本次未取到** ⇒ 标注「来源自述」。
6. **NAGA 的技术架构与强度数字**：只找到社区/B3 级来源，**未找到论文级一手说明** ⇒ **未找到来源**。
7. **「爆头 / 财飘」的公开讨论**：同 #1，**未找到**公开一手来源；A 22:20 的「爆头是宽听口副产品」是**我方真机数据**，不是文献结论。
8. **P4 的 100×、P3 的 70.44% / 1850、C3 的 83%**：均为**来源摘要自述，我方未独立复现**。
9. **Wagatsuma 2014「Estimating risk of discarded tiles in Mahjong using SVR」**：仅经 C1 参考文献表间接得知标题（IPSJ SIG GI），**未找到 DOI 或正文**。
10. **Kume 2019「Estimating the performance of mahjong players using error rates」**：仅经 C2 参考文献表间接得知（IPSJ SIG Tech Report）⇒ **未找到一手来源**。
11. **kanachan「能击败 NAGA/Suphx」**：README 明写这是**目标**（"The goal of this project is…"），**不是已公布结果**。
12. **P2 自称的 sanma state-of-the-art**：自述，摘要**无数字**，未找到第三方复现。
13. **`web_search` 不可用**：本机工具禁用 ⇒ 本报告**不覆盖**非 arXiv/Crossref/GitHub/Bing 索引内的中文期刊/学位论文/日文社区帖，**已知覆盖缺口**。
14. **P11 在 Riichi Mahjong 上的强度增益**：论文只验证「发现的特征对应公认战略概念」，**无麻将强度数字**。

---

## 5. 复现路径（证据可复查）

```bash
# arXiv 全量（27 条）
curl -sSL "https://export.arxiv.org/api/query?search_query=all:mahjong&start=0&max_results=40"
curl -sSL "https://export.arxiv.org/api/query?search_query=ti:%22mahjong%22&start=0&max_results=50"
# 关键摘要（按 id 批量）
curl -sSL "https://export.arxiv.org/api/query?id_list=2003.13590,2202.12847,1906.02146,2307.12087,2506.14246,2605.20577,2108.06832,2508.07522,1903.03294,1904.07491,2408.02380,1911.07960,2609.04303,2406.04201,2305.04145,2103.04931"
# GitHub 生态
curl -sSL "https://api.github.com/search/repositories?q=mahjong+ai&sort=stars&order=desc&per_page=30"
curl -sSL "https://api.github.com/search/repositories?q=MahjongGB+OR+%22Chinese+Standard+Mahjong%22"
curl -sSL "https://raw.githubusercontent.com/Cryolite/kanachan/master/README.md"
curl -sSL "https://raw.githubusercontent.com/mjx-project/mjx/master/README.md"
# Crossref（对手建模 / 价值评估 / 地方变体）
curl -sSL "https://api.crossref.org/works/10.1109/cig.2015.7317929"
curl -sSL "https://api.crossref.org/works/10.1109/cog60054.2024.10645601"
curl -sSL "https://api.crossref.org/works/10.1049/cit2.12031"
# Suphx 全文
curl -sSL "https://arxiv.org/html/2003.13590v2"
```

**落档**：agent-c，2026-09-28 23:2x（合并版，保留 `605a982` 全部已验证来源）。
只读外网 + 只写本文件；未改仓库任何既有文件、未跑训练、未碰 src/** / tools/** / verify/** / notes/**。
