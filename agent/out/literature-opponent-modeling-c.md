# 外部文献调研：麻将 AI 的对手建模 / 手牌预测 / 出牌决策

**作者**：agent-c（小龙虾）· 2026-09-29 · 用户 18:58 窗口内自主执行
**方法**：`web_search` 不可用（X-Access-Token 无效）⇒ 改用 **arXiv API**（`export.arxiv.org/api/query`）+ `web_fetch`。
**局限先声明**：arXiv API 的查询解析会**擅自改写布尔式**（实测 `abs:riichi AND abs:reinforcement learning`
被解析成 `abs:riichi AND (abs:reinforcement OR all:learning)`），所以「命中数」只能当**下界**，
不能当「文献里只有这么多」。以下每条都标了「原文依据 / 我的推论」。

---

## 0. 一个诚实的覆盖缺口（先记）

- 查询 `abs:"opponent modeling" AND abs:mahjong` ⇒ **totalResults = 0**。
  ⇒ **arXiv 上没有以「对手建模 + 麻将」为题的论文**（至少摘要词层面没有）。
- 查询 `all:mahjong AND all:opponent model`（宽松匹配）⇒ 11 条，但**无一条**是真做「预测对手暗手/听牌」的。
- **推论（我的判断）**：我们那条「逐牌种占用估计」在公开文献里**没有直接对标**。
  这**不是**「方向错」的证据，而是：**这条具体路径要靠我们自己立判据、自己验证**——
  与 B' 18:40 说的「Oracle guiding 借来估占用是我们的挪用、文献无直接先例」一致，且范围更大。

---

## 1. Suphx（arXiv:2003.13590，MSRA）— 4 人日麻，天凤超 99.99% 人类

（B' 18:05 已整理，此处只补**与占用/顺序相关**的三点原话定位）
- **look-ahead features**：对每个候选弃牌，显式计算「摸若干张能成哪些胡牌型」的**概率 × 番**，作为特征；
  **计算时忽略对手行为**（只算自己）。⇒ **依据：原文**。
- **Oracle guiding**：先训一个能看**全信息**（含对手手牌+牌墙）的「先知」，再逐步撤掉完美信息、蒸馏成只读公开信息的 agent。
  ⇒ 目的在原文是**解决信用分配 / 加速训练**，**不是**「估对手暗手」。⇒ **依据：原文**（B' 18:40 判断一致）。
- **pMCPA**：在线用蒙特卡洛随局内公开信息**微调**已训策略。⇒ 原文；**需要大量模拟**。

## 2. Meowjong（arXiv:2202.12847，Zhao & Holden）— 3 人麻将 deep RL ★与「顺序编码」最相关

- **原文**：「we define an informative and compact **2-dimensional data structure** for encoding the
  **observable information** in a Sanma game」；为 5 个动作各预训一个 **CNN**，再对主模型（弃牌）做自对弈 RL
  （Monte Carlo policy gradient）。
- **对「顺序/时间」问题的意义（我的推论，较可靠）**：
  它把**可观测信息（含各家弃牌）编成 2D 张量**喂 CNN——**顺序是靠张量的「通道/位置」显式编码的**，
  不是靠 RNN/Transformer 的时序机制。⇒ **支持「显式编码 + 卷积/树模型」这条路线**，
  与我们「加 recency 特征」同族（我们是显式编码 + 树；它是显式编码 + CNN）。
- **它没有做的事**：**不预测对手暗手**（CNN 输入是「可观测信息」，对手手牌只在训练标签/模拟里）。
  ⇒ 也就是说，**连 SOTA 的 sanma RL 也没走「在线估对手暗手」这条路**。

## 3. Gao et al.（arXiv:1906.02146，2019）— CNN 评估函数，监督学习

- 弃牌一致率 **70.44%**（前 SOTA 62.1%，Mizukami & Tsuruoka 2015）；组合 5 个预测网络后在天凤 **rating ≈1850**。
- **数据**：人类高手对局日志（监督）。⇒ **必须有「谁算强」的标签**（B' 18:40 判断一致）。
- 注意与 Suphx/Meowjong 的分工：Gao 是**纯监督**（学人），Suphx/Meowjong 是**监督预训 + 自对弈 RL**。

## 4. Mahjax（arXiv:2605.20577，2026）— GPU 加速立直麻将模拟器（新）

- JAX 全向量化，8×A100 上 **no-red 2M steps/s、red 1M steps/s**；支持 **tabula rasa（从零 RL）**，
  对齐 AlphaZero 谱系；原文特别指出「prior research heavily relied on supervised learning from human play logs
  to pre-train the policy」。
- **对我们的意义（推论）**：**从零 RL 的瓶颈在模拟器吞吐，不在算法**。我们本机 16 核 + RTX 5060Ti 8G、
  自对弈速率约 3.5 s/场 ⇒ 与「百万 step/s」差 5–6 个数量级。
  ⇒ **「不靠自对弈 RL 也能提分」的方向（特征/评分函数/监督）更适配我们的算力**，
  这条与我 18:26 的容量扫描（容量非瓶颈）合起来看：**我们的瓶颈是「信息/归纳偏置」，不是「算力」。**

---

## 5. 三条与我们直接相关的可执行结论（标强度）

| # | 结论 | 来源强度 |
|---|---|---|
| 1 | **「顺序证据」不必须用序列模型**：Meowjong 用 2D 张量 + CNN 显式编码弃牌；Gao 用 CNN。⇒ 我们「显式 recency 特征 + 树」路线**有同类先例**（虽非同一模型族） | 原文 + 推论 |
| 2 | **SOTA 麻将 AI 都不在线估对手暗手**：Suphx 忽略对手算 look-ahead；Meowjong 只编码可观测信息；Gao 纯监督弃牌。⇒ 我们这条「占用估计」**是文献空白**，要么自己立判据，要么改走 Suphx 的 Oracle guiding（离线蒸馏） | 原文 + 我的判断 |
| 3 | **我们该走「低算力高归纳偏置」路线**：从零 RL 需百万 step/s 级模拟器（Mahjax），我们差 5–6 个数量级 ⇒ 优先「显式特征 / 评分函数 / 监督」，**不优先自对弈 RL 扩规模** | 原文数据 + 推论 |

## 6. 未找到 / 待补

- **arXiv 无「麻将对手建模」专文**（`abs:"opponent modeling" AND abs:mahjong` = 0 条）。
- **未找到**「对手弃牌序列时间衰减 → 显式特征 → GBDT」的**直接**先例（Meowjong 是最接近的，但用 CNN）。
- `web_search`（AIGW 网关）**凭据无效**，本轮无法用常规搜索引擎；GitHub/博客类来源**未覆盖**。
  若要补，需先修 `websearch` 的 X-Access-Token，或改走 arXiv/GitHub API。

## 7. 复现

```bash
# 关键词查询（atom XML）
curl -s 'http://export.arxiv.org/api/query?search_query=abs:riichi+AND+abs:reinforcement+learning&max_results=20'
# 具体论文摘要页
# Suphx   https://arxiv.org/abs/2003.13590
# Meowjong https://arxiv.org/abs/2202.12847
# Gao2019  https://arxiv.org/abs/1906.02146
# Mahjax   https://arxiv.org/abs/2605.20577
```
