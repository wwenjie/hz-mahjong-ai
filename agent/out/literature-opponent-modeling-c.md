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

---

# 附录：子代理深挖版（同一条线 agent-c，2026-09-29 19:0x，子代理 mahjong-ai-literature-c）

> 上面正文是 agent-c 主会话 18:58 窗口的落档（已 commit `fc325ed`），**原样保留**。
> 下面是同一任务由子代理做的**更详细**版本：新增了 Suphx/Gao/Meowjong 的**逐段原文引文定位**、
> kanachan/Mortal 项目证据、Mxplainer 树模型对照数字、以及**检索通道实测失效记录**。
> 两版结论一致（SOTA 麻将 AI 都不在线估对手暗手；「时间衰减→显式特征→GBDT」无先例）。
> 若两者细节冲突，以本附录的**带引号原话**为准。

## 外部调研：麻将 AI 的对手建模 / 手牌预测 / 出牌决策训练方法

- 调研人：小龙虾（agent-c，子代理 mahjong-ai-literature-c）
- 时间：2026-09-29
- 背景与动机：我方用 GBDT 预测「三个对手的暗手占用」（每牌种 0..4 张），23 维静态聚合特征；
  加时间衰减特征后仍打不过单变量「已见张数」基线；容量已确认不是瓶颈（max_iter 50× 仍平）。
  本调研要回答：外部论文里「对手暗手/听牌估计」主流形态是什么、有无把弃牌序列时间衰减做成显式特征
  喂树模型的先例、以及 Suphx 的 oracle guiding 到底是干什么的。
- **检索通道**：`web_search` 无 provider（本环境已确认）；`export.arxiv.org/api` 在本次会话中**持续 429**
  （多次退避 40s~200s 后仍 429），Semantic Scholar 429、OpenAlex 503。
  实际可用通道：**`arxiv.org/search/?query=...` HTML 页**（返回 200）、**`arxiv.org/pdf/<id>` 全文**、
  **Crossref API**（200）、**CrossRef/raw.githubusercontent.com**（200）、GitHub API（前段 200，后段限流）。
  见文末「覆盖缺口与方法学警告」。

---

### 0. 一句话结论（先给答案）

- **(a)** 麻将/不完美信息牌类游戏的 SOTA 线（Suphx / Gao / Meowjong / kanachan / Mortal）**没有一家用树模型或
  显式「对手暗手占用」预测器**；主流是**把全场可观测信息编码成二维张量（34×N 或 token 序列），
  交给 CNN/Transformer/ResNet 的决策网络，让「读牌」隐含在网络内部完成**。**未找到**任何「树模型 + 显式 recency 特征」
  做对手暗手估计的论文先例（无论支持或否定）。
- **(b)** **未找到**把「对手弃牌序列的时间衰减」编成显式特征喂 GBDT/树模型的先例。最接近的是
  **Gao et al. 2019（1906.02146）把最近 6 巡离散地编码为独立 plane（硬窗口，不是指数衰减），且喂的是 CNN 不是树模型**——
  它给了「时间近的弃牌信息量更大、必须单独编码」的**原文依据**，但不是树模型先例。
- **(c)** Suphx 的 **oracle guiding 是「训练加速」技术，不是对手建模技术**：先用一个能看见
  对手暗手+牌墙的「先知 agent」跑 RL，再通过 Bernoulli dropout 把完美特征比例 γ 从 1 衰减到 0，
  退化为只用公开信息的正常 agent。**未找到**其他论文拿 oracle guiding 做对手建模。

---

### 1. 已知论文（逐条确认原话）

#### 1.1 Suphx — arXiv:2003.13590（v2, 2020-03-30）

来源：https://arxiv.org/abs/2003.13590 ；全文 PDF https://arxiv.org/pdf/2003.13590v2 （本地已抽取，28 页）

**① Oracle guiding（§3.3，原文）**
- 原文选段：「we introduce an oracle agent, which can see all the perfect information about a state:
  (1) private tiles of the player, (2) open (previously discarded) tiles of all the players, (3) other public
  information such as the accumulated round scores and Riichi bets, **(4) private tiles of the other three players,
  and (5) the tiles in the wall**. Only (1)(2) and (3) are available to the normal agent, while (4) and (5) are
  additional "perfect" information that is only available to the oracle.」
- 原文选段：「With the (unfair) access to the perfect information, the oracle agent will easily become a master
  of Mahjong after RL training.」
- 原文选段：「simple knowledge distillation does not work well: it is difficult for a normal agent, who only has
  limited information access, to mimic the behavior of a well-trained oracle agent, who is super strong and far
  beyond the capacity of a normal agent.」
- 机制（原文）：对完美特征加 Bernoulli dropout，`P(δ_t(i,j)=1)=γ_t`，**γ_t 从 1 逐渐衰减到 0**；
  γ_t=0 时模型从 oracle agent 变成 normal agent；之后继续训练，学习率降到 1/10，
  并丢弃重要性权重过大的 state-action 对（否则连续训练不稳、不再提升）。
- 用途定性：**「To speed up the RL training」**（原文措辞）——即训练加速/信号增强，**不是**用来建模对手。
- 标注：**原文依据**。

**② Look-ahead features（§3.1，原文）**
- 原文选段：「we design look-ahead features to encode the rich possibilities of different winning hands and
  their winning scores of the round, as a support to the decision making of our RL agent.」
- 抽取方式（原文）：「(1) We perform depth first search to find possible winning hands.
  (2) **We ignore opponents' behaviors and only consider drawing and discarding behaviors of our own agent.**
  With those simplifications, we obtain 100+ look-ahead features, with each feature corresponding to a
  34-dimensional vector.」
- 含义（原文示例）：「a feature represents whether discarding a specific tile can lead to a winning hand of
  12,000 round score with replacing 3 hand tiles by tiles drawn from the wall or discarded by other players.」
- 标注：**原文依据**。注意：look-ahead 是**对「自己未来成和」的展望**，不是对手暗手预测；
  且**显式忽略对手行为**。

**③ pMCPA（parametric Monte-Carlo policy adaptation，§3.4，原文）**
- 动机（原文）：顶级人类会因初始手牌好坏而改风格（好牌激进多赢、差牌保守少输），
  所以「adapt the offline-trained policy in the run time」。
- 三步（原文）：(1) Simulations — 「Randomly sample private tiles for the three opponents and wall tiles from
  the pool of tiles excluding our own private tiles, and then use the offline-trained policy to roll out and
  finish the whole trajectory.」生成 K 条轨迹；(2) Adaptation — 用轨迹做梯度更新微调；(3) Inference — 用微调后策略打这一局。
- 关键性质（原文）：「the policy adaptation is performed for each round independently」；K 不需要很大，
  因为是参数化方法，能泛化到未访问状态。
- 标注：**原文依据**。注意 pMCPA 里「随机采样对手暗手」**只是 rollout 的采样起点**，**不是**要预测对手暗手。

#### 1.2 Gao et al. 2019 — arXiv:1906.02146（v2, 2019-06-05）

来源：https://arxiv.org/abs/1906.02146 ；全文 PDF https://arxiv.org/pdf/1906.02146v2

**摘要要点（原文）**：提出新的数据模型表示桌面可用不完美信息；CNN 训练；以「弃牌一致率（agreement rate）」
为基准，测试达 **70.44%**，此前 SOTA 为 **62.1%**（Mizukami & Tsuruoka 2015）。
程序在 Tenhou 达 rating ≈ **1850**。
标注：**原文依据**。

**对问题 (b) 最关键的原话（§IV，原文）**：
- 「Past actions that players made also have great influence on present choice making. People can usually learn
  what other players need and not need from their past discard tiles.」
- 「Although we already have planes of players' discard tiles already, it can only show what tiles players already
  discarded, **but the discard orders cannot be learned. The closer the discard tiles are, normally more
  information they possess.**」
- 「After experiments, we include the **last six rounds' information** including own hand tiles, four players'
  discard tiles and four players' stealing tiles which means **nine planes for each round**.」
- 即：**「时间近的弃牌信息量更大」+「必须把最近若干巡单独编码」在同一篇论文里被明确写下**，
  但喂的是 CNN（34×4 的 plane，共 86 planes），**不是树模型，也不是指数衰减权重**。
- 额外 ablation（原文）：「Compared with models without riichi information planes contained in, the agreement
  rate of discard tile on test dataset **raises over 1%**」；含 dora indicator plane 而非直接编码 dora
  「the agreement rate accuracy even **raises about 1%**」——说明**显式把对手状态/关键牌信息编进输入**确有增益。
- 标注：**原文依据**。

#### 1.3 Meowjong — arXiv:2202.12847（v3, 2022-02-25）→ 重点：可观测信息编码

来源：https://arxiv.org/abs/2202.12847 ；全文 PDF https://arxiv.org/pdf/2202.12847v3 ；
代码/README：https://github.com/VictorZXY/Meowjong
（发表版标题 *Towards a Competitive 3-Player Mahjong AI using Deep RL*, IEEE CoG 2022）

**2D 数据结构（原文）**：
- 「we use a **34× 366 array** to represent a state」——34 行 = 34 种牌；**366 列 = 22 类特征**。
- 「we use 366 columns in total to represent 22 features, as listed in TABLE III.」
- 两类编码（原文）：
  - **Tile features**（出现集合/序列的牌类特征，如私手、副露、弃牌）：
    「encoded by setting the corresponding row indices to 1 and leaving the rest to 0」——**每列是一个 (34) 的 0/1 向量**。
  - **Numerical features**（如分数）：「binary-encoded into multiple columns, each being either all zeros or
    all ones」（每列整列同值 0 或 1）。
- TABLE III 关键行：`Self discards = 30`；`Other players' open tiles and discards = (36+4+30)×3 = 210`；
  `Other players' Riichi status = (1+5)×3 = 18`。
- **时间信息**（原文）：「For triplets, quads and Riichi status, we include not only the triplet/quad tiles and
  the Riichi status, but also **the turn numbers of the Pon/Kan/Riichi calls**.」
  → 即 Meowjong **只给副露/立直记录「第几巡」这一标量，没有把弃牌序列拆成「最近 N 巡」逐巡 plane**。
- 「We do not include the number of remaining tiles, because it can be calculated from the number of discarded tiles.」
- 网络（原文）：每个动作一个 CNN，4 层卷积 + 全连接；弃牌模型用 REINFORCE 自我对弈强化；
  明确**不用 pooling**（「the data structure is not an image, but a compact encoding of discrete feature data」）。
- 标注：以上均为**原文依据**。
- **推论（我的）**：Meowjong 把「对手弃牌」压成 210 列（每列是某家某张牌的 0/1/…），
  **弃牌的先后顺序基本被丢掉**（只有副露/立直带了巡目标量）。这与它「数据是紧凑离散编码、非图像」的定位一致，
  也解释了为什么对「占用预测」这类**强依赖顺序**的任务，这种编码可能不够——但这句是**我的推论，非原文结论**。

#### 1.4 kanachan（GitHub，非 arXiv）— 社区最新设计，供对照

来源：https://github.com/Cryolite/kanachan ；数据格式 https://github.com/Cryolite/kanachan/wiki/Notes-on-Training-Data

- 设计取向（原文 README）：「**No Human-crafted Features**」——牌一律当 token，无人工特征；
  「The discarded tiles and the meldings played by each player are represented as **a sequence of tokens
  representing the order in which they occur**.」
- 数据格式（原文 wiki）：第 3 字段是 *progression features*，「**represents a sequence** of non-negative integers.
  Each integer stands for some event in a round… **The order of the integers in the sequence directly represents
  the order in which the events occurred**」；这些整数用作 embedding 索引。
- 且明确（原文）：「**positional encoding must be applied** to the embeddings if they are to be used as a part of
  inputs to models such as ones using **transformer**」。
- 目标（原文）：用大模型 + Transformer，对标 NAGA / Suphx。
- 标注：**原文依据**（一手来源为项目 README/wiki，非同行评审论文；列为社区实践证据，非论文证据）。

#### 1.5 其他相关对手建模论文（非麻将，供迁移参考）

| 论文 | arXiv | 做什么 | 来源 |
|---|---|---|---|
| Opponent Modeling in Multiplayer Imperfect-Information Games | 2212.06027 | 通过**重复交互收集对手行为观测**建模，三玩家 Kuhn 扑克；胜过精确 Nash 策略 | https://arxiv.org/abs/2212.06027 |
| L2E: Learning to Exploit Your Opponent | 2102.09381 | **隐式**对手建模（不显式预测风格/策略），少样本快速适应新对手 | https://arxiv.org/abs/2102.09381 |
| DouZero+: Improving DouDizhu AI by Opponent Modeling and Coach-guided Learning | 2204.02558 | 斗地主，在 DouZero 上加**对手建模** + coach 网络 | https://arxiv.org/abs/2204.02558 |
| Mizukami & Tsuruoka 2015（**麻将对手模型**，最接近本问题） | DOI 10.1109/CIG.2015.7317929 | 标题即 *Building a computer Mahjong player based on Monte Carlo simulation and **opponent models***；也是 Gao 所引 62.1% 基线 | Crossref 确认 DOI 存在；**摘要被出版商 elided，未能读到全文** |
| StratFormer | 2604.25796 | 自适应对手建模与利用（不完美信息，非麻将） | （arxiv HTML 搜索命中，未取全文） |

- 标注：2212.06027 / 2102.09381 / 2204.02558 三条为**摘要原文依据**（已读摘要）；
  Mizukami 2015 仅为**题录存在性依据**（Crossref + Semantic Scholar 均无 abstract）。

---

### 2. 三个重点问题的回答

#### (a) 对手暗手/听牌估计的主流形态？树模型+显式 recency 够不够，还是要序列模型？

**回答（分「有来源」与「未找到」两半）：**
1. **在麻将 AI 论文里，未找到任何一篇训练「专门的对手暗手占用预测器」**（即把「每家每牌种 0..4 张」当监督目标）。
   可检索到的麻将 AI（1906.02146、2202.12847、2003.13590）都是**决策网络内隐式读牌**：
   把可观测信息（含对手弃牌/副露/立直）编码成输入张量，网络自己学。
   - 唯一在标题层面自称做「opponent models」的麻将论文是 **Mizukami & Tsuruoka 2015**
     （DOI 10.1109/CIG.2015.7317929），但其摘要/正文**未取得**，**无法确认其形态**。→ 标为缺口。
2. **主流表征是二维张量/序列，不是树模型**：
   - Gao 2019：`34×4` plane，86 planes，CNN（1906.02146 原文）；
   - Meowjong：`34×366`，每动作一个 CNN（2202.12847 原文）；
   - Suphx：34 通道 + look-ahead 100+×34，ResNet 类（2003.13590 原文）；
   - kanachan：事件 token 序列 + positional encoding + Transformer（README/wiki 原文）；
   - 反例（用树模型做**对手行为预测**）：**Mxplainer**（2506.14246）明确说**决策树方法 top-3 准确率只有 34.8%**，
     而它的参数化搜索/网络达 90%+。这是**唯一一条「树模型在麻将对手行为预测上显著弱于序列/网络模型」的直接证据**，
     但它预测的是**动作（要打哪张）**，不是「暗手占用」。→ 与本问题**方向相关但不能直接外推**。
3. **「树模型 + 显式 recency 特征够不够」——文献未给出答案（未找到）**。
   - 没有任何论文把「recency 特征 vs 序列模型」在**同一任务上做过对照实验**（未找到）。
   - 因此**不能从文献得出「必须上序列模型」的结论**，也**找不到「树模型够用」的支持**。
   - 可间接引用的两点：(i) Gao 证明「把最近 6 巡单独编码」对弃牌一致率有正贡献（原文依据）；
     (ii) 所有 SOTA 麻将 AI 都选了保序表征（原文依据）。但两者都**不构成**「树模型不行」的证明。
   - 标注：3 的「未找到」是**如实缺口**（且受检索通道限流影响，见 §4 方法学警告）；「Mxplainer 34.8%」是**原文依据**。

#### (b) 有没有论文把「对手弃牌序列的时间衰减」做成显式特征喂 GBDT/树模型？

**回答：未找到。**
- **未找到**任何论文把「对手弃牌序列的时间衰减」编成显式特征喂给 GBDT/树模型。
- 最接近的**原文依据**是 **Gao 2019（1906.02146）**：
  - 明确写下「**The closer the discard tiles are, normally more information they possess**」+「discard orders
    cannot be learned（在仅有的弃牌 plane 里）」→ 于是**把最近 6 巡逐巡单独编码为 plane**。
  - 但注意：这是 **硬窗口（最近 6 巡各一个 plane）**，**不是**指数/线性衰减权重；且**喂 CNN，不是树模型**。
- 其余用 recency 的方式（原文依据）：Meowjong 只把副露/立直的**巡目标量**编进输入（2202.12847）；
  kanachan 用**全序列 + positional encoding**（README/wiki），连「衰减」都不显式做。
- 结论：**「时间衰减 + GBDT」这一组合在麻将/牌类文献中无先例**（未找到），
  既没有支持证据，也没有否定证据。→ 这是一个**真实空白**，不是文献已否决的方向。

#### (c) Suphx 原文的 oracle guiding 是干什么用的？有没有别的论文拿它做对手建模？

**回答：**
- **用途 = RL 训练加速**（原文措辞「To speed up the RL training」）。机制：先训一个能看
  **对手暗手 + 牌墙**的全信息 oracle agent，再用 Bernoulli dropout 把完美特征 γ_t 从 1 衰减到 0，
  平滑退化为只用公开信息的 normal agent；理由是「简单知识蒸馏不行」（原文）。
  见 §1.1① 的完整引文。标注：**原文依据**。
- **oracle guiding 不是对手建模**：oracle 用完美信息是为了**给自己当教师**，不是为了**推断对手**。
  标注：**原文依据 + 我的定性**。
- **有没有别的论文拿它做对手建模？——未找到。**
  - 用 `arxiv.org/search` 查 exact phrase `"oracle guiding"`：返回的 25 条**全部无关**（都是 LLM/形式验证/硬件
    领域的 "oracle-guided"），**未见任何游戏/对手建模论文**。
  - `export.arxiv.org/api` 的全文检索在本次会话中**持续 429**，未能做补充确认。
  - 相关但**不等同**的通用概念是「privileged information / asymmetric actor-critic」（训练时用特权信息），
    但**未找到**它在麻将或对手建模上的应用（未找到）。
  - 标注：这是**如实缺口**（受限流影响，置信度中等偏低）。

---

### 3. 对本案的直接含义（推论，非原文）

> 以下**均为我的推论**，用于把文献映射到「GBDT 预测对手暗手占用打不过已见张数基线」这一问题，
> 不代表论文观点。

1. **基线为何可能难打**：单变量「已见张数」≈ (4 − 某牌种已现数) 的**边际分布**；
   它已经吃掉了「财神状态 + 全场可见牌」的绝大部分可用信息。文献里**没有**任何「对手暗手占用预测器」
   可作对照，说明这条题**并非业界常规切入点**——我方遇到「加特征仍不涨」并不奇怪。
2. **若坚持做占用预测**：文献一致表明**保序信息重要**（Gao 的最近 6 巡、kanachan 的 positional encoding）。
   但我方是树模型，**文献没有树模型的对照结论**。可考虑的**最小实验**（不违反硬约束）：
   - 用**离散「最近 k 巡」聚合**（照 Gao 的硬窗口思路）替代指数衰减，看是否优于「已见张数」；
   - 把**对手副露/立直的巡目标量**显式加入（照 Meowjong）；
   - 先把基线口径对齐（§6：换口径要同批数据重跑），再谈「提升」。
   - （以上为推论 + 由原文做法派生，非原文结论。）
3. **oracle guiding 在本项目的定位**：它**不能**用于「推理时读对手牌」（违反红线，见 §5）；
   只能作为**离线训练技巧**（其「normal agent」阶段必须完全丢掉暗手特征）。
   对树模型而言它基本不适用（没有「完美特征 dropout」这一机制）。→ **推论**。

---

### 4. 覆盖缺口与方法学警告

1. **arXiv API 全程 429**（`export.arxiv.org/api`）：本次多次退避（40s/60s/75s/90s/120s/200s）后仍 429。
   故改用 `arxiv.org/search/?query=...` HTML 页（200）。**但该 HTML 搜索：**
   - **不做短语精确匹配**：查 `"oracle guiding"` 返回 25 条无关结果；
   - 查 exact-phrase（带引号）的组合（如 `"opponent modeling" mahjong`、`mahjong transformer`）**返回 total 0**，
     而单独查 `mahjong opponent` 有 2 条 —— 说明**引号会导致零命中**，**「查不到」可能只是引号所致，不能当「文献没有」**。
   - **没有相关性排序**（按提交日期倒序），靠前的结果全是最近的无关论文。
   → 因此本报告的**所有「未找到」结论置信度受限**：它们成立的前提是「通道能查到」，
     而本通道**已被证明会漏**。**不要把本报告的「未找到」理解为「学界没有」。**
2. **未能取得的一手材料**：
   - Mizukami & Tsuruoka 2015（DOI 10.1109/CIG.2015.7317929）——**IEEE 付费墙，摘要被出版商 elided**，
     这是**唯一标题自称「麻将 opponent models」的论文**，却是最大缺口。
   - NAGA、Mortal 的**方法与特征细节**（Mortal README 只给文档站链接；NAGA 为商业产品，无公开论文）。
   - 日文社区（如注目的 riichi AI 博客）——未检索。
3. **未覆盖的通道**：Google/Bing/DuckDuckGo（硬打搜索页会 202/验证码）、
   Semantic Scholar（429）、OpenAlex（503）、IEEE/ACM/Springer 全文（付费墙）。
4. **二手 vs 一手**：kanachan 的结论来自 **GitHub README/wiki**（项目自述，**二手转述的性质**，
   非同行评审）；Mortal 仅取了 README（一手为项目自述）。
   本报告中凡标「原文依据」者，均指**已读到该论文/项目自身的文字**，并尽量给原文引号。
5. **检索日期**：2026-09-29。部分 arXiv 结果号段（2605.xxxxx 等）为未来号段，已实测可访问；
   引用时以本文件记录的 ID 为准。

---

### 5. 「禁入」表（映射本仓硬约束）

| 候选做法 | 违反的硬约束 | 结论 |
|---|---|---|
| 推理时读取/推断对手暗手（作为输入特征） | AGENTS.md §4「代码中不得存在读取或推断对手手牌的路径」 | **禁入**（对手暗手只能作**离线标签**） |
| oracle agent 直接上线（保留完美特征推理） | 同上（推理输入仅限公开信息） | **禁入**；oracle 只能离线训练、且其 normal 阶段须丢特征 |
| 用 torch/tensorflow 做在线推理 | §5「运行路径零第三方依赖」「模型产物纯 Python 可推理」 | 若无法导出纯 Python 确定性前向 ⇒ **禁入**；能导出则可行 |
| 引入 Suphx/Gao/Meowjong/kanachan 代码 | 许可证：Mortal 为 **AGPL-3.0**（传染）⇒ 不可抄入本仓；kanachan 需逐个核实许可 | 只读思想，**不抄代码** |
| 把「避免放铳风险」当主收益 | 本平台**只能自摸、无点炮** | **不适用**（该收益路径在本变体断裂） |

> 无「与硬约束相容且可行」的**直接可搬方案**——本报告定位是**给方向头绪 + 立空白**，非给现成方案。

---

### 6. 每条来源清单

| # | 来源 | URL / ID | 我读到的内容 | 标注 |
|---|---|---|---|---|
| 1 | Suphx 摘要 | https://arxiv.org/abs/2003.13590 (arXiv:2003.13590v2) | 全文摘要 | 原文依据 |
| 2 | Suphx 全文 | https://arxiv.org/pdf/2003.13590v2 | §3.1 look-ahead、§3.3 oracle guiding、§3.4 pMCPA | 原文依据 |
| 3 | Gao et al. 2019 摘要 | https://arxiv.org/abs/1906.02146 (arXiv:1906.02146v2) | 全文摘要（70.44% vs 62.1%） | 原文依据 |
| 4 | Gao et al. 2019 全文 | https://arxiv.org/pdf/1906.02146v2 | §IV 86 planes、最近 6 巡、ablation +1% | 原文依据 |
| 5 | Meowjong 摘要 | https://arxiv.org/abs/2202.12847 (arXiv:2202.12847v3) | 全文摘要（2D 结构、5 CNN、REINFORCE） | 原文依据 |
| 6 | Meowjong 全文 | https://arxiv.org/pdf/2202.12847v3 | §IV 34×366、TABLE III、编码规则 | 原文依据 |
| 7 | Meowjong 仓 | https://github.com/VictorZXY/Meowjong | README（CoG 2022 发表信息） | 原文依据（项目自述） |
| 8 | kanachan | https://github.com/Cryolite/kanachan | README（no human-crafted features、sequence） | 原文依据（项目自述） |
| 9 | kanachan 数据格式 | https://github.com/Cryolite/kanachan/wiki/Notes-on-Training-Data | progression features、positional encoding | 原文依据（项目自述） |
| 10 | 多人不完美信息对手建模 | https://arxiv.org/abs/2212.06027 | 摘要 | 原文依据 |
| 11 | L2E | https://arxiv.org/abs/2102.09381 | 摘要 | 原文依据 |
| 12 | DouZero+ | https://arxiv.org/abs/2204.02558 | 摘要 | 原文依据 |
| 13 | Mizukami & Tsuruoka 2015 | DOI 10.1109/CIG.2015.7317929 | 仅题录存在（Crossref+Semantic Scholar 无摘要） | 题录依据（内容未得） |
| 14 | Mxplainer（树模型对照） | https://arxiv.org/abs/2506.14246 | 摘要：决策树 top-3 34.8% vs 其方法 90%+ | 原文依据 |
| 15 | StratFormer | https://arxiv.org/abs/2604.25796 | 仅标题（HTML 搜索命中） | 题录依据 |
| 16 | Mortal | https://github.com/Equim-chan/Mortal | README（AGPL-3.0、文档站） | 原文依据（项目自述） |

**明确未找到（如实记录）**：
- 未找到任何「预测对手暗手占用」的麻将论文。
- 未找到「对手弃牌时间衰减 → 显式特征 → GBDT/树模型」的先例。
- 未找到除 Suphx 外使用「oracle guiding」的游戏/对手建模论文。
- 未取得 Mizukami & Tsuruoka 2015 的摘要/正文。
