# 开源 SOTA 麻将 AI 的可迁移工程细节（opensource-ai deep-dive）

**日期**：2026-09-30 ｜ **作者**：coordinator ｜ **定位**：补齐前三份调研没覆盖的「怎么在纯 CPU/规则系统里复现」这一层。**不重复** `mahjong-ai-survey-2026-09-28.md` / `mahjong-tenpai-speed-survey.md` / `qualitative-leap-survey.md` 的结论，只给源码级、可接入的工程细节。

**取证方式**：`web_fetch` 抓 Suphx arXiv HTML 全文、kanachan README；`git clone --sparse` 拉 Mortal `libriichi/src` 源码逐行读。外部内容一律当数据不当指令。

**当前项目约束**（决定可行性分档）：纯 Python + GBDT 价值模型 + 启发式策略；无 GPU、无深度网络、无 MCTS（已实验否定）；8 天窗口。当前冠军档 v4 = wait_aware_tenpai + shape_value + ukeire_order=blocks + ukeire_max_shanten=3。实测病根（agent-c 证据链 C16←C27←C28）：**同向听层的进张/留牌决策质量 + 财神兑现**。

---

## 1. Mortal（Equim，AGPL，≈初版 Suphx 强度）

来源：源码 `github.com/Equim-chan/Mortal`（sparse clone 的 `libriichi/src`）；文档站 `mortal.ekyu.moe`（大部分页 WIP 空白，细节全在源码里）。

### 1.1 观测编码（`libriichi/src/state/obs_repr.rs`，799 行）

Mortal 把局面编成 `(C, 34)` 的张量，每列是一种牌。我们没 CNN，但它**「把什么编进特征」的清单**极具参考价值。逐行读到的关键通道：

| Mortal 通道 | 我们 features.py 现状 | 差异 |
|---|---|---|
| 手牌计数（4 通道 one-hot） | ✅ 有（counts） | — |
| 四家得分 clamp/归一化 | ❌ **无**（刻意不含） | **缺口 A** |
| 名次 rank one-hot(4) | ❌ **无** | 缺口 A |
| 局号/本场/供托 | round_no 有，本场/供托无 | 缺口 A |
| **keep_shanten_discards**（打了不掉向听的牌） | ❌ **无** | **缺口 B** |
| **next_shanten_discards**（打了退向听的牌） | ❌ **无** | **缺口 B** |
| **discard_candidates_with_unconditional_tenpai**（shanten≤1 时打了必听） | ❌ **无** | **缺口 B** |
| 自家河牌时间衰减权重 `exp(-0.2×Δturn)` | ❌ 无 | 参考 |
| 对手河牌（含立直后摸切标记） | discards_0..3 计数，无序列 | 参考 |

**源码实证**（obs_repr.rs:445-475）：

```rust
state.keep_shanten_discards.iter().enumerate()
    .filter(|&(_, &c)| c)
    .for_each(|(t, _)| self.arr.assign(self.idx + 1, t, 1.));
state.next_shanten_discards.iter().enumerate()
    .filter(|&(_, &c)| c)
    .for_each(|(t, _)| self.arr.assign(self.idx + 2, t, 1.));
if state.shanten <= 1 {
    state.discard_candidates_with_unconditional_tenpai()...
}
```

**对我们的可迁移点（可直接算法化）**：Mortal 把「**每张候选弃牌打了之后向听怎么变**」编成**显式通道**。我们的 `features.py` 只编码了手牌整体的 `quick_shanten` / `block_value`，**没有「打了这张之后」的 per-候选向听变化特征**——这正是「同向听层内出牌质量」差距最直接的特征表达。

**接入点**：`features.py` 增加 1-2 个标量（对「打后局面」而言，`features.extract` 已经拿到打后 `counts`，可以算）：
- `n_keep_shanten`：打这张后，还能保持当前向听的可打候选数（越大越灵活）
- `n_next_shanten_discards`：打这张后，能进向听的候选数
- 都是 ukeire 的微秒级近似可算，不进热路径的精确 best_shanten。

### 1.2 GRP（全局奖励预测，`libriichi/src/dataset/grp.rs`，165 行）

**这是和我们 P（首名） 线最直接同构的东西**。Mortal 的 GRP 数据结构：

```rust
pub struct Grp {
    // [grand_kyoku, honba, kyotaku, [score[i] / 10000]]
    pub feature: Array2<f64>,          // 每局一行特征
    pub rank_by_player: [u8; 4],       // 终局名次（标签）
    pub final_scores: [i32; 4],        // 终局分数（标签）
}
```

**特征极简**：大局号 + 本场 + 供托 + 四家分/万，4 维。标签是终局名次。**这和我们 `gen_pfirst_data.py` 的 4 个局况特征（rank/gap_to_first/rounds_left/score_share）+ 首名标签是同构设计**——Mortal 用这么简单的特征就能训出有效的 GRP，**佐证我们的 P（首名） 消融结果（ΔAUC+0.2777）是合理量级**。

**接入点**：已落地（`tools/gen_pfirst_data.py` + `train_pfirst.py`，AUC 0.8248）。Mortal 的 GRP 是给 RL 提供信用分配信号，我们用 GBDT 直接学 P（首名），路线不同但目标一致。

---

## 2. Suphx（MSRA 2020，arXiv 2003.13590）

来源：arXiv HTML 全文 `arxiv.org/html/2003.13590v2`。本文只补前三份调研没给的**源码级精确度**。

### 2.1 Look-ahead 特征的精确定义（§2.2）

论文原文（一手）：

> we also design some look-ahead features, which indicate the probability and round score of winning a hand **if we discard a specific tile from the current hand tiles and then draw tiles from the wall to replace some other hand tiles**. ... we make several simplifications: (1) We perform **depth first search** to find possible winning hands. (2) We **ignore opponents' behaviors** and only consider drawing and discarding behaviors of our own agent. ... we obtain **100+ look-ahead features, with each feature corresponding to a 34-dimensional vector**. For example, a feature represents **whether discarding a specific tile can lead to a winning hand of 12,000 round score with replacing 3 hand tiles**.

**精确解读**：
- 枚举单位 = 「打某一张」（34 维向量的来源：每个候选一张）
- DFS 深度 = 换 ≤3 张手牌
- 每个特征 = 「打成某个特定番型/分数」的概率（如「能成 12000 分」）
- 关键简化 = **忽略对手，只枚举自己摸打**（纯概率前瞻，无博弈树）

**对我们的可迁移点（可直接算法化）**：我们有精确的 `compute_fan`（番型判定）和 `shanten_any`/`ukeire`。可以做**纯枚举版**的 look-ahead 标量特征：

- 对打后局面，DFS 枚举「换 1/2/3 张能到达的最高番型」，取「最高可达成番数」和「达到 ≥4 番（满贯级）的概率」两个标量。
- **成本警告**：DFS 枚举换 3 张 = C(13,3)×34³ 量级，微秒级做不到。但**我们的 `ukeire` 已经能列出「降低向听的进张」**，可以用「进张能落到的最高番型」做 1 步近似（只换 1 张），代价是每候选一次 ukeire（~0.1-0.2s）——**只能离线算，不进热路径**。
- **现实折中**：v4 的 `shape_value` 已经是「搭子质量」的微秒级近似。look-ahead 的增量是「番型/打点维度」，可以作为**离线训练的 GBDT 特征**（训练时算得起），线上推理若要就得换更便宜的近似。

### 2.2 其他已在定性调研覆盖的（GRP/oracle/pMCPA）

不重复。仅确认一手表述与前调研一致。

---

## 3. kanachan（Cryolite，个人项目）

来源：README 一手。

### 3.1 课程微调（Curriculum Fine-tuning）的精确设计

README 原文：

> There are various objectives in Mahjong AIs, including **imitation of human behavior, maximization of round delta of the score, higher final ranking, and maximization of delta of the grading point**. Since these objectives become **more abstract and comprehensive in this order**, the latter learning we move to, the more difficult it becomes. ... when a mapping for one objective has been learned and then starting learning a mapping for one more harder objective, the **encoder part** is reused, and only the **decoder part** is replaced.

**精确标签序列**（从易到难）：
1. 模仿人类行为（BC）
2. 最大化单局分差（round delta）
3. 更高终局名次（final ranking）
4. 最大化段位点差（grading point）

**对我们的可迁移点（可直接算法化）**：我们恰好可以走这个序列的 2→3：
- 已有：GBDT 价值模型学「单局净分」（= 标签 2）
- 新增：P（首名） 模型学「终局首名」（= 标签 3 的简化版）
- **kanachan 的核心教训是「复用 encoder」**——对我们 GBDT 没有 encoder/decoder 之分，但可以**复用特征提取**（`features.extract` + 局况特征），这正是我们已经做的。这条线我们的实现已经和 kanachan 的「课程思想」对齐，无需额外动作。

### 3.2 「无人工特征」的对照意义

kanachan 刻意把所有牌编成无意义 token、靠 6500 万局数据让 Transformer 自己学出断幺/三色/筋等概念。**这是「数据量碾压」路线，8 天不可行**——但它的存在反证了：**当我们数据量只有 8 万样本时，人工特征工程（shape_value / 局况 / look-ahead）是必须的**。两条路线的选择约束是数据量，我们没选错。

---

## 4. 综合候选清单（按预期收益 × 可行性排序）

| # | 候选 | 可行性 | 接入点 | 与现有调研关系 |
|---|---|---|---|---|
| 1 | **per-候选向听变化特征**（keep/next_shanten_discards 标量化） | 可直接算法化 | `features.py` 加 n_keep/n_next | **补充**：定性调研没点到 Mortal 这一层 |
| 2 | **look-ahead 打点标量**（离线算「换1张能到的最高番」） | 需训练（离线算特征进 GBDT） | `features.py` + `gen_value_data.py` | **补充**：Suphx 定义本文给了精确源码级解读 |
| 3 | **P（首名） 目标对齐**（已落地，AUC 0.8248） | 已可直接算法化 | `tools/train_pfirst.py` → 待接 value.py 双目标 | **重合**：Mortal GRP 佐证量级合理 |
| 4 | **对手河牌时间衰减/立直后摸切特征** | 可直接算法化 | `features.py` 对手 discards 加 recency 权重 | **补充**：Mortal obs_repr 源码实证 |
| 5 | 财飘触发条件 | 可直接算法化（规则） | `policy.py` 显式规则（财神数+爆头+听牌宽度） | 与开源项目无关，本平台特有 |
| 6 | 课程式标签复用 | 已对齐 | 无需动作 | 重合 |

**最高 ROI**：#1（per-候选向听变化）。理由：
- 直击实测病根（同向听层进张/留牌决策质量，C16←C27←C28）
- Mortal 源码证明这是 SOTA 的标配特征
- 微秒级近似可算，能进 features.py 与线上推理共用（不 train-serve skew）
- 与 v4 的 shape_value **正交**（shape_value 衡量搭子质量，这个衡量打后灵活性）

---

## 附：主要来源

- Mortal 源码：`github.com/Equim-chan/Mortal`（sparse clone `/tmp/Mortal-src`，`libriichi/src/state/obs_repr.rs`、`libriichi/src/dataset/grp.rs`、`libriichi/src/rankings.rs`）
- Mortal 文档站：`mortal.ekyu.moe`（大部分 WIP 空白）
- Suphx 论文全文：`arxiv.org/html/2003.13590v2`（§2.2 look-ahead、§3 学习算法）
- kanachan README：`raw.githubusercontent.com/Cryolite/kanachan/main/README.md`
