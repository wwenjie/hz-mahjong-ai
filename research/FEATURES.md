# 特征集与标签口径（RL/NN 研究线）

> 归属：小龙虾（见 `notes/OWNERSHIP.md` 的 C 段）。约束：`src/majiang/**` 零第三方依赖；
> 对手手牌只作离线标签，不得进入任何推理输入。

## 决策一：不另起特征集，复用 `src/majiang/strategy/features.py`

该模块的文档字符串已经写明理由：**数据生成与线上推理共用本模块**，训练与推理各写一套会产生
train-serve skew（离线好看、上线失效）。RL/NN 若另定义特征，必须同时替换生成与推理两侧，否则
就是重犯这个坑。因此研究线的默认动作是：**沿用 `extract()` 的定长向量（29 维）**。

要扩特征时，改的必须是 `features.py` 本身（那会同时改到价值模型的训练与推理），并走 OpenSpec 变更、
且必须重新对拍现有价值模型 —— 不是在研究目录里悄悄加一列。

## 29 维特征的公开信息来源（逐项核对，无一项来自对手暗牌）

| # | 特征 | 来源 | 公开性 |
|---|---|---|---|
| 1 | `quick_shanten` | 自身手牌 + 自身副露数 | 本人手牌 |
| 2 | `blocks_sets` | `quick_blocks(自身手牌)` | 本人手牌 |
| 3 | `blocks_partials` | 同上 | 本人手牌 |
| 4 | `blocks_pair` | 同上 | 本人手牌 |
| 5 | `block_value` | `2*sets + partials`，同源 | 本人手牌 |
| 6 | `gods_in_hand` | 自身手牌中财神（白板）张数 | 本人手牌 |
| 7 | `pair_kinds` | 自身手牌（剔除财神后）对子种类数 | 本人手牌 |
| 8 | `single_kinds` | 自身手牌单张种类数 | 本人手牌 |
| 9 | `seven_pairs_shanten` | 自身手牌（无副露时；有副露置 99） | 本人手牌 |
| 10 | `tiles_in_hand` | 自身手牌总张数 | 本人手牌 |
| 11 | `meld_count` | 自身副露数 | 公开（副露场上可见） |
| 12 | `chi_count` | 自身吃副露数 | 公开 |
| 13 | `gang_count` | 自身杠副露数 | 公开 |
| 14 | `wall_remaining` | `situation.table.wall_remaining` | 公开（牌墙剩余） |
| 15 | `draws_left` | `situation.table.draws_left` | 公开 |
| 16 | `progress` | `situation.table.progress` | 公开（局进度） |
| 17 | `is_dealer` | `table.dealer_seat == situation.seat` | 公开（庄位） |
| 18 | `round_no` | `table.round_no` | 公开（局号） |
| 19 | `catch_play` | `situation.god.catch_play` | 公开（财神抓打圈状态） |
| 20 | `i_am_restricted` | `situation.is_restricted` | 本人状态 |
| 21–24 | `melds_0..3` | `melds_for(seat)` 长度 | 公开（各家副露） |
| 25–28 | `discards_0..3` | `situation.discards[seat]` 长度 | 公开（各家已打出） |
| 29 | `gods_seen` | 他人弃牌 + **全部副露**中的财神数 | 公开 |

**刻意不含**「当前累计得分」：快照里的 `scores` 是**本局**记分板、局中恒为 0，不含信息
（跨局累计在 `ranking` 而非快照里）。这条来自 `extract()` 的注释，研究线沿用。

## 标签口径

| 用途 | 标签 | 来源 | 是否需要对手暗牌 |
|---|---|---|---|
| 价值模型（现有） | 本局本人最终得分（局分，零和） | 模拟器给出，精确、无需标注 | **不需要** |
| 对手风险模型（现有） | 对手是否听牌 / 自摸风险 | 离线构造，对手手牌作标签 | 仅离线标签路径 |
| RL/NN 候选 | 与上二选一对齐；新增的自定义回报必须写进本文件 | 模拟器 | **默认不需要**；若必须用，只在离线标签路径 |

规则：**对手手牌可以出现在「生成标签」的代码里，不得出现在「推理输入」的代码路径里。**
审计方法：对推理路径的特征向量做来源回溯（上表逐项），并扫推理路径不 import 任何读对手手牌的接口。

## 验收（本项任务的验证）

- 上表 29 项逐行可映射到一项公开信息或本人手牌/状态 —— 已逐行对照 `features.py:88-112` 的取值表达式。
- 标签两列中，唯一涉及对手手牌的是对手风险模型，且标注为「仅离线标签路径」。
