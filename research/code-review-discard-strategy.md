# 出牌/风险策略代码评审（丢牌与进张效率）

评审对象（只读）：
- `src/majiang/strategy/policy.py`
- `src/majiang/strategy/risk.py`
- `src/majiang/rules/shanten.py`

方法：逐行读码 + 在真机事件流上**离线复现**决策（`data/auto_sessions/*/events/*.json`，
`majiang.sim.replay` 重建四家手牌，`HeuristicReadyModel` + 当前默认档公式复算
`total = -10×向听 + 骨架 - 3×喂牌 - 财神罚`）。除特别注明外，样本 = 前 40 个房文件
（含我方 `u_a7f7c67bb14a` 的局），共 **2529 个同向听并列决策点 / 4428 个出牌点**。
所有复现脚本均为一次性只读计算，未改动仓库任何文件。

结论先行：**没有发现会让策略崩溃或算错的实现 bug（含 v3 修复路径，已验证正确）**；
发现的是**一整族「看起来在算、实际不判别」的退化项**，以及**中段形质/进张权重被喂牌项
结构性压倒**的设计性缺陷。这也精确解释了「比强 bot 慢 ~1 摸到听」。

---

## 一、已验证事实（读码 + 复现证明）

### F1. `shape_value` 不是退化的——但 `quick_blocks` 的 `2×面子+搭子` 确实退化，两者常被混淆
**文件:行号**：`src/majiang/rules/shanten.py:363`（`partials = min(partials, slots - sets)`）、
`shanten.py:367-443`（`shape_value`）；调用点 `policy.py:851 / 877 / 896`。

**问题**：`shape_value` 的文档（`shanten.py:369-380`）把 **77.2% 全并列** 记在
`shape_value` 名下，实测该数字属于 **旧键** `2×面子+搭子`，不是 `shape_value`。

**复现**（2529 个并列点）：

| 副露数 | 决策点 | 旧键 `2×面子+搭子` 全并列 | `shape_value` 全并列 |
|---|---|---|---|
| 0 | 1695 | **79.4%** | 36.2% |
| 1 | 706 | **88.7%** | 57.9% |
| 2 | 123 | **91.9%** | 71.5% |
| 3 | 5 | 80.0% | 40.0% |
| 合计 | 2529 | **82.6%** | 44.0% |

旧键不同取值数分布 `{1:2088, 2:436, 3:5}`（几乎只有一个值）；
`shape_value` 为 `{1:1112, 2:998, 3:290, 4:103, 5:23, 6:3}`。
两键 `argmax` 不一致率 **15.2%**，两者**同时**全并列仍有 **43.3%**。

**影响**：① 旧键名下的结论（"形质退化"）方向正确，但**归因标错了函数**，会把修法引到
`shape_value` 上去（它已经比旧键好一倍）。② `shape_value` 仍有 **44% 全并列**，
其中副露 1/2 组时高达 58%/72% —— **`shape` 实验臂只解决了不到一半的退化**。
③ `ok` 的另一面：形质项分辨率上限 `0.9×0.4=0.36`，而喂牌项差值最大 `3×0.6×Δthreat`，
在 `threat≈1` 时约 1.8 ⇒ **形质项在数值上根本压不过喂牌项**（见 F6）。

**建议**：修正 `shanten.py`/`policy.py` 里"77.2% 属于 shape_value"的注释；对剩余 44%
全并列引入**结构性**次级键（听口张数的一拍近似、孤张价值、幺九/中张分层），而不是继续
微调 `mean_w`；并把 `shape`/`shape-only` 臂的判读从"是否已修复退化"改为"退化只修了一半"。
**置信度**：high（计数为直接复现）。

### F2. `EXACT_UKEIRE_MAX_SHANTEN = 1` 是一个**结构门**：向听 ≥2 时进张完全不出现在决策里
**文件:行号**：`policy.py:65`（常量）、`policy.py:143`（`ukeire_max_shanten: int = 1`）、
`policy.py:769`（`if exact and top_shanten > self.config.ukeire_max_shanten: return None`）。

**问题**：精确进张次排序只在 `top_shanten <= 1` 生效；其余一律 `return None`，出牌完全由
`-10×向听 + 骨架 - 3×喂牌 - 财神罚` 决定。

**复现**：真机出牌点向听分布 = `{0: 1163, 1: 1366, 2: 1115, 3: 622, 4: 147, 5: 14, 6: 1}`
⇒ **向听 ≥2 占 43%（1914/4428）**，这些出牌点**一次进张计算都没做过**。
而这些正是"第 4 摸均向听就落后 0.16"发生的那一段。

**影响**：`ukeire-wide`（只改候选数 2→6）测平是**必然的**——门没打开，候选面再宽也不看进张。
仓库自己的注释（`policy.py:135-141`）已经写明这一点，但被 `ukeire-early/g2/g5/deep` 四条臂
反复"重新发现"，说明该门的**归因一直没被固化**。

**建议**：把"向听 ≥2 不看进张"这件事写进 `policy.py:65` 的注释，并明确
`ukeire-early/g2/g5` 的判据是**一次性**的（任一为正 ⇒ 门结构有误；全为负 ⇒ 门合理但需换统一期望得分）。
**置信度**：high（行号+分布直接可见/可复现）。

### F3. 候选面截断会切掉**精确进张最优张**（实测 23.4%）
**文件:行号**：`policy.py:779-782`
（`if not wait_aware and not two_ply: tied = tied[: max(1, self.config.ukeire_candidates)]`），
默认 `ukeire_candidates = 2`（`policy.py:66, 138`）。

**问题**：并列候选按 `total` 排序后**先截断再算进张**，而 `total` 在同向听时被
`-3×feed` 主导（F6 实测 97.3%），于是"骨架更好但喂牌稍多"的候选在进张比较前就被丢弃。

**复现**：筛出"向听 ≤1 且并列候选 ≥3"（即截断确实生效）的决策点 47 个，
对每张并列候选算**精确进张**（`shanten.ukeire` 求和），检验进张最优张是否落在
`total` 前 2 名之内：**命中 36（76.6%），被截掉 11（23.4%）**。
截断生效面本身很大：并列 >2 的决策点 2068/2529 = **81.8%**。

**影响**：一个可复现的、方向单向的损失（每 4~5 个可截断决策就有 1 个丢掉真正最宽进张的那张），
且与"我们听口窄于对手约 21%"（`tools/analyze_wait_quality.py`）方向一致。
（n=47，只在前 40 房；`ukeire-wide` 臂当年测平的机制解释之一。）

**建议**：把截断改为**先算进张再截断**（或对并列全算，仅对耗时做 0.6 s 墙钟保护，已有）。
`ukeire_order="blocks"` 只改排序键、不改截断时机，因此**不是**这条的修法
（实测 `top2(total序) ≠ top2(blocks序)` 只有 1.8%，两个键的前 2 名几乎一样）。
**置信度**：medium-high（机制与方向确定；单点收益大小受 n 限制）。

### F4. 中段（向听 ≥2）**几乎从不打中张**——中张废弃是本代码最硬的实测指纹
**文件:行号**：`policy.py:854-858`（`block_value` 退化）、`policy.py:857-858`
（`threat = Σ ready_probability; feed = visible_need(tile) × threat`）、
`risk.py:146-154`（`visible_need`：中张 1.0 / 边张 0.6 / 字牌 0.4）。

**问题**：向听 ≥2 时 `block_value` 全并列（F1），`total` 只剩喂牌项，
而喂牌项对中张的价格是字牌的 **2.5 倍**（1.0 vs 0.4，`3×` 后差 1.8 分），
于是中张被系统性避开。

**复现**（4428 个出牌点）：候选牌共 46412 张，其中中张 25727 张 = **55.4%**；
**实际打出中张 703/4428 = 15.9%**。分向听：

| 出牌时向听 | n | 打中张比例 |
|---|---|---|
| 0（已听） | 1163 | 40.8% |
| 1 | 1366 | 15.7% |
| 2 | 1115 | **1.3%** |
| ≥3 | 784 | **0.0%** |

打出最多的牌 = 东/南/西/北/中/**发**（字牌）与 1w/9w（幺九）——**中张几乎从不入池**。
（`focus` 探针在 277 个向听 ≥2 决策点上也得到 0.0% 中张。）

**影响**：这就是"我们比强 bot 少打 ~47% 中张"的直接来源。中张是牌效里**唯一能成两面**的
资源；把它们当天敌丢光，等于每局主动放弃有效进张——**恰恰是"慢 1 摸到听"的机制**。
而且这是**权重结构**问题，不是常数没调好：F6 显示即使开 `shape_value`，
在向听 ≥2 这一段 `total` 仍然 97% 由喂牌决定。

**建议**：这条必须**和 F6 一起**修，单独调 `feed_weight` 无效（注释已记录 `feed-low` 曾测平）。
最小可验证改动：在 `policy.py:854-858` 这一段，让 `block_value` 的判据从
`2×面子+搭子` 换成能区分中张潜力的量（例如把 `shape_value` 接上，并让 `need` 裁剪**保留雀头槽**），
同时把喂牌项改成**只在对手确实接近听牌时**才计价（见 F7 的 threat 归一化）。
**置信度**：high（比例与分向听曲线均为直接复现）。

### F5. v3 修复路径**正确**，且没有发现残留的静默退化
**文件:行号**：`shanten.py:295-296`（`current = shanten(...)`; `if current == 0: return ()`）、
`policy.py:774`（`wait_aware = exact and top_shanten == 0 and self.config.wait_aware_tenpai`）、
`policy.py:378-402`（`_wait_copies`，`except ValueError: return None`）。

**问题（待验的怀疑）**：听牌时 `shanten.ukeire` 返回空元组 ⇒ 调用方把"全 0"当平局。
**验证结果**：
1. `ukeire` 在 `current == 0` 仍返回 `()`（`shanten.py:296`），**未改**——
   依赖面因此**必须**由调用方修。
2. 调用方的修法**是充分的**：`wait_aware` 分支完全绕开 `ukeire`，改用 `win.winning_draws`。
3. `wait_aware_tenpai=False`（即 v2/`heuristic` 默认）时：`ukeire` 分支里每个候选
   `copies` 恒为 0 ⇒ `best` 停在 `scores[0]`（`total` 第一名），**即"精确进张次排序静默失效"**——
   这不是 bug（`return None` 语义正确），而是**默认档仍带着 v2 的退化**。
4. `shanten_any == 0` 是否真等价于"能胡/听任意"（`_wait_copies` 依赖它）：
   在 **1579** 个 `shanten_any==0` 的候选上，`winning_draws` **全部非空（1579/1579）**，
   不一致 **0** 例。**不含**听任意（爆头）——爆头由 `is_baotou` 另行处理。

**影响**：怀疑 3 的修复路径**确认无残留退化**；真正的问题是**默认档不是 v3**
（`cli.py:97` `heuristic` = 零 override 的 `PolicyConfig`，`wait_aware_tenpai=False`），
且 `check`搜索臂/RL 线整族坐在 v2 底上——若线上跑的是 `heuristic`/`v2`，听牌时就是在按
"骨架+喂牌"选牌，**完全不看听口**。

**建议**：把这条从"需要修 bug"改判为"**需要确认线上实际 `--decider`**"；
若线上是默认 `heuristic`，切 `v3` 是零风险的既有收益（仓库已有 2000 配对场 + 两种子证据）。
**置信度**：high（含 1579 例一致性复现）。

### F6. 同向听决策 **97.3% 由单一项（喂牌）决定**，形质项无判别力
**文件:行号**：`policy.py:854`（`block_value = 2*blocks[0] + blocks[1]`）、
`policy.py:857-858`、`policy.py:875-878`。

**复现**（2529 个并列点，用真机 threat 复算 `total`）：
`argmax(total) == argmax(-feed)` 一致 **2461/2529 = 97.3%**。
（与仓库已确立的 97.7% 一致，本次用默认档公式独立复现。）

**影响**：确认"中段策略 ≈ 只做一件事：少喂牌"。配合 F4（中张是喂牌最贵的牌），
策略在中段的行为被完全解释：**丢幺九/字牌 + 丢中张 = 只剩安全牌可打**，
手牌骨架在向听 ≥2 期间几乎不推进。这是"慢 1 摸到听"的根因，而不是某个常数偏了。

**建议**：把 F1/F4 的形质修复与 F7 的 threat 归一化合并为一次改动，
判据用**机制量**（自对弈到听摸序、中张出牌占比回升）而非仅 A/B 名次分。
**置信度**：high。

### F7. `threat` 系统性高估的**全部**来源已定位（常数、缺归一化、两处重复计数）
**文件:行号**：`risk.py:34-36`（`BASE_READY=0.06 / DISCARD_PROGRESS_WEIGHT=0.30 /
MELD_READY_WEIGHT=0.16`）、`risk.py:83`（`+0.18 * game_progress`）、
`risk.py:31`（`CONSERVATIVE_UPLIFT = 1.25`）、`risk.py:79-82`、
`risk.py:112-115`（`risk = ready × conditional × uplift`）、
`risk.py:41,91-94`（`EXPECTED_WAITS=4.0`、`CONDITIONAL_DRAW_MAX=0.40`）。

**问题**（怀疑 5 的具体落点）：
1. **无归一化的加法先验**：`ready = 0.06 + 0.30·(1-e^{-弃牌/6}) + 0.16·副露数 + 0.18·进度`
   （`risk.py:79-84`）——四个正项直接相加，**没有对"三项同时发生"做折扣**。
   "弃牌多"与"进度深"在本局里几乎完全共线（都是同一把时钟），于是同一事实被计两次。
2. **0.18 与 0.30 共线再计一次**：`0.18*progress` 硬编码在 `estimate` 里，
   **不在** `PolicyConfig`/常数表里（`risk.py:83`），调参时极易漏。
3. **`CONSERVATIVE_UPLIFT = 1.25` 从未标定**：`risk.py:31` 的注释说它是为补偿
   "听任意不可观测"，但校准工具（`tools/calibrate_threat.py`）**只复刻了 `ready` 公式，
   完全没带 `uplift` 与 `conditional`**，所以 1.25 从头到尾没被任何数据检验过。
4. **训练模型也偏正**：`cli.py` 的 `risk-v3` 注释记录，同批位置标定 GBDT 仍高估 **1.6 倍**
   （21.3% vs 实测 13.2%）。连"更准"的模型也是正的 ⇒ 偏差更像是**进入 `total` 的口径**问题，
   而不只是模型问题。
5. **`hand_counts` 未被使用**：`HeuristicReadyModel.estimate` 只吃
   `副露数 / 弃牌数 / 进度`，`Situation.hand_counts_for()`（公开的暗手张数）**完全没用**——
   副露 3 组的对手暗手只有 4 张，模型却只给 `+0.16×3`，**低估**"接近听牌"的尾部，
   与整体高估叠加后，在**最该防守的副露手**上反而偏差方向相反。

**校准证据**（`tools/calibrate_threat.py` 注释内已落盘的数据）：手写模型 42.0% vs 实测 28.2%
⇒ **1.49×**；与本次复现的 `threat = Σready` 分布中位 **1.002**（三家合计）一致
⇒ 真实值约 **0.67**。

**影响**：`feed = visible_need × threat`，threat 放大约 1.5 倍 ⇒ 喂牌惩罚被放大 1.5 倍
（`feed_weight=3.0` 的实际效果 ≈ **2.0 的威胁 × 3.0 的权重 = 4.5**）；
同一 threat 还进 `lap_survival`（`risk.py:120-127`）⇒ 低估生存率 ⇒ 过于保守不敢飘。
仓库自己的建议是"`feed_weight ≈ 3.0/1.49 ≈ 2.0`"。

**建议**：三选一并量化，不要只调 `feed_weight`：
① 把 `ready` 改成**乘性/逻辑斯蒂**形式（`progress`、`discards` 合成一个"时钟"项，避免三次计同一事实）；
② 显式标定 `CONSERVATIVE_UPLIFT`（让 `calibrate_threat.py` 复刻 `uplift × conditional` 后再比，
并把工具里 `draws/16` 的 `progress` 代理换成与 `TableState.progress` 同一口径——
见 F8）；
③ 接入 `hand_counts_for(seat)`（副露手 = 张数少 = 更近听）作为特征。
**置信度**：high（数值、共线项、未标定项均为直接读码；倍数为仓库已落盘数据 + 本次分布复现）。

### F8. 校准工具与 `risk.py` 的 `progress` 口径**不一致**（会让标定结果漂移）
**文件:行号**：`risk.py:83` 用 `situation.table.progress` =
`1 - draws_left/MAX_DRAWS`（`src/majiang/rules/table.py:60-62`）；
`tools/calibrate_threat.py:46`（`DRAW_SPAN = 16.0`）与 `predict()` 用
`min(1.0, draws / 16.0)`。

**问题**：同一"进度"在两处定义不同（`draws_left/MAX_DRAWS` vs `draws/16`），
两者只在开局附近近似相等（`MAX_DRAWS` 与 16 不等）。

**影响**：`calibrate_threat.py` 报出的"高估 1.49×"本身带一个**口径误差**，
用它去反推 `feed_weight` 会带偏差。这不改变"高估"的方向（该工具已在文档里自陈
"按房抽样"的坑），但会改变量值。

**建议**：让工具直接调用 `HeuristicReadyModel` / `TableState.progress`，
不要手抄常数（工具注释里"逐字对齐"的做法恰恰是漂移的入口）。
**置信度**：medium-high（两处定义均可读；量级影响未单独量化）。

### F9. 设计缺口：`visible_need` 只看牌种，**完全不看我方手牌需要什么**
**文件:行号**：`risk.py:146-154`；调用点 `policy.py:858`。
**问题**：喂牌项 = `F(牌种) × Σ ready`。`F` 是写死的三档（中张 1.0 / 边张 0.6 / 字牌 0.4），
不含"这张我是否在用""这张对手若已有三张则喂牌价值骤降"。
**影响**：① 与 F4 叠加，把"中张=天敌"这一错误结论固化进策略；② 我方正需要的牌
天然也是对手需要的牌——正确口径应是**边际的**（我舍它，对手手牌推进多少），
而现在对"我舍的是不是我的孤张"无感。
**建议**：`visible_need` 加两项：我方该牌种的块数（我已在用的牌，弃之代价高）
与**已见张数**（`visible` 已有数据，见 `shanten.visible_counts`，`policy.py:783-787`）。
**置信度**：high（读码即可确认缺项）。

### F10. 静默兜底与异常吞没清单（怀疑 7 的结论：**未发现会掩盖错误的分支**）
**文件:行号**：`policy.py:284`、`policy.py:365`、`policy.py:396`、`policy.py:745`、
`policy.py:873`、`runtime/decider.py:104-119`。

**验证结果**（逐项判读）：
- `policy.py:396` 的 `except ValueError: return None`（`_wait_copies`）**是正确设计**：
  `win.winning_draws` 抛 `ValueError`（张数不符），返回 `None` 让调用方跳过并计数
  （`policy.py:800-805` 写 `wait_copies_failed`），**没有**把"算不出"伪装成 0。此处与文档一致。
- `policy.py:284/365` 吞 `ShantenError` 后返回 `None`：调用方分别落到
  `SHANTEN_MAX`（`policy.py:873`，8 向听 ⇒ 该候选几乎必被弃）与"跳过候选"。
  **方向安全**（宁可低估也不高估），但 `_score_discard` 里 `value is None` 时**不留痕**，
  真机上若发生会看不出。建议加 `last_detail["shanten_failed"]` 计数。
- `policy.py:745` `self._break_ties_by_ukeire(...) or best`：tiebreak 返回 `None`（超时/全跳过）
  时静默回落 `total` 第一名。`last_detail` 只在 `wait_copies_failed`/`tiebreak_timeout` 留痕，
  **没有**"tiebreak 整体失效"这一项。建议补一个 `tiebreak_noop` 计数。
- `runtime/decider.py:104-119`：异常/非法动作/超预算**统一降级为 `FirstLegalDecider`**
  （优先胡杠碰吃，其次出牌）。这是有意的护栏（注释：绝不无响应），但它意味着
  **任何 `HeuristicDecider` 异常都会静默变成"打第一张合法牌"**，且 `on_fallback` 回调
  是唯一信号。建议确认线上该回调确实落日志（否则"打第一张"会伪装成策略选择）。

**结论**：怀疑 7 **未证实**——存在的吞没都是窄类型且方向安全；风险在**可观测性**
（缺计数）而非正确性。
**置信度**：high（逐点读码）。

### F11. 默认档与实验档的**开关落地**风险（一次已修复的 Severe 类问题的残留面）
**文件:行号**：`policy.py:32-55`（`VARIANT_FIELDS`）、`policy.py:451-476`（`configure` 用 `replace`）、
`cli.py:97`（`heuristic` 零 override）。
**问题**：历史事故（`configure` 重建 `PolicyConfig` 丢字段 ⇒ 真机静默跑默认档）已修，
但**仍有两个入口不被 `VARIANT_FIELDS` 覆盖**：
① `configure` 只保留 `VARIANT_FIELDS` 里列出的字段——新增开关必须手动加进去，
   否则会被 `replace` 保留（现在用 `replace` 所以安全），但 `_refresh_name()` 不会显示它，
   **日志里看不出真机跑的档位**；
② `ukeire_candidates` 已在表里，但 `wait_aware_tenpai` / `two_ply_shanten1` 的**组合臂**
  （`two-ply`、`shape` 等）只在 `cli.py` 注册，**没有**版本快照（`versions.py` 只有 v1/v2/v3）。
**影响**：可复现性风险："旧默认档"只有 v1/v2/v3 三个快照，其余臂一旦成为默认就无法对拍。
**建议**：新增默认前按 `versions.py` 的纪律补快照；并把 `configure` 的字段保留改为
**自动**（遍历 `PolicyConfig` 的所有字段做 `replace`，而不是手维护 `VARIANT_FIELDS`）。
**置信度**：medium-high（读码可确认；未在真机复现）。

### F12. `god_discard_penalty = 25.0` 是未经标定的硬闸门
**文件:行号**：`policy.py:234`、`policy.py:859`、`policy.py:875-878`（`- god_penalty`）。
**问题**：打出财神的惩罚恒为 25，而其余所有项幅度 ≤ 10（向听）与 ≈ 3（喂牌）。
⇒ 该惩罚**在任何局面下都一票否决**（`25 > 10 + smoke`），等价于"只在无其他可打牌时才打财神"。
`policy.py:236-241` 的注释自陈这是 `preserve_god` 想验证的假设"尚未测过"。
**影响**：与 F4 叠加后，**财神实际上被策略当百搭用掉**（`preserve_god=False` 时
只有 25 这一道，而它只影响"是否打财神"这一项，不影响"是否把财神当面子用"）。
爆头/财飘链（`chase_baotou`、`_baotou_by_discard`）依赖"留一张闲余财神"，
而这条路径由 `_choose_win_or_piao` 负责、与 25 无关 ⇒ 两者口径不统一。
**建议**：把 25 换成"财神作为链货币的边际期望"（`_god_route_penalty` 已是这条路，
`policy.py:922-926`，但在 `route_aware=False` 的默认档里**根本不执行**）。
**置信度**：medium（数值与调用路径可读；"一票否决"是推断，可由一次数值扫描证实）。

---

## 二、疑点待验证（需要测量，不建议直接改码）

### H1. `shape_value` 在副露 ≥1 时**只计 1 张财神加权**，会低估多财神手的形质
**文件:行号**：`shanten.py:434-435`（`if counts[GOD]: weights.append(1.4)`，**单数 God**）、
与 `shanten.py:325-332`（`quick_blocks` 把 `partials += counts[GOD]`，**复数**）不一致。
**复现**（同一骨架 `1234w`，仅财神数不同）：
`gods=0 → legacy 2 / shape 2.0`；`gods=1 → legacy 3 / shape 3.36`；
`gods=2 → legacy 4 / shape 3.36`；`gods=3 → legacy 5 / shape 3.36`。
⇒ **2 张与 3 张财神给出完全相同** 的形质值，而旧键会递增。
当两个候选的财神数不同时，`shape_value` 会**翻转两者的相对顺序**。
**待验**：在真机决策点里，`shape_value` 与 `quick_blocks` 的 argmax 分歧中，
有多少由财神倍数差造成，以及该分歧是否指向更差的进张（用 `tools/probe_shape_ukeire.py` 口径扩到财神分层）。
**建议**：`weights.extend([1.4] * counts[GOD])`（或按 `min(counts[GOD], need)` 计）。
**置信度**：medium（退化已复现＝事实；但"是否造成实质更差选择"未测）。

### H2. `shape_value` 按 `need = SETS_PER_HAND - meld_count - sets` 裁剪时**可能丢掉雀头槽**
**文件:行号**：`shanten.py:437-439`（`need = max(0, 4 - meld_count - sets)`；
`weights.sort(reverse=True); taken = weights[:need]`）。
**问题**：`need` 是"还差几个块"，而其中**必含 1 个雀头槽**；按权重降序取前 `need` 个，
当 `weights` 数 > `need` 时，权重 1.0 的**对子**会被 1.2 的两面或 1.4 的财神挤掉
（雀头本身不显式计权）。
**已复现的对照**：`X=111w222w3w` 与 `Y=111w22w34w`（均 7 张、副露 2 组）目前都给 4.0，
⇒ **我未能构造出实际翻转排序的最小例**（说明真实影响可能小于机制听起来的严重程度）。
**待验**：扫描真机决策点，统计"`need < len(weights)` 且被丢弃的 `taken` 中含
本可为雀头的对子"的比例，再看这些点上进张是否更差。
**建议**：若比例可观，改为"先保 1 个雀头（对子/财神），其余槽位按权重取"。
**置信度**：low-medium（机制为读码事实；无意料中的量级，需测量才可下结论）。

### H3. `natural_shanten` 只在"有财神"时给 1 点，未计入财神张数
**文件:行号**：`shanten.py:243-254`（`- (1 if wildcards else 0)`）。
**问题**：与 F2/H1 同族：财神 1 张与 3 张给出**相同**的 `natural_shanten`。
该方法仅由 `natural_route=True` 的实验档使用（`policy.py:869-872`），默认档不受影响。
**待验**：若 `natural` 臂曾参与排位，其"向听"口径与 `shanten` 不同源，对比时需分层。
**置信度**：low（影响面限于实验档）。

### H4. 截断损失的**净期望**（不是单点进张数）
F3 量的是"丢掉的进张张数"，不是"丢掉的胜率"。需要把 F3 的 11/47 个点接到
到听摸序 / 本局胡牌率上（`agent/verify/recompute_search_v3.py` 的机制门口径）。
**待验**：`ukeire_candidates ∈ {2, 4, 8, 全量}` 四档在同一批局上的到听摸序差。
**置信度**：n/a（未测）。

### H5. `EXACT_UKEIRE_BUDGET_SEC = 0.6` 在向听 ≥2 全算时的可行性
F2 若把 `ukeire_max_shanten` 放大，候选面 ×(向听≥2 的 1914/4428 个点) 会让
精确 `ukeire`（42–157 ms/张）逼近 1800 ms 预算；`policy.py:808-818` 的
deadline 检查是 `... and best is not None`，**第一个候选不设保护**。
**待验**：实测 `ukeire_max_shanten=5` 时每决策耗时分布与 `GuardedDecider` 降级率。
**置信度**：medium（代码路径可读；耗时未测）。

---

## 三、覆盖范围与限制

已核实：`policy.py` 的出牌主路径（`_score_discard` / `_choose_discard` /
`_break_ties_by_ukeire` / `cheap_ukeire` / `_wait_copies` / `_god_route_penalty` /
`PolicyConfig` 常量）、`risk.py` 全文、`shanten.py` 的 `shape_value` / `quick_blocks` /
`ukeire` / `natural_shanten`；并在真机事件流上复现了 F1/F3/F4/F6/F7 的计数。

**未做**（不在本次权限/预算内）：
- 未跑自对弈 A/B，因此**没有**任何"改动会提升胜率"的因果结论；本报告只给机制证据。
- 未验证 `fan`/`score`/`routes`/`win` 的其余内容（未在任务范围内）。
- `HeuristicReadyModel` 的**逐 (副露,弃牌) 分桶偏差**未在本机重算
  （需按房重跑 `tools/calibrate_threat.py`，属 CPU-heavy，已让给正在跑的采集作业）；
  本报告引用的是该工具文档内已落盘的数据（30 房 / n=58663）。
- F1 的分副露表来自 40 房（1695/706/123/5 点），副露 3 组仅 5 点，该行不具统计意义。
- F3 的 n=47 偏小；H2/H4/H5 为明确的未测项。

**读码所依据的默认档事实**：`--decider` 默认 `"heuristic"`（`cli.py:334`），
`heuristic` = `PolicyConfig.for_mode(mode)` 零 override（`cli.py:97`）
⇒ `tiebreak="exact-ukeire"`、`wait_aware_tenpai=False`、`ukeire_max_shanten=1`、
`ukeire_candidates=2`、`feed_weight=3.0`、`shape_value=False`。
