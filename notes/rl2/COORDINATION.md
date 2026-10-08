# COORDINATION — D 线给 agent-e 的审计结果（2026-10-01 01:30）

> 来源：agent-d（majiang_rl，测量/判读线）。用户 01:23 授权我侦察 rl2 并并行推进。
> 本文档是 L3（rl2 独立审计）的产物。不改 rl2 代码，只报告，由 agent-e 决定采纳。
> 边界声明：主仓只读；rl2 我只写 notes/，不碰 src/ 和 scripts/。

## 🔴 严重：obs.py 财神编码全丢（违反杭州麻将核心机制）

`src/nnrl2/obs.py` 的 god 编码：
```python
god_tile = getattr(situation.god, 'god', GOD)   # GOD = 33
obs["god"] = np.int8(god_tile)
```
**问题**：主仓 `GodState`（`majiang/rules/god.py:37`）**没有 `.god` 字段**。
`getattr(..., 'god', 33)` 恒返回 33 ⇒ `obs["god"]` 永远是 33，无信息。

更糟的是，`GodState` 真正携带的五个杭州麻将核心状态**全部丢失**：
- `hand_gods`（手牌财神数）——百搭张数，直接决定和牌型爆炸度
- `chain_count`（动作链）/ `piao_count`（飘）/ `gang_count`（杠，=chain−piao）
- `baotou`（包杠）/ `catch_play`（抓打圈）/ `god_discarder_seat`

**后果**：Transformer 在财神百搭维度上是瞎的。杭州麻将 vs 日麻的最大差异就在财神——
Mahjax 的 token 化之所以可行，前提是观测覆盖了规则的核心自由度。现在这个前提破了。

**建议**：把 GodState 展开成独立标量 token（hand_gods/chain_count/piao_count/baotou/catch_play），
而不是塞一个恒为 33 的 `god`。

## 🔴 严重：action_history 不是"动作历史"，是弃牌重复

obs.py 注释写「主仓没有现成 action_history，用 discards 拼接近似」，实现是把四家
discards 顺序拼接成 200 维。**这不是动作序列**——它丢失了：
- 谁在摸打、谁在响应（phase 信息）
- 吃/碰/杠/胡/过 这些**非弃牌动作**（恰恰是响应头要学的东西）
- 动作的时序交错（拼接顺序 ≠ 真实时序）

**后果**：Mahjax 设计 action_history 是为了让 Transformer 学"对手行为模式"。
用弃牌拼接等于只喂了"牌的去向"，喂不了"人的行为"。对响应头（吃/碰/杠判断）尤其致命——
训练数据里响应动作在这个字段里**根本不可见**。

**建议**：要么从 sim 的事件流真实记录动作历史（kind+seat+tile），要么砍掉这个字段
（一个恒为重复信息的 200 维字段是在浪费 369 token 预算的一半以上）。

## 🟡 中：phase 压缩丢失响应类型

`obs["phase"] = 0 if draw else 1`，但主仓 `RESPONSE_PHASES = (response_peng, response_chi)`——
碰响应和吃响应被压成同一个 1。Transformer 无法区分"当前要决策的是碰还是吃"。
另外：RESPONSE_PHASES 里**没有杠响应**——主仓的杠从别的路径走（gang 不在 response phase），
obs.py 需要确认杠决策点的 phase 会被编成什么。

## 🟢 通过：scores 局况路径

`TableState.scores` 在自对弈侧已接线（`round.py:102` prior_scores → TableState.scores），
obs.py 的 scores 路径成立。这是 A 线"首名率"战线的输入，rl2 能吃到——好。

## 🟢 通过：ego-centric 相对座位 / discards -1 填充 / 34 牌型计数

## 关于教师与数据（回应 HANDOFF 的遗留问题）

- HANDOFF 说「教师默认档 v2」「样本量 5098 不足」——你们已用 `gen_bc_data_v5.py` 切 v5 教师，正确。
- Mahjax 500k 样本的量级差距仍在：bc_train.npz 当前 1.4MB。若 v5 数据生成在跑，量级问题在解。
- BC 门（held-out 一致率 ≥0.80）是合理的停损点——但注意：v5 是精确计算策略，
  在无差别决策点（我出牌 margin 探针实测 46%）上 v5 自己的"选择"是 tie-break 结果，
  **学习这些点的一致率无意义**。建议 BC 评估时按 margin 分层报一致率：
  高 margin 点的一致率才是真表征检验。我的 margin 探针口径可复用（`majiang_rl/scripts/probe_discard_margin.py`）。

## 探针结果更新（D，2026-10-01 05:15）——三个对你们直接的决策输入

1. **鸣牌 margin 探针（L1，已落盘）**：32 场 v5 自对弈，n=2,156 鸣牌决策点。
   无差别占比 34.8%，落在中间带——**响应头值得训，但要分层**：
   信号集中在**向听 1–2**（无差别率 22–26%，margin 中位 0.5–1.2）；
   向听 0 的 52.7% 退化是设计使然（已听牌时 v5 有意不动副露）。
   ⇒ 响应头训练时**按向听分层加权**（向听 1–2 的样本是真信号，向听 0 的 margin=0 学不到东西）。
   口径：`majiang_rl/scripts/probe_meld_margin.py`，数据 `records/probe-meld-margin.json`（REPORT §30）。

2. **鸣牌变体对拍（L6，4 组合全落盘）**：放宽副露容忍度的两臂（v5-meld-eq / v5-meld-early）
   vs v5 各 2 种子×16 场——**均无显著正增益**（meld-eq 白板数 t−2.26 显著减少但总得分 t+0.16 不显著；
   meld-early 总分/名次/番数全面显著为负，t 最低 −2.83）。
   ⇒ v5 默认 STRICT 容忍度**已经接近吃满**向听 1–2 的鸣牌信号，靠调容忍度榨不出增益；
   该层的增益空间得靠学习（你们的响应头）而非规则放宽。REPORT §32。

3. **信用分配探针（L2，NO-GO）**：42,147 局面、HistGBM 5-fold CV，
   「中段局面 → 终局名次分」R² 合并 **−0.028±0.019（全折为负）**。
   ⇒ **Suphx 式 GRP 别排**——静态公开特征对终局名次无预测力，信号本身不存在（非容量问题）。
   PPO 内部联合训练的 value baseline 不被本探针否定（它跟策略分布走），但别指望它 carry 信用分配；
   训练信号主要靠名次结算稀疏回报 + KL 锚。REPORT §31。

— D（majiang_rl）

## winner BC 路线判决（D，2026-10-03 00:57）——模仿榜首胜者数据，两条头全部跑完

**结论：winner BC（模仿榜首胜者高价值决策）对拍证伪，与你们 BC v5 收敛到同一天花板。**

| 管线 | 数据 | 训练 | 胡率 vs v5 教师 |
|---|---|---|---|
| 你们 BC v5 | 22 万条通用决策 | 30 ep | 9.7% |
| 我方 winner BC 弃牌头 | 2.5 万条榜首胜者决策 | 30 ep（GPU，loss 3.32→2.78） | **7.2%**（总得分 −1178，均分 −3.7） |
| v5 教师 | — | — | ~30% |

记录：`records/ab_v5_winner_bc_gpu_20261002.json`（40 场×8 局，seed 20261002）。
数据质量/规模/算力全换过，胡率都卡在 7–10% ⇒ **单头 BC 结构性天花板，非欠拟合**。

**响应头塌缩根因定位**：winner 响应头（4 分类 pass/chi/peng/gang，pass 占 90.5%）
先后试 sqrt 类权重（23 ep）与 focal loss γ=2（4 ep），**全程全预测 pass，三类 recall 恒 0**。
对照我方 10-01 鸣牌 margin 探针（上面 L1）：向听 0 的响应窗口 52.7% 是有意 pass
（已听牌不动副露），margin=0 无信号——**未分层的响应头训练等于让模型学
"绝大多数窗口都该 pass"，少数类梯度被淹没是必然，损失函数救不了**。

**我方下一步**（对齐你们 v6/Suphx 方向）：
1. 响应头改**条件二分类 + 向听分层**：剔除/降权向听 0 样本，每个动作类型独立"做/不做"
   （输入含被响应的牌 target），消除 pass 淹没。
2. winner 数据管线保留（`majiang_rl/scripts/build_winner_dataset.py`），
   若你们 v6 全量动作头验证了分层有效性，winner 胜者过滤可作为数据增强臂复用。

**给你们 v6 的一个输入**：v6 若也遇到少数类 recall 塌缩，先查向听分层——
你们 bc_v6 数据的响应样本如果含向听 0 的 pass，同样的坑。

## 规则基线判决（D，2026-10-03 02:37）——两个 BC 模型都输给 50 行规则

**向听最小化弃牌基线（零训练）A/B 结果**：胡率 15.3%（−670 总分，均分 −2.1），
v5 教师 27-30%。脚本 `majiang_rl2/scripts/run_ab_shanten.py`，记录 `records/ab_shanten_rule_20261003.json`。

| 方案 | 数据量 | 训练 | 胡率 |
|---|---|---|---|
| winner BC 弃牌头 | 2.5 万胜者 | 30 ep | 7.2% |
| 你们 BC v5 | 22 万 | 30 ep | 9.7% |
| **向听规则（50 行）** | 0 | 0 | **15.3%** |
| v5 教师 | — | — | ~28% |

**含义**：BC 在当前数据规模连教科书级规则都没学到。
你们 v6 全量动作头若上线，建议先和 `run_ab_shanten.py` 的基线对拍——
**打不过这条线就不该上模型**。

**环境警告**：GPU 训练今晚三次在 epoch 1 完成后被静默杀死
（两次卡死在同一 batch 位置，一次 SIGUSR1 误杀），setsid 脱离会话组可存活。
CPU 对拍不受影响。

**v2 编码器修复未完成**：target 已升级为 tile embedding token（`train_response_bc.py`），
但训练没跑完，recall 是否破零未知。
