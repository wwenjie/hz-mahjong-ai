# Spec Delta

## Purpose

从原始事件流 JSON 独立复算比赛关键指标（胜率、公平份额、名次分、均番、副露、胡牌速度），以结构性不变量作为不依赖任何推理的验收层，用于纠正或确认 Agent A 的策略结论。

## ADDED Requirements

### Requirement: 事件流独立解析

复算工具 SHALL 直接读取 `data/auto_sessions/*/events/*.json` 原始文件，MUST NOT import 或复用 `tools/measure_strength.py`、`tools/analyze_declined_wins.py`、`src/majiang/sim/replay.py` 中的任何解析逻辑。局边界 SHALL 由 `blocks[].round_no` 变化判定，每局起手手牌 SHALL 取自该局首个带非空 `start_hands` 的 block。

#### Scenario: 一份文件多场对局
- **WHEN** 解析一份包含 8 局、每局事件被切分为多个 block 的事件流文件
- **THEN** 工具按 `round_no` 变化切出 8 局，每局使用各自的 `start_hands`，跨局不共享牌面状态

#### Scenario: chi 事件去重
- **WHEN** 处理 `chi` 事件（`data.tiles` 已包含被吃的那张牌）
- **THEN** 副露只计入 `data.tiles` 一次，不再额外拼接 `tile` 字段

#### Scenario: 座位 0 保留
- **WHEN** 处理 `seat` 为 0 的事件
- **THEN** 该事件被正常计入座位 0，不被默认值逻辑丢弃

### Requirement: 结构性不变量校验

在计算任何指标之前，工具 SHALL 对每一局验证：全场每种牌合计不超过 4 张、四种牌合计 136 张、每次出牌时该牌在出牌者手中。任一不变量被违反的局 MUST 被逐条列出（文件、round_no、seq、牌），且该局的指标 MUST NOT 计入汇总。

#### Scenario: 全量数据守恒
- **WHEN** 对全部事件流文件运行不变量校验
- **THEN** 输出违反总数与逐条明细；若违反数大于 0，汇总指标标记为不可信并退出码非零

### Requirement: 指标复算

工具 SHALL 输出：我方（user_id `u_a7f7c67bb14a`）手数/胜次数/胜率、有胡率与公平份额（有胡率÷4）、按房四家总分排序的名次分布与名次分（+3/+1/−1/−3）、我方与其余玩家均番、每局副露次数（吃/碰/杠合计）、胡牌时已出牌手数。

#### Scenario: 流局计入有胡率分母
- **WHEN** 一房 8 局中存在流局（`round_ended` 且 `draw=true`）
- **THEN** 有胡率 = 有赢家的局数 ÷ 总局数，公平份额 = 有胡率 ÷ 4，不以「其他玩家胜率均值」为基准

#### Scenario: 不一致定位
- **WHEN** 复算结果与既有报告值不一致
- **THEN** 工具能输出差异对应的具体文件、round_no 与事件 seq，而非仅给出不同总数

### Requirement: 手算小样本对照

工具 SHALL 支持随机抽取 10 局并打印可供人工核对的明细：每手手牌张数、各玩家副露、牌墙剩余张数、赢家与番数。

#### Scenario: 人工核对
- **WHEN** 对抽取的 10 局逐局比对工具输出与原始 JSON
- **THEN** 手牌张数、副露数、牌墙剩余、赢家四项全部一致，否则定位到具体事件
