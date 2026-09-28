# Spec Delta

## Purpose

定义**整场回报预测器**能力：由当前**公开局面**预测本场最终名次分，作为残差式 RL 的**代替学习信号**（Suphx「global reward prediction」思想的本地化）。用于**筛掉无信息方向**，而非直接充当策略。

## ADDED Requirements

### Requirement: 公开信息特征输入

预测器 SHALL 只使用**公开信息 + 本人手牌**（主仓 `features.extract` 的 29 维），MUST NOT 使用对手手牌、MUST NOT 实现或调用 `observe_state`。

#### Scenario: 合规性
- **WHEN** 构造特征
- **THEN** 特征全部来自 `Situation` 的公开字段与本人手牌；代码中不存在读取他人暗牌的路径

### Requirement: 标签为整场回报

预测器的训练标签 SHALL 是**整场**（默认 8 局）累计名次分 `place_points`，MUST NOT 使用单局得分。

#### Scenario: 与单局得分区分
- **WHEN** 收集训练样本
- **THEN** 同一场内所有出牌决策共享同一个"整场最终名次分"标签

### Requirement: 按场聚簇的显著性判据

预测质量 SHALL 用 **held-out 上的 Pearson r** 度量，其 95% 置信区间 SHALL 通过**以"场"为重采样单位**的 bootstrap 得到（同一场内决策高度相关，不得按决策独立重采样）。

#### Scenario: 判据预登记
- **WHEN** 判定 Go/No-Go
- **THEN** 判据在任何数据产生前写死：CI **不含 0 且 > 0** 为 GO；否则 NO-GO

#### Scenario: 弱信号处理
- **WHEN** GO 但效应微弱（如 R² 在 1% 量级）
- **THEN** 结论必须显式标注"仅用于筛掉无信息方向，不可直接当策略"，并给出后续升级判据

### Requirement: 反假阴性的对照

报告 SHALL 至少包含两项对照，以区分"信号弱"与"模型/特征无信息"：① **训练集上的 r**；② **按局面进度分桶的 r**。

#### Scenario: 训练集对照
- **WHEN** 验证集 r 接近 0
- **THEN** 必须同时报告训练集 r；若训练集 r 也接近 0，则结论为"模型/特征无信息"，而非"过拟合"

### Requirement: 可复算产物

每次运行 SHALL 落盘 JSON（配置、样本量、r 与 CI、R²、对照项、随机种子、环境版本），并在文档中给出路径。

#### Scenario: 复现
- **WHEN** 他人按文档重跑
- **THEN** 使用显式 `--seed` 可得同量级结论，且产物路径与文档一致
