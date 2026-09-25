# Proposal

## Why

Agent A 的策略结论在本轮已被数据推翻四次（自对弈判据有偏、听牌率重建 bug、chase_baotou 误判、小样本噪声）。A 复算自己的数字用的是自己写的脚本，同一个 bug 会在两处犯同样的错。比赛 10/8 12:00 截止提交，需要一个**独立于 A 的工具链**从原始事件流复算全部关键指标，同时补上 A 没时间的空白：稳定性演练与提交物验收。

## What Changes

- 新增 `verify/` 独立复算脚本集：**从原始 JSON 重写**，不 import `tools/measure_strength.py`、`tools/analyze_declined_wins.py`、`src/majiang/sim/replay.py`。
  - 第一层：结构性不变量校验（每种牌全场 ≤4 张、总量 136、出牌必在手），先于一切指标。
  - 第二层：指标复算——手数/胜率、公平份额（有胡率÷4）、名次分布与名次分、均番对比、副露次数、胡牌时已出手数。
  - 第三层：手算小样本对照（随机 10 局人工核对）。
- 新增 `scripts/` 与 `tests/test_stability*.py`：7.2/7.3 稳定性与边界演练，全部离线（模拟器 + 假 transport，不打平台）。覆盖网络中断、连续 5xx、`stage_crashed` 重赛、进程被杀重启接管、局间 settled 停顿 409、长轮询 pending、动作竞态、多场并发决策。
- 新增 `docs/` 提交版《使用说明》+ 干净 venv 零依赖验收脚本。
- 追加式汇报写入 `notes/agent-b.md`（不改历史行）。

## Capabilities

### New Capabilities

- `independent-verification`: 从原始事件流独立复算比赛指标的能力，含结构性不变量校验、指标计算、小样本人工对照三层验收。
- `stability-drills`: 离线故障注入与边界演练能力，验证运行时在网络/平台/时序异常下的行为（失败时的表现为核心交付）。
- `submission-docs`: 干净环境验收与面向评审的《使用说明》产出能力。

### Modified Capabilities

（无）

## Impact

- 新增目录：`verify/`、`scripts/`、`docs/`、`tests/test_stability*.py`、`notes/agent-b.md`
- **不触碰**：`src/majiang/strategy/**`、`src/majiang/runtime/**`、`src/majiang/sim/replay.py`、`tools/measure_strength.py`、`tools/analyze_declined_wins.py`
- 不启动任何访问比赛平台的进程（A 的两个长跑进程占满令牌桶）
- 无第三方依赖新增：`verify/` 与稳定性脚本只用标准库 + 项目已有依赖
