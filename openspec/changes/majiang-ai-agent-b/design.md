# Design

## Context

仓库已有 A 的策略主线与测量工具，但 A 的工具链出过「多局污染」级 bug（一份事件流 = 一场 8 局，被当成一局跑）。事件流实测特性：`blocks` 是按 ≤128 事件切分的块，同一 `round_no` 可跨多个 block，仅每局首 block 带非空 `start_hands`（庄家 14 张）；`round_ended.data` 含 `draw/fan/scores/detail`；`gang.data.kind ∈ {ming, bu, an}`；`tile_drawn` 的牌面对全场可见；`game_ended` 数量少于文件数（数据在增长，存在未完成对局）。硬约束见 `notes/OWNERSHIP.md`：不改 A 的文件、不打平台、不杀 A 进程、凭据只走环境变量。

## Goals / Non-Goals

**Goals:**
- 独立复算：从原始 JSON 重建每局牌面并验证不变量，再算指标；实现与 A 零共享代码
- 稳定性演练离线可复现，一条命令跑完，输出「失败时的表现」
- 提交物在干净 venv 可验收

**Non-Goals:**
- 不改任何策略/运行时代码（那是 A 的所有权）
- 不做新的策略实验或 A/B
- 不为复算建通用框架——脚本能回答问题即可，不做过度抽象

## Decisions

1. **复算放 `verify/`，纯标准库单文件脚本。** 每脚本一个问题（`verify/invariants.py`、`verify/metrics.py`、`verify/spotcheck.py`）。
   备选：一个可配置大脚本——否决，单脚本单职责让「不一致定位」更容易，也避免我自己引入共享 bug。
2. **重建状态机自己写**：手牌 = `start_hands` − 出牌 − 副露（chi 不重复拼接 `tile`）+ 摸牌；牌墙 = 136 − 起手 − 可见摸牌。杠的处理：ming（碰后摸牌杠）/bu（补杠）/an（暗杠）都使 4 张离场入杠区，补牌从牌墙尾/盲牌区——**守恒校验只数「全场 ≤4」不追踪墙序**，避免对平台摸牌顺序做假设。
   备选：照抄规则引擎——否决，违反「独立」要求，且规则引擎针对决策而非审计。
3. **名次分按 `rounds[]` 汇总分排序**（与 A 口径独立的第二来源：`blocks` 内 `round_ended.data.scores` 累加）。两来源不一致本身就是要报告的信号。
4. **稳定性演练用 `tests/test_stability*.py`（pytest）+ 假 transport**。假 transport 实现 client 接口的可控故障注入（连接拒绝、固定 5xx、409、pending 挂起）。进程被杀演练用快照文件 + 子进程重启模拟。
   备选：mock 平台 HTTP 服务器——更真但更重；client 接口边界注入已足够覆盖 7.2/7.3 场景。
5. **《使用说明》放 `docs/USAGE.md`**（评审版），与 A 的 `README.md`（开发版）并存，互不覆盖。干净环境验收脚本放 `scripts/verify_clean_env.sh`。

## Risks / Trade-offs

- [复算脚本与 A 犯同类错误（如「一文件=一局」假设）] → 不变量层不依赖任何解析假设，直接数牌；三层验收强制手算对照
- [数据持续增长导致数字漂移，与 A 的对比失真] → 每次运行记录文件数与总 seq 范围作为快照指纹，汇报时附上
- [假 transport 与真实 client 接口漂移] → 演练只依赖 client 的公开接口；若接口变动测试会红，即为信号
- [未完成对局（无 `game_ended`）污染名次分] → 名次分只对 `status=="finished"` 且有 8 个 round_ended 的文件计算，其余单独计数报告

## Open Questions

- `timeout` 事件（988240 次）中 `kind:"discard"` 是否意味着该次出牌是超时强制出牌？这影响「胡牌时已出手数」的语义解读。可延后：先按出牌事件计数，口径写进报告。
