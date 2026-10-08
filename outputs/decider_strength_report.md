# Decider 强度报告（全量日志扫描）

- 日志总数（`logs/a_*.jsonl`）：**987** 份
- 有效扫描：987 份
- 无最终得分的日志：9 份
- 得分口径：每份日志最后一条 `status=finished` 的 `tournament.status` 事件中 ranking 里我方 user_id 的 score（整场 10 局 × 8 轮的总分）
- 低置信阈值：局数 < 30
- 数量口径对齐：此前报过 957 / 987 两数，本次实测 `logs/a_*.jsonl` 共 **987** 份（含 9 份无 finished 状态、无得分，按 0 局计入日志份数但不计入有效场数）。

## 选择规则执行

按拍板口径逐条执行：

1. **`first-legal` 直接剔除** —— 8 场 0 胜，场均 -313.1，远低于任何 heuristic 配置。✅ 已剔除。
2. **优先「近期主导 + 实测胜率高」** —— 见下方时间线。
3. **最新配置胜率不是最高时，以胜率为准** —— `heuristic`（裸配置）虽最近仍在跑（10-07 还有 2 场），但胜率仅 19.9%，不是最高。

### 时间线分布（日志份数 / 天）

| 日期 | [0] ukeire-candidates=3 | [1] heuristic 裸 | [2] wait-aware | [3] 无 ukeire-candidates | [4] piao-threshold | [5] first-legal | 其他 |
|---|---|---|---|---|---|---|---|
| 09-24 ~ 09-28 | 0 | 273 | 89 | 0 | 0 | 8 | 3 |
| 09-29 ~ 09-30 | 0 | 0 | 87 | 28 | 0 | 0 | 0 |
| 10-01 ~ 10-03 | 133 | 0 | 2 | 72 | 56 | 0 | 0 |
| 10-04 ~ 10-07 | 265 | 4 | 0 | 0 | 0 | 0 | 3 |

（[3] = `heuristic[ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]`，即 [0] 去掉 `ukeire-candidates=3`）

**近期主导**（10-04 起，265/269 = 98.5%）= `heuristic[ukeire-candidates=3,ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]`。

## 推荐保留

| 优先级 | 配置串 | 理由 |
|---|---|---|
| **首选** | `heuristic[ukeire-candidates=3,ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]` | ① 样本最大（397 场，占 40.6%）② 近期主导（10-04 起独占）③ 场均得分 -66.0（仅次于 [3] 的 -56.2，但 Welch t=-0.43 不显著，且 [3] 样本仅 99）④ 胜率 26.4%（与 [3] 的 27.3% 相当）⑤ 决策数 747,726，BC 数据量最大 |
| 备选 | `heuristic[ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]` | 场均得分 -56.2 名义最高、胜率 27.3% 名义最高，但样本仅 99 场、且 10-02 后已停用；与首选差异在统计上不显著，并入首选风险可控 |
| 不推荐 | `heuristic`（裸） | 胜率 19.9% 全场最低（剔除 first-legal 后），虽样本 292 场且时间跨度大，但近期几乎不再使用（10-06/07 仅 4 场），说明已被迭代淘汰 |
| 不推荐 | `heuristic[ukeire-preselect=5,...,piao-threshold-scale=1.3]` | 胜率 14.3%、场均 -102.1，调参回退的明确证据（10-02 引入、10-03 停用） |
| 剔除 | `first-legal` | 0% 胜率，纯占位 |

### 拍板建议

**方案 B 数据源 = `heuristic[ukeire-candidates=3,ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]` 的 397 场（747,726 决策）**。

如需扩充样本量，可并入 `heuristic[ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]` 的 99 场（187,925 决策），合计 496 场 / 935,651 决策——两者仅差 `ukeire-candidates=3` 一个开关，行为分布接近。

| decider 配置串 | 决策数 | 日志份数 | 有效场数 | 胜率 | 场均得分 | 中位得分 | 首次出现 | 末次出现 | 建议 |
|---|---|---|---|---|---|---|---|---|---|
| `heuristic[ukeire-candidates=3,ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]` | 747726 | 398 | 397 | 26.4% | -66.0 | -96 | 2026-10-01T01:21:52.222 | 2026-10-06T22:55:39.971 | 候选 |
| `heuristic` | 555874 | 297 | 292 | 19.9% | -85.3 | -101 | 2026-09-24T19:09:40.220 | 2026-10-07T01:06:33.122 | 候选 |
| `heuristic[wait-aware-tenpai=True]` | 230127 | 122 | 120 | 23.3% | -75.3 | -104 | 2026-09-28T14:56:38.271 | 2026-10-01T01:21:18.798 | 候选 |
| `heuristic[ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True]` | 187925 | 100 | 99 | 27.3% | -56.2 | -111 | 2026-09-30T00:59:01.454 | 2026-10-02T15:57:49.079 | 候选 |
| `heuristic[ukeire-preselect=5,ukeire-candidates=3,ukeire-max-shanten=3,ukeire-order=blocks,wait-aware-tenpai=True,shape-value=True,piao-threshold-scale=1.3]` | 103298 | 56 | 56 | 14.3% | -102.1 | -94 | 2026-10-02T16:13:27.788 | 2026-10-03T22:17:32.591 | 候选 |
| `first-legal` | 17195 | 8 | 8 | 0.0% | -313.1 | -327 | 2026-09-27T17:02:29.417 | 2026-09-27T22:01:41.819 | 剔除 |
| `heuristic[meld-tolerance=equal]` | 7273 | 4 | 4 | 25.0% | -142.2 | -126 | 2026-09-26T03:49:39.086 | 2026-10-07T00:54:29.813 | 低置信 |
| `heuristic[tiebreak=exact-ukeire]` | 3698 | 2 | 2 | 50.0% | -66.0 | -66 | 2026-09-26T04:28:36.459 | 2026-09-26T05:15:52.956 | 低置信 |
