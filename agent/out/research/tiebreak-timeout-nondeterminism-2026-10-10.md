# 引擎非确定性：破平层 `exact-ukeire` 墙钟上限在负载下改变结果（2026-10-10）

> 执行：B'（coordinator）独立复核车道。触发 = 用户 12:0x 新报障 3 例（seq21/23/24）。
> **零 `src/` 改动**；本轮只读复现 + 采样，未触碰任何在途实验。

## 结论（一句话）

**`HeuristicDecider` 对给定局面不是确定性的。** 当破平层进入 `exact-ukeire` 分支且
机器负载高时，`_break_ties_by_ukeire` 的 **0.6 秒墙钟上限**（`EXACT_UKEIRE_BUDGET_SEC`）
会**提前 `break`**，于是返回**已算出的最优**（按 `blocks` 截断后的候选面里、进张最多的一张）
——这一步**依赖外部 CPU 负载**，从而改变最终出牌。这与本项目多处文档
（如 `tools/replay_report.py`「引擎对给定局面是**确定性**的 ⇒ 事后可逐位复现当时的建议」）
的假设直接冲突。

## 复现（用户报障 seq23 = `report_20261010_120618_seq23`）

- 局面：向听 2，破平层 `tiebreak_total_slack = inf`（默认口径），cap = `ukeire_candidates` = 3。
- **无负载**：单次决策 **0.048–0.083 s**（≪ 0.6 s）⇒ 每次都选 **8t**（`tiebreak=向听 2 并列 3 张，精确进张 17 张`）。
- **负载下**（机器 `load average ≈ 137`，16 核，A 的在途实验占用）：同一输入反复采样 12 次得到
  - **8t（`timeout=None`，进张 17）× ~10**
  - **4w（`tiebreak_timeout=True`，进张 15）× 2**

即：**同一局面，同一 arm，输出在 8t / 4w 之间摆动**；改变的是**负载**，不是输入。

### 机制

1. `_choose_discard` 按主分排序，取 `scores[0].shanten` 层为并列层（本局 4 张：`8t/9w/8w/4w`，均向听 2）。
2. `exact-ukeire` 分支把候选面按 `blocks` 降序截断到 cap=3 → `[8t, 9w, 8w]`（`4w` 被截掉）。
3. 对 `tied` 逐张算精确进张；循环内含
   `if deadline is not None and time.monotonic() > deadline and best is not None: self.last_detail["tiebreak_timeout"]=True; break`（`deadline = start + 0.6s`）。
4. **无负载**：3 张在 0.08 s 内算完 ⇒ `best = 8t`（进张 17）。
   **有负载**：单次仍只 ~0.08 s，但进程被调度器抢占 ⇒ 循环墙钟越过 0.6 s ⇒ 在第 k 张后 `break`，
   此时 `best` 可能已是别的候选（采样中为 `4w`，`tiebreak_timeout=True`）。
5. ⇒ 墙钟 deadline 把**外部负载**变成了**决策输入** ⇒ 破平层结果不确定。

### 同一机制的「失效模式」也适用于 seq21

seq21（`report_20261010_120431_seq21`，向听 3，并列 **11** 张，cap=3）：

- `blocks` 降序前 3 = `[4w, 7w, 1b]`（三者 `blocks` 并列 4.135，次序由 `sorted` 稳定性 = 最小牌索引决定）。
- `7b`（用户首选）`blocks` 也 = 4.135，但**落在 cap 之外**，其精确进张 **57 = 全场最高**；
  而 cap 内最高仅 `4w` 的 52 ⇒ 破平层在**被截断的子集**上选最优，漏掉了更好的 `7b`。
- 这正是 `v7-tiedfull`（A 11:20 落地、11:35 关闭）针对的截断失效；本轮确认它在**用户真实报障**上可复现。

## 影响评估

- **实验可复现性**：`tools/replay_report.py`、以及所有以「引擎确定性」为前提的离线对拍，
  在**高负载**（本机长期 `load ≫ 核数`）下可能读出与真机/低负载不同的结果。
  本轮 seq23 在两次 `replay_report.py` 连测的多臂运行中即出现 `v5→4w` 与报告时 `v5→8t` 的不一致。
- **在途实验污染（推断，未直接测量）**：本机所有 `--jobs N` 的并行 arm（`v7-pairs` 场地 A/B、
  `cf_point_mc`、`trigger_counterfactual`）在 `load ≈ 137` 下并发跑，**每个 worker 都可能触发该超时**；
  若基线/处理两侧触发率不同，结果里会混入**负载噪声**。**当前未见任何人量化过这一点** ⇒ 建议复核。
- **不涉及默认档 `v5`/冠军档 `v7` 的常量**：`EXACT_UKEIRE_BUDGET_SEC=0.6` 与平台的 3 s 硬预算
  （10:30 帖「本地 3 会话 4,264 决策 p50 0.1 / p99 52.9 ms」）之间仍有 ~50× 余量 ⇒ 平台安全无虞；
  问题在**离线复现/实验**，不在真机。

## 建议（供 A 裁决；我不改 `src/`）

1. **可复现性判据**：把 `tiebreak_timeout=True` 视为**结果无效**，而非静默回退——
   离线工具在读到该位时应**重跑或丢弃该点**，而不是当成一次正常决策。
2. **两种修法**（择一，均默认关以确保不动回归）：
   - (a) 提高/移除该墙钟上限（代价：高负载下决策变慢，但仍有 3 s 余量）；
   - (b) `deadline` 只用**该决策内部的 CPU 时间**或**确定性预算**（与负载解耦）。
3. **在途实验**：若要对本日结论做交叉复核，应在 `load` 低的窗口复跑，或记录每点
   `tiebreak_timeout` 比例作为噪声指标。

## 复现脚本（临时，未入库）

```bash
# 无负载 vs 负载下同一局面采样
for i in $(seq 1 12); do
  .venv/bin/python - <<'PY'
import sys,json; sys.path.insert(0,'src'); sys.path.insert(0,'tools')
import replay_report as RR
from majiang.cli import make_decider
from majiang.strategy.policy import Mode
from majiang.rules.action import legal_actions
from majiang.rules import tiles as T
d=json.load(open('webapp/reports/report_20261010_120618_seq23.json'))
sit=RR._situation_from_view(d['decision']['situation']); acts=legal_actions(sit)
dec=make_decider('v5', Mode.QUALIFIER); dec.choose(sit, acts, budget_ms=3000)
print(T.to_code(dec.last_detail['discards'][0].split()[0]) if False else dec.last_detail.get('tiebreak_timeout'))
PY
done
```

## 关联

- `src/majiang/strategy/policy.py`：`EXACT_UKEIRE_BUDGET_SEC=0.6`（L103）、`deadline`（L1242）、
  `tiebreak_timeout`（L1267 / L1272）、`_break_ties_by_ukeire`（L1173+）。
- `tools/replay_report.py`（「确定性」假设）。
- 用户报障：`webapp/reports/{report_20261010_120431_seq21,report_20261010_120618_seq23,report_20261010_120735_seq24}.json`。
- 相关轴：`v7-tiedfull`（A 已关闭，`10c768f`）、`v5-tieslack0`（A 已裁决关闭，`7614159`）。
