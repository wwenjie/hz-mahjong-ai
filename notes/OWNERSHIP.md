# 文件所有权与交接约定

两个 agent 同时在这个仓库上工作。**约束的目的只有一个：任何时刻，同一个文件只有一个所有人。**

## 所有权表

| 范围 | 所有人 | 说明 |
|---|---|---|
| `src/majiang/strategy/**` | A | 策略主线，正在改吃碰闸门 |
| `src/majiang/runtime/**` | A | 对局循环 |
| `src/majiang/sim/replay.py` | A | 事件流重建 |
| `tools/analyze_declined_wins.py` | A | 反事实评估 |
| `tools/auto_session.py` / `watch_process.py` | A | 长跑与观测，**正在运行中** |
| `tests/test_replay.py` | A | 与 replay 同步 |
| `docs/**` | B | |
| `scripts/**` | B | 稳定性脚本 |
| `tests/test_stability*.py` | B | 7.2 / 7.3 演练 |
| `verify/**` | B | 独立复算脚本（**不要复用 A 的工具**） |
| `notes/agent-b.md` | B | 追加式记录 |
| `notes/agent-a.md` | A | 追加式记录 |

交界文件（**只追加，不改别人的行**）：`notes/OWNERSHIP.md`、`notes/HANDOFF.md`。

## B 的硬约束

1. **不改** `src/majiang/strategy/**`、`src/majiang/runtime/**`、`src/majiang/sim/replay.py`
2. **不启动任何打平台的进程**。A 的 `auto_session.py` / `watch_process.py` 正在跑，共用同一个令牌；
   平台限速按用户 16/s，两个采集器必然互相挤兑。需要动平台请先问。
3. **不杀 A 的进程**：`pgrep` 匹配前先确认 pid 与启动时间，`auto_session.py` 的匹配会命中
   自己的 bash wrapper（踩过）。
4. 提交前 `git pull --rebase`；**禁用** `git reset --hard`、`git checkout -- .`（会抹掉对方未提交的工作）。
5. 一次只提交一小块。

## 关键教训（A 已犯过的错，B 复算时请重点怀疑）

这轮 A 有四次断言被后续数据推翻，每次都是被数据而非推理纠正：

| A 曾断言 | 实际 | 教训 |
|---|---|---|
| 自对弈说「策略已到局部最优」 | 真机证明我们是最弱的打法 | **自对弈是有偏的判据**，不能用来判定强弱 |
| 我们听牌率 83.6%，是活跃玩家最低 | 真值 21.3% | 事件流重建曾把 8 局铺在同一局面上跑，**派生指标全部作废重算** |
| `chase_baotou` 白丢胜率 | 反事实算出净赚 +3.8 分/局 | 假设要用反事实检验，不要靠直觉 |
| A/B 中 `no-chase` 有 2.8 点优势 | 只有 6 场/臂，p≈0.10 | **看样本量再下结论** |

因此 B 的复算**必须自己从原始事件流写代码**，不要 import A 的工具，否则同一个 bug 会在两处犯同样的错。
