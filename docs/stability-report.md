# 7.2 / 7.3 稳定性与边界演练报告

全部离线（假 transport 注入故障，不打平台）。代码：`tests/test_stability.py`。
运行方式：`uv run pytest tests/test_stability.py -v`（当前 9 项全过，约 17 秒）。
重点是**失败时的表现**，每项附观察结果。

## 7.2 网络与平台故障

| 演练 | 注入 | 观察到的行为 | 失败时的表现 |
|---|---|---|---|
| 网络中断 | 连续 5 次 `TransportError` 后恢复 | 每次中断计 `errors+1`、记 `error.transport` 日志、退避 1s 后重试；恢复后正常决策提交，对局完成 | 不崩溃、不丢本地状态；中断期零动作提交（不会把过期局面打出去） |
| 平台 5xx 风暴 | 连续 8 次 `ApiError(500, HTTP_500)` 后恢复 | 未知 code 走可重试分支：计错、退避 0.5s、继续；恢复后对局完成 | 不崩溃；错误全部有日志计数，无静默吞掉 |
| 对局级永久错误 | `ApiError(404, GAME_NOT_FOUND)` | **一次即停**：`run()` 返回 `api:GAME_NOT_FOUND`，不重试轰炸 | 永久错误不计入 `errors`（是终止信号不是故障）；整场只有 1 次请求 |
| 阶段中断重赛 | `stage_open` 下 `stage_crashed` 翻转 | ready 去重键含 `status:stage_status:stage_crashed` 三段，崩溃→恢复每次状态变化都重新到位（实测 ready 3 次） | 若键漏掉 crashed 位，重赛确认会被吞掉导致失格——现有实现正确 |
| 进程被杀重启接管 | 首进程打半场后停；二进程全新内存启动 | 新进程靠 `/api/me active_games` 重新发现对局，首次拉取必为 `seq=0` 全量快照；接管后看到终态收尾 | **本地无持久化状态，也不需要**：平台侧是权威。关键验证：两进程合计只提交 1 次动作，无重复提交 |

## 7.3 时序边界

| 演练 | 注入 | 观察到的行为 | 失败时的表现 |
|---|---|---|---|
| 局间 settled 停顿 | 连续返回 `phase="settled"` 快照 | `_absorb` 识别 settled 并记 `game.settled`，**不用于牌型计算、不提交动作**（实测 0 次提交） | settled 残牌不会触发误决策 |
| 跨局长轮询 pending | 连续 `{"pending":true}` ×4（阈值=2） | 增量轮询每满 `DEFAULT_LONG_IDLE_RESYNC_ROUNDS` 次自动发一次 `seq=0` 全量重同步；不误判掉线/结束 | 恢复事件流后决策提交正常；重同步请求路径已断言（`seq=0`） |
| 动作冲突竞态 | 提交被 `409 INVALID_ACTION` 拒绝 | 记 `action.rejected`（含当时 phase/round_no），**不原地重试**，不计入已提交，也不算 errors | 竞态被视为正常事件；对局继续直至完成 |
| 多场并发决策窗口 | 两局同时待决策，`max_in_flight=1` | 两局各自完成、各提交 1 次（窗口键按局隔离，互不串）；全程在飞请求峰值 ≤1 | 闸门生效；`RuntimeOptions.max_in_flight` 卡住总并发 |

## 附带发现（观测性，非崩溃级）

1. **`games_completed` 口径偏宽**：`Runtime._reap`（`src/majiang/runtime/engine.py:653`）对任何
   退出的 future 一律 `games_completed += 1`，包括被 `request_stop` 中止的（返回 `"stopped"` 的）runner。
   被杀进程的对局也会被记为「完成」。赛后看 summary 时勿把该字段当「打完的场数」；建议按
   `future.result()` 的 reason 区分（`finished` / `stopped` / `api:*` / `error`）。
2. **5xx 退避无上限**：可重试错误（含未知 code）的循环没有最大重试次数，靠外层
   `duration_sec` / stop 事件兜底。正式比赛有对局限时，风险可接受；记录于此。
