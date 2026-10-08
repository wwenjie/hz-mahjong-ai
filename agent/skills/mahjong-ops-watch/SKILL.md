---
name: mahjong-ops-watch
description: 只读巡检比赛进程与日志新鲜度，异常时给出含退出原因与最后事件的告警，对平台零请求。需要值守或排查进程异常时使用。
---

# 只读值守与告警

## 何时用

- 比赛进程疑似掉了、卡了、或日志不再更新。
- 要一份「现在到底健不健康」的结论。

## 命令

```bash
uv run python agent/patrol/patrol.py --once        # 单轮巡检（前台，给结论）
uv run python agent/patrol/patrol.py               # 常驻循环（写 agent/out/）
tail -20 agent/out/alerts.log                      # 告警队列
cat agent/out/patrol.status                        # 心跳（alive_at / 各类计数）
```

## 巡检项

| 项 | 判据 | 告警级别 |
|---|---|---|
| 目标进程存活 | pid 存在且 cmdline 匹配（先核对 pid 与启动时间） | 高 |
| 平台在线判据 | 最后一次已认证请求距今 < 90 秒 | 高 |
| 错误/超时事件速率 | 窗口内 `error` / `timeout` 计数超阈值 | 中 |
| 守护熔断状态 | 守护脚本连续快速失败计数命中熔断 | 需人工介入 |
| B 的守护心跳 | `verify/out/watch.status` 的 `alive_at` 未超时 | 低 |

## 硬规则

1. **只读。** 不修改 `verify/**`、`scripts/**`，不写别人的文件；自己的产物只落 `agent/out/`。
2. **对平台零请求。** 不要为了确认在线去主动打平台接口——那是 A 的令牌配额。
3. **不自动重启任何进程。** 重启只由 `scripts/supervise.sh` 和 systemd 单元负责。
4. **同一根因只告警一次**，人工确认（`--ack <根因键>`）后静默，避免刷屏掩盖新问题。
5. **终态不误报**：进程以退出码 0 结束、或平台返回 `TOURNAMENT_CLOSED` 类终态时，
   不产生「进程死亡」告警，只给收尾摘要（运行时长、各身份汇总、未处理异常）。
6. **状态落盘**（`agent/out/patrol.state.json`），跨重启恢复计数——否则「重启后计数器归零」会变成静默偏置。

## 验收

- 单轮巡检输出含各项判据的实际数值与结论。
- 连续跑一个观察窗后：平台请求计数为 0，`agent/out/patrol.log` 无由巡检发起的进程启动/重启记录。
