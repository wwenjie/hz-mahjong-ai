---
name: mahjong-log-patrol
description: 按事件类型检索本仓库的 JSONL 日志与事件流，汇总计数与时间范围，判断进程是否满足平台在线判据。需要巡检或排查异常时使用。
---

# 日志巡检

## 何时用

- 想知道「最近有没有异常 / 超时 / 断线」。
- 要确认比赛进程是否仍满足平台在线判据（最近 90 秒内有已认证请求）。
- 要统计某类事件的数量与时间分布。

## 数据位置

| 内容 | 位置 |
|---|---|
| 运行日志（JSONL，每行一个 JSON，已脱敏） | `logs/*.jsonl` |
| 真机事件流（**不入库**，一份 = 一场 8 局） | `data/auto_sessions/*/events/*.json` |
| B 的巡检产物（**只读，别改**） | `verify/out/inbox.log`、`verify/out/watch.status`、`verify/out/watch.log` |
| 自动状态页（**别手改**） | `notes/STATUS.md` |

## 做法

```bash
tail -f logs/*.jsonl                                   # 实时跟踪
grep '"event":"decision"' logs/*.jsonl | head           # 决策记录
grep '"event":"error"' logs/*.jsonl                     # 异常
grep -c '"event":"timeout"' logs/*.jsonl                # 超时计数
ls -l --time-style=full-iso logs/*.jsonl | tail -5       # 新鲜度：最后写入时间
cat verify/out/watch.status                              # B 的守护心跳（alive_at / files_seen）
tail -20 verify/out/inbox.log                            # B 的待办队列与数据警报
```

## 硬规则

1. **输出里不得出现令牌或其他凭据**（日志本身已脱敏，但不许你把 env 打出来）。
2. **不许修改 `verify/**` 与 `scripts/**`**。要自己的巡检产物，写 `agent/out/`。
3. **巡检只读，且对平台零请求。** 不要为了「确认在线」去主动打平台接口抢 A 的令牌配额。
4. **别 kill 别人进程**：`pgrep` 匹配前先核对 pid 与启动时间；`pkill -f` 会命中自己的 bash wrapper（A 自伤过一次）。
5. **计数要带时间窗**，否则数据增长会被误读成行为退化。附文件数或 seq 范围作为快照指纹。
6. **重启会让「从零开始」的计数器变成静默偏置**，看到计数器归零先问一句「是不是重启了」。

## 验收

- 输出含：三类事件计数、时间范围、日志新鲜度（距最后写入的秒数）、快照指纹。
- 无令牌泄漏；`git status` 显示 `verify/**`、`scripts/**` 无改动。
