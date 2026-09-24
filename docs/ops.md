# 运维手册

比赛期间程序必须**持续在线**——平台判据是「最近 90 秒内有已认证请求」。进程不在线，
或长时间没有已认证请求，都会被判离线并让位。本手册覆盖启动、停止、巡检与故障处理。

---

## 一、令牌准备

令牌**只经环境变量传入**，不入仓、不入命令行、不入日志。

```bash
mkdir -p ~/.majiang && chmod 700 ~/.majiang
cat > ~/.majiang/env <<'EOF'
MAJIANG_TOKEN_MAIN=<参赛令牌>
MAJIANG_SERVER=https://<服务器地址>:<端口>
EOF
chmod 600 ~/.majiang/env          # 必须仅本人可读
```

`MAJIANG_TOKEN_` 前缀下的**每个**环境变量都会被当作一个独立身份（各自独立限速），
一个进程内并发运行。

> 用参赛 scoped 令牌，不要用全局令牌——全局令牌会得到 `TOKEN_NOT_SCOPED`。

---

## 二、启动

### 方式 A：直接前台运行（调试用）

```bash
set -a && source ~/.majiang/env && set +a
uv run python -m majiang --env-prefix MAJIANG_TOKEN_
```

### 方式 B：守护脚本（推荐，无需 root）

```bash
set -a && source ~/.majiang/env && set +a
nohup ./scripts/supervise.sh > /dev/null 2>&1 &
```

崩了 3 秒后自动拉起；**正常运行收工（赛事终态、退出码 0）时不会重启**。

### 方式 C：systemd 用户级服务（最稳，注销后仍在跑）

```bash
mkdir -p ~/.config/systemd/user
cp scripts/majiang-ai.service ~/.config/systemd/user/
# 按实际路径改 WorkingDirectory 与 EnvironmentFile
systemctl --user daemon-reload
systemctl --user enable --now majiang-ai
loginctl enable-linger "$USER"     # 关键：注销后继续运行
```

---

## 三、停止

```bash
# 方式 B
pkill -TERM -f supervise.sh

# 方式 C
systemctl --user stop majiang-ai
```

程序已注册 `SIGTERM`/`SIGINT`：收到信号后进入优雅收尾，不会截断正在提交的动作。
`TimeoutStopSec=20` 是留给它的时间。

---

## 四、巡检（日志位置）

日志按房间分文件：`logs/<tournament_id>.jsonl`，每行一个 JSON。守护脚本另有
`logs/supervisor.log`。

```bash
tail -f logs/*.jsonl                              # 实时跟踪
grep '"event":"error' logs/*.jsonl                # 所有错误
grep '"event":"decision.made"' logs/*.jsonl | tail -20   # 最近的决策轨迹
grep '"event":"action.submitted"' logs/*.jsonl | wc -l   # 已提交动作数
grep '"event":"tournament' logs/*.jsonl           # 阶段流转
grep '"event":"game.settled"' logs/*.jsonl | wc -l # 已结算局数
tail -20 logs/supervisor.log                      # 守护视角：有没有反复重启
```

常用事件名：`runtime.*` / `tournament.*` / `game.*` / `decision.*` / `action.*` / `error.*`。

**判活的最快办法**：看 `logs/*.jsonl` 最后一行的时间戳是否在 90 秒内。

---

## 五、常见故障处理

错误一律按平台返回的 `code` 判型，不看 HTTP 状态码。

### 进程反复重启

先看 `logs/supervisor.log`，再到 `logs/*.jsonl` 里找退出前的最后一条 `error.*`。

| 日志里的 code | 含义 | 处理 |
| --- | --- | --- |
| `TOURNAMENT_CLOSED` | 赛事已关闭 | **正常**。守护不会重启（退出码 0）。确认赛程即可 |
| `TOKEN_NOT_SCOPED` | 用了全局令牌 | 换成参赛 scoped 令牌 |
| `UNAUTHORIZED` / `FORBIDDEN` | 令牌无效或无权限 | 重新在门户领取令牌 |
| `PORTAL_BINDING_REQUIRED` | 未绑定 AI 身份 | 先在门户完成绑定 |
| `FEATURE_DISABLED` | 该功能被平台关闭 | 确认赛事入口是否开放 |
| `NOT_REGISTERED` | 未报名到位 | 开赛前在确认期内完成到位 |
| `NOT_QUALIFIED` | 上轮未晋级 | 已淘汰，属正常终止 |
| `TOURNAMENT_NOT_FOUND` / `GAME_NOT_FOUND` | id 失效 | 通常是跨轮重开导致，重跑即可 |

### 被限速（`RATE_LIMITED` 持续出现）

自动退避会处理，若持续刷屏说明速率设高了。降 `--rate`（默认 14，平台上限 16/s），
或减少同时运行的身份数量。**切勿把 `--rate` 调到 16 或以上**——突发容量也会影响判定。

### `INVALID_ACTION`（竞态）

动作提交时局面已变。程序会自动重建快照并重新决策，**无需人工干预**；偶发属正常。

### 被判离线

按顺序查：① 进程还在吗（`pgrep -f "majiang"`）；② 网络能通平台吗；③
`logs/*.jsonl` 最后一条时间戳距今多久。90 秒内必须有已认证请求。

### 阶段确认漏掉

出席确认**不跨阶段继承**，漏一次即让位。每个阶段开赛前都要在确认期内确认到位。
用 `grep '"event":"tournament' logs/*.jsonl` 确认每轮的 `tournament.ready` 都出现过。

---

## 六、赛前 Checklist

- [ ] `~/.majiang/env` 权限 600，内容是**参赛**令牌
- [ ] 守护方式已启用，且 `loginctl enable-linger` 已开（方式 C）
- [ ] 干净环境验收通过：`git clone` → `uv sync` → 启动 → 到位
- [ ] 平台指南版本自检无告警
- [ ] 日志目录可写、磁盘充足
- [ ] 赛前完成**出席确认**（每轮都要）
- [ ] 开赛后 5 分钟内检查 `game.settled` 在增长（证明真的在打）

---

## 七、长稳测试（大赛前跑一次）

```bash
# 连续运行 2 小时，观察内存/连接数/轮询频率
set -a && source ~/.majiang/env && set +a
uv run python -m majiang --env-prefix MAJIANG_TOKEN_ --duration 7200
```

观察要点：内存无持续增长（无泄漏）；`game.settled` 数稳定增长；无 `RATE_LIMITED` 堆积。

---

## 八、测试房数据采集（仅测试房）

测试房跑完一轮后再次到位可开启下一轮：

```bash
uv run python -m majiang --env-prefix MAJIANG_TOKEN_ --reopen-test-room
```

**正式赛事绝不要打开此开关。**
