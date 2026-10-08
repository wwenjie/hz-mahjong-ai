# TOOLS.md — 技能与入口清单

每个技能对应 `agent/skills/<name>/SKILL.md`。要用某个技能，先读该文件，按里面的命令照抄执行。

| 技能 | 目录 | 用途 | 何时触发 |
|---|---|---|---|
| `mahjong-openspec-change` | `agent/skills/mahjong-openspec-change/` | 走 OpenSpec 流程建提案 / 写规格 / 拆任务 / 校验 | 新功能、行为改动、修复 |
| `mahjong-test-and-selfplay` | `agent/skills/mahjong-test-and-selfplay/` | 测试、自对弈、A/B、校准、强度测量 | 验证策略强弱、跑回归 |
| `mahjong-verify-metrics` | `agent/skills/mahjong-verify-metrics/` | 从原始事件流独立复算指标 | 复核别人的数字、查数据污染 |
| `mahjong-log-patrol` | `agent/skills/mahjong-log-patrol/` | 按事件类型检索 JSONL 日志与事件流 | 排查异常、确认在线判据 |
| `mahjong-train-model` | `agent/skills/mahjong-train-model/` | 数据生成、训练、纯 Python 导出、与基线对拍 | 训练新模型 / RL-NN 研究 |
| `mahjong-ops-watch` | `agent/skills/mahjong-ops-watch/` | 只读值守与告警 | 进程异常、健康检查 |
| `mahjong-delegate-codemaker` | `agent/skills/mahjong-delegate-codemaker/` | 把重活委派给 CodeMaker CLI（非交互） | 大范围改动、要第二意见、长上下文通读 |

## 部署入口（`agent/deploy/`）

| 入口 | 用途 | 验证方式 |
|---|---|---|
| `probe_link.sh` | 探测 CodeMaker Hub 可达地址形态，写/更新 `LINK.md` | 输出至少一个成功地址，或明确记录全部失败 |
| `LINK.md` | 链路结论：地址形态、鉴权形态、探测时间 | 与生效配置一致 |
| `configure.sh` | 生成/更新 `~/.openclaw/openclaw.json`（地址、端口、模型档位） | `openclaw config validate` 报告有效 |
| `install-gateway.sh` | 安装网关 + 装 systemd 用户单元 + 起服务 | `systemctl --user status` 为 active |
| `selfcheck.sh` | 链路自检：主链路可用 / 用回退 / 均不可用（有界重试） | 三种结果之一，且打印被探测地址 |
| `smoke.sh` | 冒烟：配置 + 链路 + 服务 + 渠道白名单 + 只监听回环 + 控制台可达 | 退出码 0 |
| `relink.sh` | 单一命令完成「改配置 → 校验 → 重装服务 → 冒烟」 | 改端口后四步全过，无需改源码 |
| `install-hub-wsl.sh` | 在 WSL 内解包并启动 Linux 版 CodeMaker Hub | 起不来时打印缺失依赖与下一步 |
| `start-hub-wsl.sh` | 启动 WSL 版 Hub 并自动发现代理端口 | 打印新出现的监听端口与其 `/v1/models` 状态 |
| `fetch-syslibs.sh` | 免 root 补齐 AppImage 的图形栈依赖 | `ldd` 复查「全部满足」 |
| `configure-mcp.sh` | 给 KM-MCP / websearch 补 `X-Access-Token` | `probe` 列出工具；真调需用 agent 验证 |
| `cm.sh` | **把重活委派给 CodeMaker CLI**（非交互 `codemaker run`） | `--models` / `--usage` 免 token 可用 |
| `redact-secrets.py` | 把文件里的 JWT 形状字符串就地替换成 `<REDACTED-JWT>` | `--dry-run` 先报数，再真跑并复查 |
| `refresh-mcp-auth.sh` | 三件事：token 临期才续签、MCP 头一致性检查、**模型链路探活** | 日志 `agent/out/mcp-auth-refresh.log`；失败留 `agent/out/MCP-AUTH-FAILED` 或 `MODEL-LINK-DOWN`（含恢复步骤）并非零退出 |
| `install-mcp-refresh-timer.sh` | 装 systemd 用户定时器（每 6 小时检查），`--run-now` 立即验证，`--uninstall` 卸载 | `systemctl --user list-timers openclaw-mcp-refresh.timer` |

systemd 单元**不手写**，由 `openclaw gateway install --force --port <port>` 生成到
`~/.config/systemd/user/openclaw-gateway.service`。注意：服务已存在时 `openclaw daemon install`
只打印「already enabled」并返回 0、**不重写单元**，所以改端口必须走 `gateway install --force`。

## 值守入口

| 入口 | 用途 | 验证方式 |
|---|---|---|
| `agent/patrol/patrol.py --once` | 单轮巡检，前台给结论 | 输出各项判据数值 |
| `agent/patrol/patrol.py` | 常驻循环，产物落 `agent/out/` | `agent/out/patrol.status` 心跳连续 |
| `agent/out/alerts.log` | 告警队列（去重 + 确认后静默） | 同根因只出现一次 |

## 规则

- 每个入口对应的文件必须存在；**不存在就报告并停下，不要凭印象执行命令**。
- 所有入口都不许越界（见 `AGENTS.md` §3）与不许打平台（§4）。
