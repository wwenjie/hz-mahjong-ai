结论写进纪律：**凭据一律走仓库外文件、不回显、不进命令行；一旦暴露，处置顺序是「先重置、再清残留」**，
而不是反过来。

---

## 2026-09-29 接入 kimi-k3

链路隔了两天仍健康（`health=200`，网关 active）。

新发现：CLI 登录后 **`codemaker model-list` 直接给全量目录**（此前未登录只列免费的 `deepseek-flash`），
2026-09-29 实测含 `kimi-k3`、`kimi-k2.7-code`、`kimi-k2.6-inhouse-yd`、`deepseek-v4-pro`、
`claude-opus-5`、`gemini-3.8-flash` 等 —— 以后不必再靠挨个试探猜模型 id。

已接入并真调验证：

- `kimi-k3`（**推理型**，响应含 `reasoning_content`；用 `max_tokens` 太小时会全花在推理上、看不到正文）
- `kimi-k2.7-code` 一并登记
- 经 `openclaw agent --model netease-codemaker/kimi-k3 -m "..."` 实测返回正常

**默认档仍是 `deepseek-flash`**（外层长跑便宜），kimi-k3 定位为难任务的按需档。委派侧同理：
`CODEMAKER_DELEGATE_MODEL=netease-codemaker/kimi-k3 bash agent/deploy/cm.sh "..."`。

**教训重申**：清单里有 ≠ Hub 代理认 —— 进 provider 前必须真调验一次（这次 `kimi-k3` 恰好两边都认）。

---

## 2026-10-02 两个 MCP 静默失效 → 已治根

### 现象

用户问「能不能接 CodeMaker 的各种 MCP，比如 web-search」，去查才发现：**09-27 接的 KM-MCP 与 websearch
已经失效了好几天，而且没有任何告警**（真调才暴露）：

- KM-MCP：`Token 验证失败 user=wuwenjie01`
- websearch：`X-Access-Token 无效或身份校验失败`

根因：`auth.json` 的 `access_token` **约 24 小时过期**，而 MCP 头是写进配置的快照，不会自己更新。
上一次续签是 09-28 23:40，到 10-02 已过期 4 天。

### 排障中发现的第二个坑：静默写失败

`configure-mcp.sh` 里两次 `openclaw mcp set`，**第一次那条 websearch 没写进去**（KM-MCP 更新了、
websearch 还是旧 token），而脚本返回 0、我看 `head -3` 又截掉了输出 —— 于是"重配过了"但没生效。

已给脚本加**写回校验**：写完回读配置比对 token，不一致就非零退出并点名哪个服务。
这类"看起来成功了其实没生效"的失败，必须靠回读断言。

### 诊断方法（可复用）

`km-mcp` 裸调失败但走 OpenClaw 成功，`websearch` 反之 —— 说明两者要求不同：KM-MCP 需要额外的
`X-Auth-User` 头；websearch 只需要 `X-Access-Token`。用 `/tmp/mcp_probe.py` 的做法：
**按 JSON-RPC 走完 `initialize` → `tools/call`**，并注意响应是 **SSE**（`data:` 行），
按普通 JSON 解析会误判成失败。

### 治根：自动续期定时器

- `agent/deploy/refresh-mcp-auth.sh`：读 `auth.json` 的 `exp`，**剩余有效期 < 6h 才**跑一次最小 CLI 调用触发
  续签，然后重写 MCP 头（含写回校验）。有效期充足就直接跳过，不白花调用。
- `agent/deploy/openclaw-mcp-refresh.{service,timer}`：systemd 用户定时器，每 6 小时一次，`Persistent=true`。
  服务里显式设 `PATH`（systemd 用户环境很干净，`uv`/`openclaw`/`codemaker` 都不在默认 PATH）。
- 日志 `agent/out/mcp-auth-refresh.log`；systemd 侧不再 append 同一文件（否则日志写两遍）。

### 平台还有其他可接的 MCP（KM 文档 230243 / 274855 / 272380）

| 服务 | 端点 | 额外要求 |
|---|---|---|
| WebSearch | `https://mcp.netease.com/servers/websearch-mcp-server/mcp/` | `X-Access-Token` |
| KM 知识库 | `https://mcp.netease.com/servers/km-mcp/mcp` | `X-Access-Token` + `X-Auth-User` |
| 易协作 Redmine | `https://mcp.netease.com/servers/redmine-mcp-server/mcp` | `X-Access-Token` + `redmine-host`（如 `dap-v4.pm.netease.com`） |
| Autobot | `http://s-autobot.eros.nie.netease.com/mcp` | 内网可达即可 |
| GitLab | CodeMaker 内置，非 mcp.netease.com 端点 | 需自备 `GITLAB-ACCESS-TOKEN` 与 `GITLAB-API-URL` |

**官方警告（直接抄）**：大量启用 MCP 会持续占用上下文资源，降低模型响应速度、频繁触发上下文压缩，
并产生较高服务成本 —— **建议按需开启/关闭**。所以不一次全接，要哪个接哪个。

---

## 2026-10-06 重启后模型链路断（用户先发现）

### 现象与根因

电脑重启、WSL 一并重启后，用户报「小龙虾连不上大模型」。分层排查：

| 层 | 状态 |
|---|---|
| 网关服务 | ✅ active + enabled + 监听 18789（systemd 自启正常） |
| portproxy 规则 | ✅ 仍在（`172.19.80.1:15721 → 127.0.0.1:15721`） |
| 防火墙规则 | ✅ 仍启用 |
| **Windows 侧 CodeMaker Hub** | ❌ **没在运行**（重启后不会自启）——这就是根因 |
| WSL 网关 → Hub | ❌ `health=000` |

用户手动启动 Hub 后：`health=200`、真调 agent 回合成功、KM-MCP 真调成功。

### 一个重要收获：自动续期经受住了 4 天停机

排查时发现 token 剩余 21.8 小时 —— 说明 10-02 到 10-06 这 4 天里**定时器一直在正常续签**。
也就是说上次那套 3 小时检查 / 12 小时阈值的机制是有效的。

### 已补上：不许再靠人发现

问题是「Windows Hub 没起」这件事**完全无声**：WSL 侧一切指标正常，只有真调模型才会撞墙。
已把它纳入那个 3 小时定时器（`refresh-mcp-auth.sh`）：

- **模型链路探活**：探的地址从 `openclaw.json` 的 provider `baseUrl` 读（不硬编码）
- 不可达时写 `agent/out/MODEL-LINK-DOWN`，内容含**分步恢复指引**（先启 Hub、
  再查 portproxy/防火墙/probe_link.sh），并把退出码设为 1（systemd 显示 failed，不再是隐形故障）
- 可达时自动删除标记

**故障注入验证**（用 `/tmp/bad-config.json` 把 baseUrl 指向死端口，不碰真配置）：
坏配置 → 退出码 1 + 写标记 ✅；真配置 → 退出码 0 + 删标记 + 日志「模型链路可达 HTTP 200」✅

顺带给脚本加了 `OPENCLAW_CONFIG` 覆盖，就是为了能这样安全地测失败分支。

### Hub 的可执行路径（供设自启用）

```
D:\app\CodeMaker Hub\codemaker-hub-gui.exe
```

（Hub 自身没有开机自启，这是本次故障的直接原因。）