# 杭州麻将 · 网页对局（你 vs 3×决策器）

一个**本地网页版**杭州麻将：你在浏览器里和 3 个 **决策器**（默认 `v7`，当前线上采集档位）打一整场。
复用仓库现有规则引擎与决策器，不重复实现规则，也不改 `src/` 任何代码。

## 快速开始

```bash
# 一键启动（首次自动 npm install + 构建前端）
bash webapp/start.sh

# 换端口 / 局数 / 随机种子 / 座位
bash webapp/start.sh --port 9000 --rounds 4 --seed 42 --human-seat 0
```

启动后浏览器打开 **http://127.0.0.1:8848/**（WSL 里从 Windows 访问用 `http://<WSL-IP>:8848/`，
`hostname -I` 查 IP；服务默认监听 `0.0.0.0`）。

## 架构

```
webapp/
  server.py      HTTP 服务（Python 标准库 http.server + SSE，零外部依赖）
  session.py     对局会话：3×决策器 + 1 名人类；HumanDecider 阻塞等网页输入
  start.sh       一键启动（装依赖 + 构建 + 起服务）
  src/           Vue 3 前端源码
    App.vue      牌桌主组件（四家视角、手牌、动作按钮、日志）
    Tile.vue     单张牌面
    tiles.js     牌码 → 显示、排序
  dist/          构建产物（由 server.py 托管；已 gitignore）
```

### 关键设计

| 关注点 | 做法 |
| --- | --- |
| 引擎复用 | 直接调用 `src/majiang/sim/round.py::run_round` 与 `strategy/versions.py::build("v7")`（可用 `--decider` 换档） |
| 人类即 Decider | `HumanDecider.choose` 在游戏线程阻塞，直到网页 `POST /api/action` 提交合法动作——引擎主循环无需改动 |
| 信息边界 | 发给网页的状态**只含人类手牌 + 公共信息**（各座弃牌/副露/手牌数/牌墙），**绝不含对手手牌**，与比赛决策器口径一致 |
| 合法性 | 网页只能提交后端 `legal_actions` 里给出的动作；`submit` 会二次校验，非法即拒 |
| 实时推送 | 单线程游戏 + 事件总线；`GET /api/stream`（SSE）推送 `state` / `action_required` / `round_end` / `match_end` |

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 前端页面 |
| GET | `/api/state` | 当前状态快照 |
| GET | `/api/stream` | SSE 事件流 |
| POST | `/api/action` | 提交动作 `{kind, tile, tiles, gang_kind}` |
| POST | `/api/new` | 开新一场 `{rounds?, seed?, human_seat?, mode?}` |

## 前端开发（热更新）

```bash
cd webapp
npm run dev     # Vite 5173，/api 自动代理到 127.0.0.1:8848
# 另开一个终端先跑后端：
../.venv/bin/python server.py
```

## 说明

* 规则与番型完全由仓库引擎决定（与参赛程序同一套 `src/majiang/rules/**`）。
* `session.py` 末尾的 `__main__` 是自检：脚本化「人类」自动跑完一场，验证线程与阻塞逻辑。
* 网页端不依赖 `numpy`/`torch` 等——后端只 import 标准库与 `src/`。
