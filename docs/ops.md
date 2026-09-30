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
- [ ] **这台机器上只跑参赛程序**（比赛期间不要跑实验队列 / 训练 / 大批分析）。
      依据（2026-09-28 实测）：CPU 竞争会让轮询循环被饿到，我们在**过期快照**上决策 →
      平台以 `INVALID_ACTION` 拒掉那些提交（v2 时代 2.2 次/房 → v3 时代 **5.4 次/房**，
      全在出牌阶段、且各时段均匀分布）。**实测不丢出牌**（72.7 → 71.8 手/房），
      代价是**浪费速率额度**（平台限速 16/s，我们本就贴着用）。
      另一处旁证：同一批 job 在 6 并发下每场墙钟 ≈115 分钟、3 并发下 ≈48 秒/场（单 job 口径）；
      系统吞吐仍是并发更高，但**真机窗口余量优先**。
- [ ] **启动档位已确认为当前冠军版本**（`--decider v3`，见第九节；不写则跑默认档，
      而默认档与冠军版本**可能不是同一个**——v2/v3 这种已冻结的版本必须显式指名）
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

---

## 九、冠军档位与重部署（先读这一节的「为什么」）

### 为什么这件事需要单独一节

1. **默认档 ≠ 冠军档**。`DECIDERS` 里的 `heuristic` 是**新建决策器时的默认配置**，
   而冠军是**冻结在 `src/majiang/strategy/versions.py` 里的具名版本**（v1/v2/v3…）。
   切冠军**不修改默认值**——一改默认，已冻结版本的「逐位可复现」就没了
   （v1 的 `tiebreak=blocks` 当年就是这么丢的，只能拿近似档位当替身补救）。
2. **采集进程的档位是在启动时固定的**（`MAJIANG_COLLECT_DECIDERS`），
   所以换冠军**必须重启采集进程**，而重启需要令牌环境变量 → 这一步只能在
   有令牌的 shell 里做，不能让程序自己去读。
3. **重启会把真机数据切成两个时代**。任何跨时代的分析都必须显式过滤
   （`analyze_wait_ceiling.py --era v3` 之类；口径见 `notes/agent-a.md`），
   否则算出来的是两代策略的平均。

### 切换步骤

```bash
# 1. 先停旧守护（它会转发 SIGTERM 给 auto_session，打完当前会话再退）
pkill -f collector_supervisor.sh

# 2. 用**当前冠军版本号**重启（令牌只从 .env 读，绝不写在命令行上）
set -a; . ./.env; set +a
MAJIANG_COLLECT_DECIDERS=v3,first-legal \
MAJIANG_COLLECT_ARM_LIMIT=first-legal=8 \
  nohup setsid tools/collector_supervisor.sh >> /tmp/autoloop.log 2>&1 < /dev/null &

# 3. 核对台账：新会话的 decider 字段应是 v3
tail -3 data/auto_sessions/sessions.jsonl
```

`first-legal` 是**对照臂**，用 `--arm-limit` 限量跑（跑到量自动停用）。
它的唯一用途是「自对弈预测的刻度校准」（历史实测：自对弈预测 −21.8pp
vs 真机交错实测 −20.5pp，误差 <1.3pp），不是候选策略。

### 新增冠军的纪律

```bash
uv run python -c "from majiang.strategy.versions import describe; print(describe())"
```

新版本必须**同时**满足三条，否则不要切：
① 在 `versions.py` 里新增快照并写清日期、依据、被替换的开关；
② 有**机制级证据**（比率型机制量或上限诊断），不能只有总得分——自对弈总得分对
中段改动的功效不足，且已知存在场地偏移（见 `notes/agent-a.md` 的 `feed-high` 检定）；
③ 全量测试与干净环境验收通过（`uv run pytest tests/` + `scripts/verify_clean_env.sh`）。

### 2026-09-30 00:40 冠军改为 **v3 + v4 交错轮换**（用户授权）

**做法**：不单选一个冠军，而是让采集器在同一时段内**交替**跑 v3 与 v4：

```bash
set -a; . ./.env; set +a
MAJIANG_COLLECT_DECIDERS=v3,v4 nohup setsid tools/collector_supervisor.sh \
  >> /tmp/autoloop.log 2>&1 < /dev/null &
```

**为什么是轮换而不是单选**：`--decider A,B` 的选择是 `available[done % len(available)]`
（`tools/auto_session.py:360`，`done` 是账本里的会话总数）⇒ **严格逐场交替**，
所以拿到的是**交错对比**，而不是 v2-vs-v3 那种被房间/对手池/时段混杂污染的时代对比
（那份分析里已注明时代混杂）。会话账本按 `decider` 字段分臂，两边数据都留着、都能单独分析。
**注意不加 `--arm-limit`**：那个开关是「给对照臂设上限、到量自动停用」，用于一次性对照臂；
v3/v4 是要长期交替的两臂。

**回退（一条命令）**：把 `MAJIANG_COLLECT_DECIDERS` 改回 `v3`（或不设，用守护默认值），
再按下面「切换步骤」停旧起新即可。`versions.py` 里 **v3 的快照原样保留**，
`ab_test --treatment v4 --baseline v3` 随时可再直接对拍。

### 2026-10-01 01:00 冠军轮换改为 **v4 + v5**（用户授权；v3 从轮换中退出）

```bash
set -a; . ./.env; set +a
MAJIANG_COLLECT_DECIDERS=v4,v5 nohup setsid tools/collector_supervisor.sh \
  >> /tmp/autoloop.log 2>&1 < /dev/null &
```

**为什么退出 v3**：v3 的真机队列已攒 **122 场**（决策级样本几十万条），做**比率型机制指标**
（到听率、弃牌结构）的基线已经绰绰有余，而胜率那类指标真机本来就分辨不了（差 2pp vs 分辨率 5pp）
⇒ **继续采 v3 没有新信息**。

**为什么保留一个参考臂（v4）而不是纯 v5**：真机上唯一能分辨的是**机制量对比**，而对比**必须同期交错**
（历史队列比会引入时代混杂）。选 v4 当参考是因为**它与 v5 只差一个开关**（`ukeire_candidates` 2 vs 3），
是最近邻、对比最干净。

**注意**：改臂数**不改变总对局量**——采集器串行、总场次受平台房间/对手供给限制。
换臂只是重新分配同一批场次（v5 占比 1/3 → 1/2）。


**⚠ 与上面「新增冠军的纪律」的偏差（必须记账）**：v4 满足 ①③，但**②只满足一半**——
它有 n=10 种子、合并名次分 +0.2732（t+4.66）、10/10 同向的自对弈 A/B（比 v3 当年的两种子更强），
但 **`shape_value` 的机制门尚未独立复算**（该仪器由 agent-c 承担，2026-09-29 17:05 派下，
换档时未交付）。也就是说**这次换档的顺序与 v3 相反**：v3 是机制先行（regret 6.7%→0 才谈 A/B），
v4 是 A/B 先行、机制待验。选择这样做是因为真机只能积累、错过就没了，而回退成本是一条命令。

**可证伪的机制预测（换档时写下，供事后核对）**：
① 弃牌结构应**向强 bot 画像靠拢**——中张占比从 ~19.6% 上升（强 bot 29.7%，逐房配对差 −11.2pp ± 0.4
是当前最大的指纹差之一）；
② 同向听内的**形质分辨率**应上升（当前 `quick_blocks` 恒饱和 ⇒ 77~93% 全并列）；
③ 若真机机制量**一条都没动**，说明 A/B 的正号另有来源，需回头重审本档
（`shape-blocks` 自己就被这样绕过一次：它的机制归因一度被记成「并列次序」，后被 `seen-tiebreak` ≈0 证伪）。

### v5 延迟护栏的定时复测（**任何人可执行**，2026-10-01 01:10 立）

v5 的成本来自候选面 +1 张（每张候选多算一次精确进张，42–157ms）。换档时实测
**p99 860ms / 预算 1800ms、最大 984ms、零超预算**（n=348），但那个 n 太小、p99 估计噪声大。
**触发条件**：v5 的出牌相决策样本 ≥ 2000 后复测一次（之后每 24h 一次）。
**动作**：若 p99 > 1300ms 或出现任何超预算事件 ⇒ 在 THREAD 报警；
**回退命令**（按 `notes/PROTOCOL.md` §5.2 由持令牌方执行）：

```bash
set -a; . ./.env; set +a
MAJIANG_COLLECT_DECIDERS=v4 nohup setsid tools/collector_supervisor.sh \
  >> /tmp/autoloop.log 2>&1 < /dev/null &
```

复测命令（只读，按 decider 签名分流，出牌相）：

```bash
uv run python - <<'EOF'
import collections, glob, json
b = collections.defaultdict(list)
for path in glob.glob("logs/*.jsonl"):
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"decision.made"' not in line or '"elapsed_ms"' not in line:
            continue
        try: row = json.loads(line)
        except Exception: continue
        if row.get("phase") != "draw": continue
        n = str(row.get("decider", "")); e = row.get("elapsed_ms"); bud = row.get("budget_ms")
        if e is None: continue
        key = "v5" if "ukeire-candidates=3" in n else ("v4" if "shape-value=True" in n else None)
        if key is None: continue
        b[key].append((e, bud))
for k in ("v4", "v5"):
    xs = sorted(b.get(k) or [])
    if not xs: print(k, "无样本"); continue
    q = lambda t: xs[min(len(xs)-1, int(len(xs)*t))][0]
    over = sum(1 for e, bud in xs if bud and e > bud)
    print(f"{k} n={len(xs)} p50 {q(.5):.1f} p99 {q(.99):.1f} max {xs[-1][0]:.1f} 超预算 {over}")
EOF
```
