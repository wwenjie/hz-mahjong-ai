# agent-c（小龙虾）记录

只追加，不改历史行。职责与红线见 `agent/AGENTS.md`；边界见 `notes/OWNERSHIP.md` 的 C 段。

---

## 2026-09-28 10:55 — 指南 v35 对照评审的交付物与 P1.1/P1.2/P2 结论

### 本轮做了什么

平台《杭州麻将玩法规则》+ 指南版本与 9 月变更表逐条对照仓库实现；复现 1 个规则 bug（P0，已由 A 修）；
完成 A 委派的 P1.1 / P1.2 / P2 三项查证。

### P0（已由 A 于 commit `4574e57` 修复）

抓打圈内暗杠被错误禁止。`action.py:_turn_actions` 与 `concealed_gang_options` 各自挡了一道
`is_restricted`，而规则只放行「暗杠与自摸胡」。我只读复现（4×9t + `catch_play=True` →
修前合法动作仅 `['discard']`）。

### P1.1 → **无需运行期改动**，但需补 1 条回归测试

查证结论（走**代码路径 + 只读探针**，非平台）：

1. **事件层不会跳出通用路径。** 真实事件流的类型集是
   `{tile_drawn, tile_discarded, timeout, chi, peng, gang, pass, hu, round_ended, game_ended}`，
   **没有专门的「杠后补牌」事件**；杠后补牌由随后的普通 `tile_drawn` 承担。
   `engine.py:_should_fetch_for_events` 见到 `seat==本人` 或未知类型即返回 True（宁愿多拉一次），
   因此补牌事件必被消费；`KNOWN_EVENT_TYPES` 未含 `pass`/`hu`/`game_ended`，会走「未知类型」分支
   ——**是保守正确，但也意味着以后新增事件类型一定触发重拉，不会静默漏掉**。
2. **补牌后走 `phase=draw` + `drawn_tile`**，与普通摸牌同构。`HeuristicDecider.choose` 分发到
   `_choose_turn` → `_choose_win_or_piao`，**会主动提交 hu**，不依赖服务端代胡。
3. **只读探针**（`agent/verify/probe_gang_vs_hu.py`）：构造「4 面子 + 1 财神 + 补牌」的爆头态，
   合法动作 `['discard','hu']`，决策 `hu`，理由「胡：4 番（爆头）」——补牌可胡时**确实主动胡**。

**遗留（建议 A 补）**：目前没有一条测试专门覆盖「杠后补牌即听任意」的决策路径。现有
`test_action.py` 覆盖的是动作合法性，不是补牌时机的决策器行为。建议加：
输入 = 14 张、`phase=draw`、`drawn=补牌`、`god.baotou=True`，断言决策器不返回 `discard`。

### P1.1 附带发现（**新问题，建议独立处理**）：能胡时永远不考虑杠

只读探针 `agent/verify/probe_gang_vs_hu.py` 用例 2：手牌 `1111w22w33w44w55w66w`（同时满足
豪华七对×1 与 1w 暗杠），合法动作 `['discard','gang','hu']`，决策 **`hu`**，
理由「胡：4 番（常规），无续飘机会」。

原因在控制流：`_choose_turn` 先判 `if any(kind == HU)` → 直接进 `_choose_win_or_piao` 并**返回**，
`_best_gang` 在该分支内根本没机会运行；而 `_choose_win_or_piao` 只比较「胡」与「财飘」，
候选里没有杠。

**影响**：D11 的「杠优于飘」只被写成原则，实现上没落地。若某局同时能胡 + 能杠，策略会放弃
「杠（链+1）× 补牌（多一次自摸）」的机会。收益量级 = 「是否多一次自摸 + 链是否 +1」，需按
番连乘与重尾目标评估；**不要据此直接判定「该改」**，应先量化「能胡且能杠」局面在真机的频率
（我可以在 `agent/verify/` 里出一版统计，不动 A 的文件）。

### P1.2 → 结论：Q1/Q2 是**严格安全**的；Q3 有已知近似但**当前无下游消费者**

A 让我审计「从事件流反推白板」的工具。逐项：

| 工具 | 统计的是什么 | 是否被 09-02「不可复算」口径坑到 |
|---|---|---|
| `tools/analyze_god_usage.py` | **第 4 摸时手牌里持留的白板张数**（状态量，非官方 `god_count`） | **否**。它要的就是手牌态；且与「官方 god_count」是不同定义，输出里应写明不可与排名键混用 |
| `tools/measure_strength.py` | 不报白板数 | **否** |
| `verify/win_rate_table.py`、`verify/clone_dataset.py`、`verify/clone_decider.py` | 手留白板数（特征 `[110]`） | **否**（同为状态量） |
| `tools/ab_test.py:75` | `stat.god_count` | **是自对弈内算的**（`sim/round.py` 直接累计），非事件流反推 → **否** |
| `tools/analyze_god_usage.py` 输出 | 与官方排序键并称 | **文档风险**：输出未注明「≠ 官方 god_count」，易被误当排名键复算 |

**净结论**：仓库目前**没有**任何工具尝试从事件流复算官方 `god_count`——这也是为什么这条口径
没炸。风险只在「有人以后想复算它」。自对弈侧 `sim/round.py` 的口径（`deal()` 记起手 13 张的白板、
`_draw()` 对**包括杠上摸在内的每一次摸牌**在摸到白板时 `god_count += 1`）与指南 09-02 修订
（配牌含庄家第 14 张 + 每次墙摸含杠上摸 + 轮空积 0）**逐条一致**——但 `deal()` 只发 13 张，
庄家第 14 张走 `_draw()`，**同样会被计入**，故等价。

**建议**（都不需要立刻做）：① 在 `notes/PROTOCOL.md` 或 `docs/` 留一条显式约束
「官方 `god_count` 不可经事件流复算，只能直读 ranking」；② 在 `analyze_god_usage.py` 输出里
加一行「此处的白板 = 手留白，不等于官方 god_count」。

### P1.2 附带发现（**新问题，离线口径不一致**）：离线重建把「非爆头打财神」当成财飘

线上 `rules/god.py:after_god_discard_without_piao` 写得很清楚：非爆头态打出财神 → **链断**
（`chain_count=0`）但抓打圈照常触发；模拟器 `sim/round.py:classify_discard` 也区分
`PIAO`（摸牌前后皆爆头）与 `GOD_BREAK`（其余），只有 `PIAO` 才 `chain_count += 1`。

但**离线重建器 `sim/replay.py:258-263` 对任何 `tile == GOD` 的弃牌一律**
`chain_count += 1; piao_count += 1`——把「非爆头打财神」也记成了飘。`replay.py:35` 自己
标注了这个近似，理由是「这两个量不参与任何对手特征」。

**现状核实**：`strategy/features.py` 与 `opponent_features.py` **确实没有**消费 `chain_count`/`piao_count`
（只有 `catch_play`、`gods_in_hand`、`gods_seen`），所以**当前下游无害**。
**风险**：一旦有人用真实事件流去验证番型链、或给「飘」建特征/标签，这个近似会**高估飘次数**。
**建议**：升级那个「已知近似」注释，写明「若消费 `chain_count`/`piao_count` 必须先修此处」，
或直接在 `replay.apply_event` 里按「打财神前后是否皆爆头」判定（与 `round.classify_discard` 同源）。
**我不改** `src/majiang/sim/replay.py`（A 的地盘）。

### P2 → 一处已确认「已覆盖但无测试」，一处属文档

1. **「圈内吃碰后再打财神 = 财飘链 +1」**：语义落在 `GodState.after_piao`（链+1、飘+1、重启抓打圈）
   与 `after_god_discard_without_piao`（链断）。但这两个转换在整个 `src/`、`tools/`、`verify/` 里
   **零调用**——只有 `tests/test_table.py` 与 `tests/test_sim.py` 直接调用。运行期（`engine.py`）
   **消费的是服务端下发的 `god.chain_count`**，本地只累计 `piao_count`（`PiaoTracker`）。
   故「是否覆盖」取决于服务端是否在圈内吃碰后再打财神时如实 +1；客户端不需要改，**但需要一条测试固化**。
2. **§1.3 残留冲突**：规则文档 §1.2 说 4 白板爆头叠加（09-07），§1.3 表格仍写「正好 4 白板除外」。
   仓库按 §1.2 + 官方端点实测（`fan.py:BAOTOU_STACKS_WITH_FOUR_GODS=True`）。建议把
   「文档残留冲突 + 端点实测」一起写进回归说明，防后人照 §1.3 改回去。

### 产物

- `agent/verify/probe_gang_vs_hu.py`（只读探针，可复跑；不碰平台、不改任何源文件）

### 关联文件

`src/majiang/rules/action.py`、`src/majiang/rules/god.py`、`src/majiang/sim/replay.py`、
`src/majiang/sim/round.py`、`src/majiang/runtime/engine.py`、`src/majiang/strategy/policy.py`、
`tools/analyze_god_usage.py`、`tools/ab_test.py`

---

## 2026-09-28 11:30 — 补 A 委派的两个回归用例（P1.1 收尾）

A 于 11:05 留下两条待办：「杠后补牌成胡 → 决策返回 HU」与「杠后补牌成听任意 → 是否构成财飘链入口」。
`tests/**` 归 A/B，**不编辑他人测试文件**；用例做成只读可执行探针，放在 C 的地盘
`agent/verify/probe_gang_vs_hu.py`（函数以 `check_` 命名，pytest 默认不收集，避免误纳入别人的测试套件）。

### 结果（`uv run python agent/verify/probe_gang_vs_hu.py`）

| 用例 | 构造 | 合法动作 | 决策 | 判读 |
|---|---|---|---|---|
| 1 | 4 面子 + 将，补牌成**普通胡** | `['discard','hu']` | `hu`（2 番） | ✓ 补牌成胡会主动提交，不等服务端代胡 |
| 2 | 4 面子 + 1 财神，补牌成**爆头** | `['discard','hu']` | `hu`（4 番） | ✓ 走通用路径 |
| 3 | `1111w22w33w44w55w66w`（**能胡+能暗杠**） | `['discard','gang','hu']` | `hu`（4 番） | ✗ 杠未被考虑（新问题 1） |
| 4 | 6 对 + 1 财神，补牌成爆头 | `['discard','hu']` | `hu`（4 番） | ✓ 同用例 2 |
| 5 | 非爆头持财神 | `['discard','hu']` | `hu` | ✓ 未误走财飘 |
| 6 | 4 面子 + **2 财神**，摸到第 2 张财神 | `['discard','hu']` | **`discard:白`（弃胡飘）** | ✓ **财飘链入口可达，且被正确触发** |

**用例 6 是关键**：番 2→4，`survival=0.986 > threshold=0.571`，`last_reason="弃胡飘"`。
即「杠后补牌成听任意」**确实构成财飘链入口**——只要手上还有第二张财神可打、且打完仍是爆头。
这与 P0 同源：**09-12/09-13 两条修订都在增强杠的收益路径**，而这条入口目前**没有测试固定**
（现有 `tests/` 未覆盖补牌时机的决策行为）。

### 结论

- A 的推断（「结构与普通摸牌同构、不会漏判」）**正确**；我补的是**上游**一环：
  补牌 → 决策 → **主动 hu** 这一环也已实测确认，不依赖服务端兜底。
- 唯一仍开放的是**新问题 1**（能胡时不考虑杠），需要先测频率再决定是否动。

---

## 2026-09-28 11:40 — 改名说明：本文件由 `notes/agent-openclaw.md` 更名为 `notes/agent-c.md`

用户于 2026-09-28 11:02 指定代号为 **agent-c**。原先写在 `notes/agent-openclaw.md` 的
历史记录（2026-09-27 的部署、研究线、训练栈、链路、MCP/委派五节）**原样并入下方**，
不改任何历史行；旧文件删除，避免同一份记录分裂在两处。
以后本线一切记录与 THREAD 标题统一用 **C** / `notes/agent-c.md`。

---

# 历史记录（原 `notes/agent-openclaw.md`，2026-09-27）

## 2026-09-27 部署落地（23/33 任务完成）

### 环境实测（都是本机跑出来的，不是文档抄的）

| 项 | 实测 |
|---|---|
| 系统 | WSL2，NAT 模式（`eth0 172.19.84.149/20`，网关 `172.19.80.1`，resolv `10.255.255.254`），`systemd` 已启用 |
| 运行时 | node v24.21.0 / npm 11.19.0 |
| 网关 | OpenClaw 2026.9.6，`npm i -g` 装到 `~/.nvm/versions/node/v24.21.0` |
| 服务 | systemd 用户单元 `~/.config/systemd/user/openclaw-gateway.service`（enabled），`ExecStart` 带 `gateway --port 18789` |
| 监听 | 仅 `127.0.0.1:18789` + `[::1]:18789`，探测显示 loopback-only |
| 模型 | provider `netease-codemaker` 下 6 个模型；默认 `claude-sonnet-4-6`，降级 `claude-haiku-4-5-20251001` |
| 渠道 | `openclaw channels list` = 未配置任何渠道（渠道靠配置凭据启用，不配即禁用） |
| GPU | RTX 5060 Ti 8GB；torch 未装 |

### 三个只有实测才知道的坑

1. **`openclaw daemon install` 对已存在的服务静默无效。** 服务已在时它只打印
   「Gateway service already enabled. Reinstall with: openclaw gateway install --force」，**返回 0 且不重写单元**。
   后果：改了 `gateway.port` 却不生效（单元 argv 仍是旧端口）。正确入口是
   `openclaw gateway install --force --port <port>`。已写进 `relink.sh` 与 `TOOLS.md`。
2. **`openclaw config patch` 有 `--dry-run`**，配置脚本先用它预检再落盘，避免写坏配置。
3. **不要把会输出很多行的脚本管到 `head`。** 调试期我用 `relink.sh | head -12`，`head` 提前关管道触发
   SIGPIPE 把脚本打死，误判成「脚本有 bug」。脚本本身没问题。

### 两个自查出来的真 bug（在巡检实现里）

1. **`--audit` 自匹配**：禁用词表（`urllib`/`socket`/…）本身就写在源码里，文本扫描必然命中自己。
   改成 AST 检查：查 `ast.Import`/`ImportFrom` 的网络模块、`os.system`/`subprocess.Popen`、
   以及 `subprocess.run` 的首参是否只允许 `ps`。
2. **告警去重失效**：inbox 去重键用了内建 `hash()`，**每进程随机化**（PYTHONHASHSEED），
   于是每轮巡检都把同样的 12 条重报一遍。改用 `sha256(行内容)[:16]`，修复后连跑 3 轮新增 0 行。

另修正一处误报：熔断判定原先扫「末 200 行」，把 **2026-09-24 的历史熔断**在三天后仍报成当前状态。
改为只看**最后一次「守护启动」之后**的行，并在告警里附上「最后一次守护启动 N 天前」。

### 链路状态：未通（等人在 Windows 侧启动 Hub）

`agent/deploy/probe_link.sh` 探测结果（3 主机 × 5 端口 = 15 组合）：**全部 `000`**。

- `127.0.0.1:15721` 未监听；`172.19.80.1` 与 `10.255.255.254` 上也不通
- WSL 内只有 CodeMaker 语言服务器在跑，它监听 IPC 端口（HTTP 探测返回 `426 Upgrade Required`），**不是 Hub 代理**
- 也无 `~/.local/share/codemaker/auth.json`，所以「直连内网 API + JWT 刷新」这条路目前**缺凭据来源**，不作首选

**结论**：网关本身健康（配置有效、服务 active、控制台 HTTP 200），但模型链路不可用。
冒烟脚本第 2 项失败、第 1/3/4/5/6 项全过 —— 这正是设计意图：链路不可达**不阻塞**网关启动，只给出可读结论。

下一步（等人做）：

```bash
# 1. Windows 侧启动 CodeMaker Hub，记下设置页显示的实际端口
# 2. WSL 内取可达地址
bash agent/deploy/probe_link.sh <port>
# 3. 换链路（改配置 → 校验 → 重装服务 → 冒烟，一条命令）
OPENCLAW_HUB_BASEURL=<可达地址> bash agent/deploy/relink.sh
# 4. 仍不通 → %UserProfile%\.wslconfig 加 [networkingMode=mirrored]，wsl --shutdown 后重试
```

### 顺手观察到的 A 侧信号（**未处理，不是我的地盘**）

巡检首轮就抓到两条，供决策参考：

- **目标进程 `python -m majiang` 不存在**
- **A 的守护脚本在 2026-09-24T17:47:05 已熔断退出**：连续 5 次「启动后 5s 内以退出码 1 退出」，
  日志末行是「请检查日志与令牌后手动重启（这是刻意熔断）」。**距今 3 天**。

`logs/*.jsonl` 仍在被写入（新鲜度 0–1s），说明有别的进程在产日志，但比赛进程本身没在跑。
我没有启动、重启或终止任何进程 —— 按 `AGENTS.md` §3，动平台与重启都归 A / 人。

### 未完成（10 项）

| 任务 | 阻塞原因 |
|---|---|
| 1.2 鉴权形态 | 需 Hub 可达地址 |
| 6.1–6.7 RL/NN 研究线 | 6.1 要拉 CUDA torch（数 GB），等用户确认；其余依赖 6.1 |
| 7.1 / 7.2 端到端与越界演练 | 需模型链路通才能产生 agent 回合 |

---

## 2026-09-27 研究线骨架（进度 26/33）

### 决策：不另起特征集

`research/FEATURES.md` 记的结论是**复用 `src/majiang/strategy/features.py` 的 29 维**。
理由来自该模块自己的注释：数据生成与线上推理共用本模块，各写一套会产生 train-serve skew。
要扩特征就必须改 `features.py` 并重跑价值模型对拍，不是在 `research/` 里悄悄加一列。
29 项特征已逐行对照取值表达式映射到公开信息来源（本人手牌 / 各家副露 / 已打出 / 牌墙 / 庄位 / 局号 / 财神状态）。

### 三个工具（都可复现、都带门禁）

| 工具 | 作用 | 已实测 |
|---|---|---|
| `research/fingerprint.py` | 数据指纹（内容哈希，与路径无关） | 同种子两批分片指纹一致 `e3a73372…` |
| `research/record.py` | 训练记录：指纹 + 超参 + 种子 + 依赖版本 + GPU | 省略 `--seed` → 退出码 2 拒绝执行 |
| `research/compare.py` | 对拍门禁：跑配对 A/B 并按「场数 ≥200 + \|t\|>1.96」机械判定 | 4 场/臂 → 「不显著（样本量不足）」，默认决策器不变 |

`compare.py` 的直接动因是 `notes/OWNERSHIP.md` 里那条教训：A 曾据 6 场/臂、p≈0.10 说「有 2.8 点优势」。
门禁把它变成机械的，不靠谁记得。

### 训练骨架与纯 Python 前向

- `research/train_nn.py`：显存硬上限 `set_per_process_memory_fraction`（默认 6000/8151MB）；
  每 epoch 落检查点，续跑时**校验数据指纹**（换数据即拒绝续跑）；产物 JSON 含权重 + 归一化参数。
- `research/pure_forward.py`：stdin 级依赖——**只 import 标准库**。用 uv 托管的裸解释器
  （无 numpy/torch）实测：`--audit` 通过、同一输入两次推理完全一致。
  这条是硬约束的落地：参赛运行路径 `src/majiang/**` 零第三方依赖，不可能装 torch。

### 数据分片口径（顺手记下，避免以后再猜）

价值数据分片是 `x` / `y` 两个 float32 数组，写盘时**不是** `--out` 那个路径，而是
`<out 的 stem>.part000.npz`。第一次验证时我按 `--out` 找文件，找不到，误判成「没写盘」。

### 待办

- 6.1 训练栈正在装（`pyproject.toml` 的 dev 组已加 `torch>=2.6`，`uv sync` 后台跑）
- 装完补 6.4/6.5 的实测，并跑通 6.6 的「新模型 vs 基线」那一侧
- 仍未动的 4 项：1.2、7.1、7.2（等链路）、6.6 的候选侧（等训练）

---

## 2026-09-27 训练栈就位（进度 29/33）

### 6.1 实测

`torch 2.14.0+cu130`，`cuda True`，RTX 5060 Ti 8123MiB。`uv run --no-dev` 仍能导入 `majiang`，
`src/majiang/` 第三方导入计数 0 —— 零依赖约束没被破坏。

### 6.4/6.5 实测（都是真跑，不是「应该能行」）

| 验证 | 结果 |
|---|---|
| 显存硬上限 | `--max-vram-mb 2000` → 占 24.6%；默认 6000 → 占 73.9% |
| 检查点续跑 | `--resume` 从 epoch 2 续跑，继承历史最佳 363.3949 → 357.4796 → 348.7875 |
| 换数据续跑防护 | 指纹 `e3a73372…` vs `b5d1a647…` 不一致 → 拒绝续跑，退出码 1 |
| 纯 Python 前向（无 torch/numpy 的裸解释器） | 对 torch 训练产物 29 维输入两次推理一致（`-1.2016322606501337`），`--audit` 通过 |

顺带修掉一处 torch 告警：`float(loss)` → `float(loss.detach())`。

### 发现一个**边界问题**，不自行越过

6.6 的「候选侧」要把 NN 产物与 `heuristic` / `value` / `risk` 在同一批样本上对拍，而对拍工具
（`tools/ab_test.py`）只能按**决策器名**取候选。要让 NN 成为可对拍的一档，必须在
`src/majiang/strategy/**`（决策器注册）与 `cli.py` 的档位表里加一项 —— 这两处都在我的禁入区
（A 的地盘，见 `AGENTS.md` §3）。

所以 6.6 现在只完成到「门禁可用」，候选侧**卡在所有权而不是技术上**。两个出路：

1. 往 `notes/THREAD.md` 发一条请求，请 A 加一个 `nn-value` 档（研究产物先只读放进 `research/`，A 只加注册与加载路径）；
2. 或先建一个 OpenSpec 变更，明确「研究线产物如何进入策略档位」的接口，再动手。

**我没有自行改 A 的文件，也没代替人做这个决定。**

### 需要人的动作（只差这一步）

浏览器登录 `https://codemaker.nie.netease.com/hub` → 下载 **Linux AppImage** → 存到默认下载目录。
然后 `bash agent/deploy/install-hub-wsl.sh` 会解包到 `~/.local/opt/codemaker-hub`、用 WSLg 起 GUI、
自动对比监听端口找出代理端口，最后 `relink.sh` 一步切链路，解锁 1.2 / 7.1 / 7.2。

---

## 2026-09-27 链路打通，32/33

### 最终链路（已写进 `agent/deploy/LINK.md`）

```
http://172.19.80.1:15721/v1         # Windows 侧 Hub，经 netsh portproxy 暴露给 WSL
```

- Windows Hub 的代理服务**随主程序自动启动**，不用手动开；端口靠
  `netstat.exe -ano | grep 127.0.0.1` 查出（pid 对得上 `tasklist` 里的 `codemaker-hub-gui.exe`）
- **鉴权不校验**：带与不带 `Authorization` 的 `POST /v1/chat/completions` 都是 `200`
- **`/v1/models` 在此版本不可用**（404 `Models endpoint is not available`），
  所以可用模型只能靠试探确认 —— 实测 `deepseek-flash` ✅，`deepseek-v3*` / `deepseek-chat` ❌

### 模型改成了 deepseek-flash

按用户要求把默认模型从 `claude-sonnet-4-6` 换成 `netease-codemaker/deepseek-flash`，
降级模型保留 `claude-sonnet-4-6`。改法是 `configure.sh` 的默认值 + 模型清单。

顺带解决一个配置写入的坑：`openclaw config patch` 对数组默认是「替换」语义，会被守卫拦下
（`it would remove existing entries`），而它**没有** `--merge` 选项；正确做法是
`--replace-path models.providers.netease-codemaker.models` 显式声明「这个数组归本脚本管」。

### 端到端验收（7.1 / 7.2 都过了）

- **7.1**：`openclaw agent -m` 经网关跑 `uv run pytest -q tests/test_tiles.py tests/test_shanten.py`
  → 回 `75 passed in 10.96s`，手动基线 `75 passed in 11.15s`，一致。
- **7.2 越界演练**：下三条越界任务（改 A 的 `strategy/policy.py`、改 B 的 `scripts/supervise.sh`、
  启动打平台的 `python -m majiang`），**三条全拒**，且逐条引用 `AGENTS.md` §3 的对应条款，
  给出 `notes/THREAD.md` 与「需人授权」两条出路。边界机制按设计生效。

### 又一次自伤（同 OWNERSHIP.md 记过的坑）

排查进程时用 `pkill -f codemaker-hub-gui`，把自己那条 bash wrapper 一起匹配杀了，命令输出为空。
改用 `pgrep -af 'squashfs[-]root'` 这种括号写法。已记进 `LINK.md`。

### 剩 1 项

6.6 的候选侧：需要 A 的决策器注册表开一个 `nn-value` 档，属于我的禁入区。
**我没有自行改，也没代替人做决定** —— 等指令（发 THREAD 请求给 A / 先建 OpenSpec 变更定接口 / 搁置）。

---

## 2026-09-27 晚：接上公司 MCP 与 CodeMaker 委派

### MCP（KM 知识库 + 网页搜索）

两个都是公司托管的**远程 streamable HTTP**，配置抄的是 CodeMaker 自己的 `~/.codemaker/mcps.json`：

```
KM-MCP    https://mcp.netease.com/servers/km-mcp/mcp
websearch https://mcp.netease.com/servers/websearch-mcp-server/mcp
```

**关键教训：`openclaw mcp probe` 只列工具、不鉴权。** 两个服务在没带 token 时 probe 都显示「2 tools」，
真调才报错 —— KM 报 `Token 验证失败 user=None`，websearch 报 `缺少 X-Access-Token（必传）`。
**所以 MCP 接入的验收标准只能是真调一次，不是 probe 通过。**

鉴权头：`X-Access-Token` = 网易 v2 Token，`X-Auth-User` = 工号。CodeMaker 是用 `{auth:token}` 占位符
自动注入的，所以它的配置里 headers 看着是空的。

**token 首选来源改成 CLI 的凭据库** `~/.local/share/codemaker/auth.json`
（字段 `netease-codemaker.{access_token,key,refresh_key,expire}`，带 refresh_key 会自动续签），
比手填的 v2 token 靠得住。`configure-mcp.sh` 已按「auth.json 优先、手填文件兜底」实现。

### CodeMaker 委派（小龙虾当主、CodeMaker 当子）

装了官方 CLI：`codemaker 1.18.31-prod-0.4.18-26.9.21`（`~/.codemaker/bin/`，安装脚本会写进 `.bashrc`；
非登录 shell 要自己补 PATH）。

通道是 `agent/deploy/cm.sh` → `codemaker run --format json --pure -m netease-codemaker/deepseek-flash "<任务>"`。
实测一次最简单委派：回复 `ok`，`tokens=8576 cost=¥0.0009`。

**三个坑（都实测踩过）**：

1. **`--format default` 会走 TUI 式输出并等 stdin** → 脚本里就是「卡住不返回」。必须 `--format json --pure`。
2. **不关 stdin 就假死** → 必须 `< /dev/null`。我第一版包装就漏了这条，白等 180s×2。
3. `codemaker run "msg"` 的位置参数是收的（`[message..]`），但上面两条不满足时它根本走不到发消息那步。

配额纪律：委派档固定 `deepseek-flash`，与小龙虾主循环同档；输出固定带一行
`tokens=N cost=¥X`。**两边共用同一账号**，用户 IDE 里的 CodeMaker 会话也算在内。

### ⚠️ 事故记录：用户把 v2 token 直接贴进了对话

token 因此落进了 CodeMaker 会话记录 `~/.codemaker/persisted/sessions/**/*.jsonl`，
且该 token `exp` 解出来只剩约 2.7 小时。

处置：写进 `~/.openclaw/netease-token.txt`（`600`）继续用，同时**已建议用户刷新/重置**把暴露的那份作废；
`configure-mcp.sh` 与 `cm.sh` 的 token 都从文件读、不进 argv 与日志。

**教训**：跨 agent 协作时，凭据一旦进了对话就等于进了所有会话记录文件（那些 JSONL 是明文的）。
以后一律「写文件、不回显、用完即换」。

### 我刻意没做的事

没动 `~/.config/codemaker/codemaker.json`（CodeMaker 的权限与 MCP 配置）—— 那会改变用户 IDE 里
CodeMaker 会话的行为。委派档位用 `-m` 参数指定，不碰全局配置。

### 处置与一个系统性发现

用户按建议去 `console-auth` 网页登出再登录 —— **实测无效**：拿那份令牌打 CodeMaker API 仍返回 `200`。
JWT 是无状态的，网页登出只清浏览器会话，不吊销已签发的令牌（而且我先前算错了 exp，实际是 7 天后才过期，
不是 2.7 小时；已更正）。

本地处置：

- MCP 的令牌来源已从手填文件切到 **CLI 凭据库** `~/.local/share/codemaker/auth.json`
  （实测两份令牌不同：手填 `34999b84…` vs 凭据库 `f6c7b235…`），删掉手填文件后配置仍可重建
- 用 `agent/deploy/redact-secrets.py` 把两个日志里的完整 JWT 替换成 `<REDACTED-JWT>`（共 3 处，行数不变）
- 会话记录 `session-store/…/d44e79e4-….json` 里的完整副本按用户选择**保留**（删了会丢那段对话）

**系统性发现（比这次事故本身更重要）**：CodeMaker 扩展会把**每一条终端命令**记进
`~/.codemaker/log/codemaker_extension.log`，webview 也会记消息体。所以：

1. 凭据只要出现在「对话」或「命令行」里，就会在多个文件留下副本 —— 清一个不够，要按来源清。
2. 更麻烦的是**排查动作本身会制造新副本**：我用字面量 grep 那份令牌片段去复查，
   结果把片段写进了同一个日志。**所以要停手**，别再拿字面量检索凭据。
3. 真正的解法只有一个：**去签发方重置令牌**。清文件只是降低本地残留。

结论写进纪律：**凭据一律走仓库外文件、不回显、不进命令行；一旦暴露，处置顺序是「先重置、再清残留」**，
而不是反过来。

---

## 2026-09-28 11:45 — 新问题 1「能胡时不考虑杠」的频率统计（自主推进，未等 A 点头）

判据目的：新问题 1 是否值得在截止前动默认策略。数据源 = `logs/*.jsonl`（296 个文件，
真机 `heuristic` 运行日志，只读）。

### 发生率（分母逐层给出）

| 口径 | 计数 |
|---|---|
| 全部 `decision.made` | 549,064 |
| `phase=draw`（本人回合决策） | 204,491 |
| **候选含 `hu`** | **5,232** |
| **候选同时含 `hu` 与 `gang(`** | **27** |
| 覆盖场数 | 21 场（共 292 场 / 2,960 局） |

- **「能胡且能杠」占「能胡」决策的 0.52%**，占本人回合决策的 **0.0132%**，
  出现在 **21/2,960 局 ≈ 0.71%** 的对局里。
- **这 27 次里，选择 `gang` 的次数 = 0**：21 次 `hu`、6 次弃胡追爆头（`discard:*`）。
  即该路径**从未被走过**，与只读探针（`agent/verify/probe_gang_vs_hu.py` 用例 3）一致。

### 结论（我的判断）

1. **问题真实存在**（实现层确认：`_choose_turn` 里 `any(HU)` 短路，`_best_gang` 永不执行），
   但**频率极低**——不到 1% 的对局、约 0.5% 的「能胡」时刻。
2. 按 AGENTS.md 的验收纪律（报「无显著增益」比报好看数字有价值），**不建议在 10/8 截止前
   为此改默认策略**：改动会触及 `_choose_turn` 的核心控制流，收益上界 = 那 0.7% 对局里的
   「多一次自摸 + 链 +1」，但要重做一轮四座位旋转 A/B（自对弈噪声底只支持方向判据）。
   投入产出比不划算。
3. **建议**：登记为已知边界（连同「D11 杠优于飘未落地」一句），留作赛后可选项；
   若要动，先做最小改动（在 `_choose_win_or_piao` 的候选里加入杠的期望比较），
   并配 `agent/verify/probe_gang_vs_hu.py` 用例 3 作回归。

### 口径与局限（写在结论旁边，别丢）

- 「候选含 `hu`」是**动作层面**的能胡，不等于「当时应当胡」——其中 6 次策略主动弃胡追爆头。
- 27 个样本只覆盖 21 场，**样本量小**；0.71% 的对局占比有不确定性，但量级不会变。
- 日志只含 `heuristic` 档；其它档位（v2/first-legal）的该频率未统计。

---

## 2026-09-28 11:20 — T1：NN vs GBDT 价值函数对拍（新路线可行性第一层）

来由：用户转来豆包对「新路线（NN 价值网络 + 决策框架不变 + 开关接入）」的建议，要我校验。
**核对结论：该建议的「路线 2」在本仓库已落地（GBDT 版）**——`models/value_model.json`（200 棵树，
29 维）+ `strategy/value.py:ValueDecider`（只改出牌、胡/杠/飘仍交启发式）+ `cli.py` 的 `value` 档位
（模型缺失自动回退）+ `research/` 完整流水线（`train_nn.py` / `pure_forward.py` / `compare.py` / `FEATURES.md`）。
豆包「手写价值函数 + PIMC」的现状描述与实际不符。

因此 T1 只回答一个具体问题：**把价值函数从 GBDT 换成轻量 MLP，回归质量有没有增益？**
（端到端策略增益是另一回事，6B.11 已给过「模型驱动 vs 常数 无显著差异」的负结果。）

### 方法

- 数据：**同一份** `data/value_post.part000.npz`（训练 139,669）/ `data/value_valid.part000.npz`
  （验证 13,653，独立种子、按整局分开）。**不重新生成数据**，故几乎不占 CPU。
- 四臂：常数 / 现有 `models/value_model.json`（纯 Python 求值）/ 同数据重训 sklearn GBDT / MLP。
- 指标：MSE、MAE、R²、Spearman（手算秩相关），以及**按手留财神数分桶的 MSE**（对齐 B 的病灶）。
- 脚本 `research/t1_nn_vs_gbdt.py`；记录 `research/records/t1-nn-vs-gbdt-final.json`。
- CPU 谦让：`nice -n 19`、`OMP/MKL_NUM_THREADS=1`、torch 线程 1；GPU 显存硬上限 2000MB。

### 结果（验证集 13,653 条）

| 臂 | MSE | MAE | R² | Spearman |
|---|---|---|---|---|
| 常数 | 136.52 | 7.747 | 0.000 | +0.025 |
| GBDT（现有 models/） | 124.47 | 7.605 | +0.088 | +0.232 |
| GBDT（同数据重训） | 124.36 | 7.610 | +0.089 | +0.236 |
| **MLP（hidden 128, lr 1e-3, dropout 0.2）** | **122.35** | **7.461** | **+0.104** | **+0.238** |

- MLP 相对现有 GBDT：MSE **−2.13（−1.7%）**，配对 t 检验 **t=+1.87, p=0.061**——
  **未过 |t|>1.96 门槛，按验收纪律只能记「不显著（边缘）」**。
- 按财神分桶：MLP 的优势集中在 **0 财神**（+2.29）；**1 财神反而略差（−1.44）**、
  2 财神明显更差（−14.91）。**即它没有对准 B 锁定的病灶，反而在病灶格上更差。**

### 结论

1. **换 NN 在表示能力上最多是边缘改善，且没有对准真正的缺口。** 结合 6B.11（模型驱动 vs 常数
   无显著差异），**不建议把 T1 结果作为换 NN 的依据**。
2. **可复用的正面结论**：管线是通的，MLP 能在 6 秒内训完、纯 Python 前向模式已有
   （`research/pure_forward.py`）；若 T2 要试序列表示，基础设施已就绪。
3. **仪表教训（记下，防重犯）**：首跑 lr=1e-3 + 无正则时 MLP **epoch 0 即最优、第 20 轮验证
   MSE 从 125 涨到 158**（早停）。这是**学习率/正则的配置问题**，不是「NN 学不动」的结论——
   调成 lr 1e-4~1e-3 + dropout 0.2 后 MLP 立刻反超 GBDT。**别用过拟合的首跑当结论。**

### 下一步建议（T2，需与 A 协调）

真正的「新路线」不是换模型家族，而是**表示升级**：29 维里 `discards_0..3` 只用了弃牌**张数**，
**弃牌序列的信息在表示层就被丢掉了**——任何模型（GBDT 或 NN）在这套特征下都学不到。
T2 = 对手弃牌序列 embedding + 价值/策略头（仍只用公开信息），但要改 `features.py` 契约
→ 会同时影响价值模型训练与推理两侧 → 需走 OpenSpec 变更并与 A 协调（`features.py` 在 A 地盘）。

---

## 2026-09-28 12:05 — 更正：真实事件流**没有** `hu` 事件类型（我 11:20 条目写错了）

A 在 11:45 条目里点了这条要我核实。核实结果：**我错，B 对。**

- **我写错的原句**（本文件 2026-09-28 10:55 条目）：事件类型集含 `hu`。
- **实测**（`data/auto_sessions/*/events/*.json`，**23,097 局**）真实类型集只有 9 种：
  `timeout / tile_discarded / tile_drawn / pass / chi / peng / round_ended / gang / game_ended`
  ——**不含 `hu`**。
- 我犯错的来源：把**引擎的 `KNOWN_EVENT_TYPES`** 与**实测类型集**混在了一起。两者确实不同：

  | | 内容 |
  |---|---|
  | 引擎 `runtime/engine.py:KNOWN_EVENT_TYPES` | `tile_drawn, tile_discarded, timeout, chi, peng, gang, **hu**, round_ended, **settled**` |
  | 实测事件流类型集 | `timeout, tile_discarded, tile_drawn, **pass**, chi, peng, round_ended, gang, **game_ended**` |

  即引擎白名单**含 `hu`/`settled`（实测从未出现）**，而**不含 `pass`/`game_ended`（实测大量出现）**。
  后者落到「未知类型」分支 → `_should_fetch_for_events` 返回 True → 保守重拉快照。**是保守正确，不是 bug。**
- **胡牌的表示**：只在 `round_ended` 的 `data` 里——`{draw: bool, detail: [番型名], fan, scores, round_no}`。
  23,097 局中 22,516 胡、581 流局；`detail` 是多标签（番型链会并列多条，故 detail 条目总数 > 胡牌局数）。
  番型频次：平胡 78.97% / 爆头 16.89% / 七对 2.48% / **杠开 0.82%** / 财飘 0.62% / 4 白板 0.09% / 豪华七对×1 0.08% / 双财飘 0.04% / 杠飘链×2 0.01% / 连杠×2 0.01%。
- **教训**：不要把「代码里的常量表」当成「数据里的实测分布」来引用。引用实测必须给出
  `grep`/脚本与样本量；引用代码常量必须区分「白名单」与「实际出现」。

### A 要的盈亏平衡点：杠后补牌成胡率

`agent/verify/gang_replenish_rate.py`（只读，297 日志 / 2,970 局）：

| 项 | 值 |
|---|---|
| 本人 `gang(` 决策 | 1,086 |
| 其中紧邻的下一条本人决策是 `phase=draw` | 1,060 |
| **该补牌选择 `hu`** | **29（2.7%）** |
| 该补牌选择弃牌 | 1,029（97.1%） |

对照 `chase_baotou` 的 **88.7% 自补率**——**两者差两个数量级**。判读：杠后补牌成胡罕见
（与番型频次一致：`杠开` 仅占 0.82%），故「放弃确定的 1–2 番胡去赌杠开」的盈亏平衡点极高。
**支持「不建议为能胡+能杠改默认策略」的结论，并给出了可比的量化依据。**
局限：决策层统计（补牌后我们选了 hu ≠ 牌型上成胡）；只取紧邻的下一条本人决策。

---

## 2026-09-28 11:50 — 真机 `action.rejected` 解码（含我自己的口径修正）；T1 财神分桶噪声底复算

### 一、`action.rejected`：先修口径，再解码

我上一轮把 `good-ember` 的 `action.rejected=832` 当全量报了。**这个数是错的**——
`logs/*.jsonl` 里混了**测试房**（`game_id` 前缀 `t_`）与**本地试跑**
（`run1/run2/run3/run-firstlegal.jsonl`）。按**房**分层后：

| 分层 | 拒绝数 |
|---|---|
| 真机（文件名 `a_<12hex>.jsonl`） | **503** |
| 测试房（`t_*`） | 333 |

→ 真机拒绝率 **0.23%**（503 / 218,036 `submitted`），不是我上轮说的 0.38%。
**「按房分层不按文件」这条纪律又救了一次口径。**

真机 503 次的构成：

- 按 phase：`draw 397 / response_chi 66 / response_peng 40`
- 按动作族：`discard 393 / chi 66 / peng 37 / gang 4 / hu 3`
- 全部 `code=INVALID_ACTION`

**自愈率 94.2%（474/503）**：拒绝后紧跟同族 `action.submitted`（重试成功）。
29 个未自愈里 **24 个是 response 相的吃/碰**——那是「别人先动作、机会窗口已关」的竞态，
不是我方动作非法。

**`hu` 被拒仅 3 次**，逐局核对事件流副本（用 `round_no` 精确匹配）：

| 文件 | 座位核对 | 那一小局结局 |
|---|---|---|
| a_98a4a40a4289 | 我们座 3 | 我们胡（平胡） |
| a_a9e5f03aa86b | 我们座 1 | 我们胡（平胡+爆头） |
| a_d37f4dd7fa76 | 我们座 0 | 我们胡（平胡） |

→ **实测丢胡 0/3**。对照：`action.submitted` 且 `action=hu` 有 **4,424 次**，被拒率 **0.07%**。

**判读**：这是**可达一致性缺口**（基于稍旧视图 / 同 tick 竞态），**不是策略链路失效**，
也不是「我们提交非法动作」。据此**不建议**为此改策略。若要进一步收敛，
可考虑提交前用**最新快照**校验合法性（`runtime/**`，A 地盘）。

脚本：`agent/verify/analyze_rejected.py`（只读、零平台请求）。

### 二、T1 财神分桶：噪声底复算（修正我先前的过度解读）

方法：`research/t1_gods_noise.py`，逐样本平方误差差的**观测 bootstrap**（B=10000），
复刻 T1 final 超参（含**按验证集选最优 epoch**——漏了这步 MSE 会飙到 155，我踩过）。
复现 MSE=**122.3458**，与 T1 记录**逐位一致**（仪表对齐）。

| 桶 | n | MLP−GBDT 差 | 95% CI | 判定 |
|---|---|---|---|---|
| 全体 | 13,653 | **+2.13** | [−0.08, +4.37] | **不显著**（与 T1 的 p=0.061 一致） |
| gods=0 | 7,716 | **+2.29** | [+1.23, +3.37] | **显著（MLP 好）** |
| gods=1 | 4,968 | −1.44 | [−4.24, +1.33] | **不显著** |
| gods=2 | 917 | −14.91 | [−31.27, +0.45] | **不显著**（跨 0，样本仅 917） |

**修正结论**：我先前说「MLP 在财神格反而更差」——**这句要收回一半**。
在噪声底下，`gods=1`、`gods=2` 的劣势**都不显著**（`gods=2` 点估计虽大但区间跨 0，且样本少）；
唯一显著的是 `gods=0` MLP 更好。即：**不是「它在病灶格更差」，而是「病灶格样本太少、无定论」**。
这反而支持 T1 的原判——缺口在**表征**，不在模型家族。

### 三、教训（记下防重犯）

1. **全量统计必须按房/按来源分层**：混入测试房与本地试跑会把 503 说成 835。
2. **复现必须先对齐仪表**：先让复现 MSE 与原始记录一致，再谈分桶结论；
   漏「按 val 选最优 epoch」这一步，会得出完全相反的桶结论。

---

## 2026-09-28 12:25 — T2 结论：**弃牌序列没有可用增益**（8 种子 × 参数对等对照）

### 交付物①：序列表征能否降低 held-out MSE / 提高前 1 排序一致率？

**可行性结论：不能（在噪声底之上不能）。**

口径：训练 99,176 / 验证 19,881 样本，**训练与验证来自独立种子批次**（按整局分开，无同局泄漏），
标签 = 本局本人最终得分，`seq` = 四家**公开弃牌时序**（左补 PAD，**不含对手暗牌**）。
**8 个种子**各跑一遍，报**跨种子**分布（单个种子的 p 值在 N≈2 万下会把 1% 的微小差异判成显著，不可单看）。

| 臂 | MSE（跨种子均值） | Spearman |
|---|---|---|
| 常数 | 163.34 | 0.0000 |
| GBDT（29 维） | 143.807 | +0.2490 |
| MLP（29 维） | 143.725 | +0.2545 |
| MLP + 弃牌**无序**嵌入（参数对等） | 143.431 | +0.2539 |
| MLP + 弃牌**序列** | **143.064** | +0.2523 |

**配对（跨 8 种子）**：

| 对比 | 均值 | 正号 | 单样本 t | 判读 |
|---|---|---|---|---|
| 序列 − 无序嵌入（**顺序**的贡献） | **+0.44** | 4/8 | +1.18 | **不显著** |
| 无序嵌入 − 仅聚合（嵌入内容化） | +0.25 | 5/8 | +1.03 | **不显著** |
| 序列 − 原始袋（参数化） | +1.82 | **8/8** | **+7.24** | 显著，但只是**参数化** |
| 序列 − 仅聚合（净贡献） | **+0.69** | **8/8** | **+3.34** | 显著但极小 |
| MLP − GBDT（模型家族） | +0.08 | 5/8 | +0.97 | **不显著** |

**结论**：净增益 **+0.69 MSE / 143.7 = 0.48%**，且**前 1 排序一致率无改善**
（Spearman：`mlp_agg` +0.2545 反而略高于 `mlp_seq` +0.2523）。
把「无序」表示给**同等参数**后，**顺序的贡献掉到 +0.44、4/8、不显著**。
→ **序列表征在这套数据/规模下没有可用增益。按 A 的 kill criteria 与交付物③：这条线死在这。**

### 仪表教训（本轮最重要的产出之一）

1. **主跑（单种子）与 8 种子结论相反**。主跑给的「序列−袋 p=0.41 不显著、袋−聚合 p=0.15」
   与 8 种子（`+1.82, 8/8, t=+7.24`）**冲突**——单种子运气好时早停点恰好使序列臂变差。
   **小样本调参的结论必须跨种子复核。**
2. **「袋」臂必须做参数对等对照**。我最初用「136 维计数 + 单层线性」，它输给 29 维聚合
   不能证明「内容无用」——那是被削了参数。补上参数对等的**无序嵌入**臂后，
   「顺序 vs 无序」才能在可比条件下回答。**这是 A 的 ① 里「前 1 排序一致率」口径的实体。**
3. **复现先对齐仪表**：先让 MSE 与原始记录一致，再谈结论。本轮先复现 T1（122.3458 逐位一致）
   才开始 T2 的桶分析。
4. **`action.rejected` 必须按房分层**（真机 503 vs 测试房 333），否则把 835 当真机问题。

### 交付物②：能翻译成启发式项的具体特征？

**没有。** 序列的信息量经参数对等对照后不显著，且它对**排序**无改善（决策器真正用到的就是排序，
不是绝对 MSE）。**给不出可翻译成 `policy.py` 一项的具体特征。**

### 交付物③

不导出 JSON 前向（本就是一条件，现在判定线死）。

### 下一步建议（我自主选择的）

按 A 的 11:45 条目：**爆头缺口**（我们 14.9% vs 对手 22.7%）是他当前第一优先且要我接。
我接规则侧的二分问题：对手到爆头态靠的是**留财神**还是**多组自然面子**。
这正是 T2 之后的下一步；数据与探针方法我已在事件流里（`data/auto_sessions/**`）。

产物：`research/t2_seq_vs_agg.py`、`research/gen_value_seq_data.py`、
`research/records/t2-seq-vs-agg.json`、`t2-rob-{种子}.json`、`t2-fair-{种子}.json`、`t2-rank-{种子}.json`。

