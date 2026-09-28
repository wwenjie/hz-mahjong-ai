# agent-d 记录（NN/RL 线 · 独立仓 `majiang_rl`）

**身份**：OpenClaw `team-team-coordinator`（Chief of Staff）。独立仓
`/home/wuwenjie01/majiang_rl`（对主仓 `/home/wuwenjie01/majiang_ai` **只读**，
代码层由 `src/nnrl/paths.py` 强制）。

**通道**：本仓 `notes/THREAD.md`（与 A/C 交互）+ `sessions_send` 至
`agent:main:main`（= agent-c）。**本文件只追加。**

**边界**：不碰 `src/majiang/strategy/**`、`src/majiang/runtime/**`、`sim/replay.py`；
不启平台进程；不杀 A/C 的进程；离线重活 `nice`、同一时刻只跑一个。

---

## 2026-09-28 21:25 — 正式接入 + 用户新任务（对比水平）

### 用户 21:17/21:23 两条指令
1. 未来约 2 小时用户不在，**授权本线自主决策**。
2. 点名：「可以尝试跟 `/home/wuwenjie01/majiang_ai` 的方案尝试下自对弈或者 abtest，
   对比下水平」。
3. 明确授权：可以 **agent-d 身份**，按主仓 md 文件写的方式与 A/B/C 交互。

### 我据此做的事（本轮）
- **纠正一条影响全部历史结论的口径**：本线此前所有 A/B 都用 `--baseline heuristic`，
  而 `heuristic` == **v2**（`cli.py` 零 override）；**冠军是 v3**。对手选错。
- 新建 `src/nnrl/allarms.py`：把本线三条模型臂（`rl` / `mlp-value` / `policy-bc`）
  注册进评测器 `ev.CUSTOM`，供 `--baseline v3` 直接对拍（主仓零改动）。
- 串行队列 `scripts/queue_arms_vs_v3.sh`：等 `search@v3` 链退出后跑
  「三臂 vs 冠军 v3，20 场 × 2 种子 × 四座位旋转」，产物
  `records/ab-arms-vs-v3.json`。冒烟已验证三臂可构造。
- 收口：22:10 自动化读取结果并播报。

### 已结算的结论（供 A/C 引用）
- **`search-v3 vs v3`：同向显著负**（命中预登记规则 1）。
  名次分合并 −1.068 (t−4.74)、胡次数 −0.323 (t−5.23)、番数 −0.516 (t−5.67)；
  两种子同向。`records/ab-search-v3-vs-v3.json`（21:11 落盘）。
- **`search-v3 ≡ search`**（v3 底在搜索路径是**空干预**）：我读码与 C 的独立实证
  （`agent/verify/search_v3_base_probe.py`，21 听牌局面 0 差异）一致。
  详见 `notes/REPORT-B-v3-recheck.md`。
- **`rl@v3 ≡ rl@v2`**：RL 候选面遍历全部可打牌、`_score_discard` 不读
  `wait_aware_tenpai`，故换基座是空干预（C 独立复核 `694fd6b`）。

### 待办/在轨
- ⏳ 三臂 vs 冠军 v3 队列（等链退出）。
- ⏳ `field=search-deep-v3` 对照场（抗偏置，非判据）。
- ⏳ 用户点名的「对比水平」结论 = 上述队列落盘后给出。

---

## 2026-09-28 21:36 — 口径补记：三臂内层=v2 底；"三臂 vs v3" 该怎样归因

### 事实（逐一核实，非推断）
本线三条模型臂的**内层启发式都建在 v2 底**（`PolicyConfig.for_mode(mode)` 零 override）：

| 臂 | 内层构造 | 出处 |
|---|---|---|
| `policy-bc` | `HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))` | `policy_decider.py:25` |
| `rl` | `HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))` | `rl_play.py:27` |
| `mlp-value` | `HeuristicDecider(PolicyConfig.for_mode(...))` | `decider.py:85` |

`nnrl/` 全仓**零引用** `wait_aware_tenpai`（仅 `v3arms.py` 自建臂的 docstring）。

### 归因（结构上与搜索同源 ⇒ 内层基座差异是空干预）
三个臂都**只重排「出牌」**（：网络在候选面内重选），而非出牌动作（胡/杠/弃）仍委托内层。
而 `wait_aware_tenpai` 只落在 `_choose_discard → _break_ties_by_ukeire`（**仅出牌**分支），
候选特征又来自不读该开关的 `_score_discard`。
⇒ **与 C 已实证的 `search-v3 ≡ search` 同一条机制**：把这三臂的内层从 v2 换 v3 也是**空干预**。

**推论**：`arm(v2内层) vs v3` 干净地测**臂自身贡献**，不混入"v2 handicap"。
（与搜索不同的一点：搜索**丢弃**内层选的牌、只用其候选；三臂同理，故口径一致。）

**未独立实证**（诚实标注）：以上是结构推断，**我没有**对三臂重跑 `v3 内层` 的逐位等价探针
（C 对 RL 的 `rl_base_equivalence_probe.py` 已证 `rl@v3 ≡ rl@v2`，覆盖 `rl`；
`policy-bc`/`mlp-value` 未单独探针，但 `candidate_features`/`_score_discard` 同源）。
若需硬结论，后补一次探针即可（成本低，需安静窗口）。
