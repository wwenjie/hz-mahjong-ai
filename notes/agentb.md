# 新 agent-b（coordinator）记录

> **本文件是新加入的 agent-b 线（coordinator）的追加式记录，区别于冻结只读的 `notes/agent-b.md`（退役 B）。**
> 角色：用户指定的新 agent-b；范围 = 退役 B 移交的独立验证/提交物 + 协调（调度 `agentb-researcher` / `agentb-reviewer` / `agentb-writer`）。
> 边界：不写 `src/**`、`data/**`、`logs/**`、`verify/**`、`agent/out/**`、`research/**`、`notes/agent-{a,b,c}.md`；零平台请求；不碰 A/C 的平台进程与算力队列（`data/experiments/.lock`）。
> 产物：`agent/agentb-*/out/`。**只追加，不改历史行。**

---

## 2026-09-29 11:45 报到 + 项目通读 + 出牌策略差距初步复核

### 已读
- 协作：`notes/PROTOCOL.md`、`notes/OWNERSHIP.md`、`notes/STATUS.md`、`notes/THREAD.md`（含 C 11:47 回执）、`notes/agent-a.md`、`notes/agent-c.md`。
- 方案：`openspec/changes/majiang-ai/{proposal,design,tasks}.md`、`majiang-ai-nnrl/design.md`、`majiang-ai-agent-b/{proposal,design}.md`、`README.md`。
- 代码：`src/majiang/strategy/{policy,risk,versions,routes,features}.py`、`src/majiang/rules/shanten.py`、`tests/test_official_parity.py`、`tools/fan_calc_diff.py`。
- 他人复核：`research/code-review-discard-strategy.md`（评审报告，F1–F12）、`notes/agent-c.md` 11:21 节（C 复核 ①②③④）。
- 平台：用户 11:42 提供的**官方规则 + 变更日志**（存档于对话，不入仓）。

### 项目状态（截至 11:44）
- 数据：353 房 / 357 日志文件；采集进程 `auto_session.py --decider v3,first-legal`（A，11:43 重启）。
- 冠军档 = **v3**（wait-aware tenpai，听牌按可见听口张数选牌）。真机 38 房机制确认（窄桶 1-4 = 0.5%）。
- 队伍：A（策略主线，CodeMaker）、C（小龙虾，独立验证/值守）、我（新 agent-b，协调+独立验证）。
- 截止：10/8 12:00 提交；10/10 首轮；10/12 决赛。

### 出牌策略差距：机制链（与 C 及评审报告一致）
1. **同向听候选排序键退化**：默认 `block_value = 2×面子+搭子`（`quick_blocks` 把 partials 裁剪到 `4-sets`，**恒饱和**）⇒ 真机 77–93% 决策全并列（评审 F1：该数字属旧键，**不是** `shape_value`；`shape_value` 把并列降到 44%，仍高）。
2. **`total` 实际由喂牌一项决定**：`total = -10×向听 + block_value − 3×feed − 财神罚`；实测 `argmax(total) == argmax(-feed)` **97.3%**（评审 F6；C ③）。
3. **喂牌项本身可疑**：`visible_need(tile)` 是**静态牌种表**（字0.4/边0.6/中1.0），**不含已见张数**（C ④a）**也不看「我舍的是不是我的孤张」**（评审 F9）；`threat = Σ ready` 被高估 ≈1.4–1.6×（C 01:38 更正；评审 F7 定位来源：加法先验未归一、共线重复计数、`CONSERVATIVE_UPLIFT=1.25` 从未标定）。
4. **净行为** = 「先打字牌、再打边张、留中张」这张**写死的牌种表**。实测中张出牌：向听 2 只占 **1.3%**、向听 ≥3 **0.0%**（评审 F4）；整体我们比强 bot 少打约 **47% 相对**的中张（A）。
5. **静默门**：`EXACT_UKEIRE_MAX_SHANTEN=1` ⇒ 向听 ≥2（占 43% 出牌点）**一次进张都不算**（评审 F2）。
6. **结果**：到听慢 ~1 摸（6.23 vs 4.96–5.48）、听口窄（11.56 vs 12.2–12.7）、胡率低 ~10pp（19.6% vs 29.9–33.5%）；A 的分解显示到听率差距 **75–80% 是同副露层内的水平差**，不是副露频率。

### 方案层观察（我自己的判读）
- **优点**：测量纪律强（三层验收、按房聚类、噪声底、四座旋转、「未确认」是合格结论）；目标函数锁定 `E[总得分]`（D10）；版本冻结纪律（`versions.py`）。
- **风险 1（方案与实现漂移）**：设计的核心是 D12「两条路线各自估值再比较」，但默认档 `route_aware=False`、`commitment=NONE` ⇒ 走的是 `-10×向听+骨架−3×喂牌−财神罚`，**设计的路线 EV 核心处于休眠**。而实测缺口在**普通胡率/速度**，不在番数/爆头。
- **风险 2（可复现性）**：`wait_aware_tenpai` 之外的新臂（shape/two-ply/……）**没有版本快照**，一旦成为默认就无法对拍（评审 F11）。
- **风险 3（合规/结算）**：已有 `tests/test_official_parity.py` + `tools/fan_calc_diff.py` 对官方口径固化，方向正确；待我这条线独立复核一处规则口径（见下）。

### 待办（我这条线）
- [ ] agentb-reviewer：独立复核 policy/risk/shanten + 官方规则口径对照（含抓打圈/杠爆时机/吃 2 摊/最后 10 墩禁杠/番数连乘）。
- [ ] agentb-researcher：从原始事件流独立复算差距数字 + 候选修法排序。
- [ ] 复核 `verify/**` 归属（与用户确认）。
- [ ] 独立复核 threat 标定的真实倍数（A 1.49× / C 1.4–1.6× / GBDT 0.95×）。
- [ ] 跟进 A 对 THREAD 11:50 三条的定夺。

### 不确定性
- 我的差距数字目前**转引** A/C/评审报告，尚未自算；等 agentb-researcher 的独立复算。
- 评审 F3（截断损失 23.4%）n=47 偏小；F4 分副露表副露 3 组仅 5 点。
- 未跑任何自对弈/真机实验，**不给因果结论**。

---

## 2026-09-29 11:49 用户指令（本线边界更新）

**用户明确要求：本线（新 agent-b / coordinator）不参与修改代码**，以免多 agent 同时改产生冲突。**所有修改建议一律发给 agent-a 与 agent-c 评估**，由他们决定是否落地。

- 本线**只产出分析与建议**，落 `agent/agentb-*/out/` 与 `notes/agentb.md`；建议正文落 `notes/THREAD.md`（唯一可靠通道）。
- **不写**：`src/**`、`tests/**`、`tools/**`、`verify/**`、`research/**`、`agent/out/**`、`data/**`、`logs/**`、任何人 `notes/agent-*.md`。
- 已核对：`git status -- src/ tests/` 干净；两条子代理（reviewer/researcher）只写 `agent/agentb-*/work|out`。
- 已发建议：THREAD 11:50 三条（A）；THREAD 11:55 截断修法对齐（A、C）。

---

## 2026-09-29 13:35 授权后动作

- 用户 13:27 授权我按建议推进、给更高自主权。
- **采纳 THREAD 标识 `B'`**（记录 `notes/agentb.md`），回应 A 12:05 的第一条约定。
- **fan-calc 独立对拍**：`tools/fan_calc_diff.py --cases 60` → 72 例/胡51，**0 不一致**（支持 A ⑨）。
- **回答用户提问**（出牌策略是否推断对手听什么 + 是否按池中剩余选听口）：形成一处候选杠杆——**牌级对手听口推断缺失**；己方听口池量已有（v3 `_wait_copies`）但**只算可见张**、对手暗手仍计为可用 ⇒ 偏乐观。已作为 OPEN 建议落 THREAD 13:35。
- 边界：零平台请求（仅免认证 fan-calc）、未改 `src/**`。

---

## 2026-09-29 14:45 Plan A 生效（用户 14:42 批准）

- 用户从「方案 A / A′」里选了 **Plan A**：我接管实验队列的**注册前提签**（不接管调度权）。
- 已落 `notes/PROTOCOL.md`：§0 编制补充 + 新增 **§5.6 实验注册闸门**。规则：新臂写入 `notes/experiments.json` 前须有我 THREAD 的「登记签」，只查**前提新鲜度 / 判据预登记 / 臂冗余**三件事；**A 保留最终裁决权**（我驳回须写明理由，A 可显式「知悉、照跑」，我不拦第二次）；旧臂不追溯；签核 ≤1 个工作日，超时视为无异议。
- 已发 THREAD 14:45 通知 A/C。我的边界不变：不改 `src/**`、零平台请求、不碰队列调度权。
- 仍 OPEN：A 对 THREAD 11:50 三条（截断修法/feed_weight 重登记/结构门）的定夺；用户问的「牌级对手听口推断」杠杆（THREAD 13:35）。

---

## 2026-09-29 17:05 用户批准选项 1：跨仓臂纳入注册闸

- 用户在「纳入 / 豁免+降载 / 现状」里选了**纳入**。
- 已落 `notes/PROTOCOL.md` §5.6 补丁：任何离线自对弈臂（不分仓）须登记本仓 `experiments.json` 并过签核；跨仓臂由 A/C 代登记；未登记跨仓臂视为未授权占用。
- 已在 THREAD 17:05 正式裁决 A 15:20 的两问（第三种子不算新臂；核预算建议维持 jobs 5）。
- 待办：A/C 确认 + `majiang_rl` 线 `fix-nav` 臂补登或声明豁免。
