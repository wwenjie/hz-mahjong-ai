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
