# G1 第 2 步实现稿：`v5-godwait`（持财神听口加权）

- **状态**：实现稿，A 23:59 已批准。**policy.py 由 A 实现**（单写者纪律），本稿供 A 落地。
- **Deadline**：~03:00 前落到 `cli.py`（v5-piao05 两条 ~3h 后轮到 G1）。
- **队列位置**：A 23:59 调序——`v6a ×2`（在跑）→ `v5-piao05 ×2` → **`v5-godwait ×4 + v5-godwait-noWA ×2`** → `v7m ×4 + v7m-all ×2`。

## 1. 改动点（policy.py，仅一处）

**位置**：`wait_aware` 分支（当前在 991 行附近，`copies = _wait_copies(...)` 之后）。

**改法**：在 `wait_aware` 为 True 且 `counts_after[GOD] >= 1` 时，把候选比较键从
`copies` 改为 `copies + god_wait_boost × god_n × wait_kinds`。

伪代码（A 可按此直接写）：
```python
if wait_aware:
    copies = _wait_copies(counts, situation.hand.meld_count, visible, score.tile)
    if copies is None:
        self.last_detail["wait_copies_failed"] = ...
        continue
    # 新增：持财神听口加权
    if self.config.god_wait_boost > 0:
        god_n = counts[tiles.GOD]  # 出牌后手牌财神数
        if god_n >= 1:
            waits = win.winning_draws(counts, situation.hand.meld_count)
            wait_kinds = len(waits)
            copies = copies + self.config.god_wait_boost * god_n * wait_kinds
            self.last_detail["god_wait_boost"] = (god_n, wait_kinds, copies)
```

**关键不变量**：`god_n == 0` 时行为与 v5 逐位一致（单测钉住）。

## 2. 新增 PolicyConfig 字段

```python
god_wait_boost: float = 0.0  # 0 = 关闭（v5 行为），>0 = 持财神听口加权系数
```

## 3. 三臂注册（versions.py）

| 臂名 | 配置 | 回答的问题 |
|---|---|---|
| `v5-godwait` | v5 全旋钮 + `god_wait_boost=2.0` | 加权是否有效 |
| `v5-godwait-noWA` | v5 全旋钮 + `god_wait_boost=2.0` + `wait_aware_tenpai=False` | 消融：加权 vs 覆盖原本有效的听口排序 |

（`v5-godwait-presel8` 备选——若 A 认为预筛截断嫌疑（U1）值得单独测，可追加；默认不开。）

## 4. 预登记判据（A 23:50/23:59 已认，固化）

- **机制门（并联，任一不满足即停）**：
  1. 持财神听口种数比（godwait/v5）**≥ 1.25**（对标 bot +34%，收窄到一半以上）
  2. 平胡率**不显著下降**（t<2 方向为负即不过）
  3. 爆头率**上升**（S2 病根的直接验证；t≥2 为正）
- **胜负门**：每场名次分 **>0 且 t≥2** ⇒ 采纳；≤0 或 t<2 ⇒ 财神轴关闭
- **判读窗**：4 种子 × 120 场 × 4 旋转（godwait 主臂）+ 2 种子（noWA 消融）
- **机制量工具**：A 已扩好 `tools/selfplay_meld_rate.py`（同时报副露/平胡/爆头/均番），直接复用

## 5. 单测（绿灯后我补）

- `test_godwait_invariant.py`：`god_n==0` 时 v5-godwait 与 v5 逐决策一致（防「任何改动都扰动」伪信号）
- `test_godwait_selects_wider_wait.py`：构造持财神牌型，断言宽听候选被选中（加权后 copies 值正确叠加）

## 6. 风险与交互

- **与 wait_aware_tenpai 的叠加**：加权只在「出牌后持财神」时触发；`noWA` 消融臂区分「加权有效」vs「覆盖原本有效的听口排序」
- **与 v7m 正交**：改出牌排序 vs 改副露闸门，并行排队互不决定生死
- **性能**：`winning_draws` 在听牌档已被 `_wait_copies` 调用（同一 counts），`wait_kinds` 复用其返回值，**零额外计算**
- **上游嫌疑（备查）**：若 G1 有效但幅度不够，再查 (U1) 预筛截断（ukeire_preselect=5 用 blocks 代理，持财神时宽听候选可能被截）

## 7. 落地检查单（A 实现后我验证）

- [ ] `PolicyConfig.god_wait_boost` 默认 0.0（v5 行为不变由既有测试钉住）
- [ ] `wait_aware` 分支加权 + `last_detail["god_wait_boost"]` 留痕
- [ ] 单测：无财神时逐决策一致；持财神时宽听候选被选中
- [ ] `versions.py` 注册 `v5-godwait` / `v5-godwait-noWA`
- [ ] `experiments.json` 入队（G1 插到 v7m 前，A 已调序）
- [ ] `selfplay_meld_rate.py` 机制门数据前置采（A 自开，不占 A/B 通道）
