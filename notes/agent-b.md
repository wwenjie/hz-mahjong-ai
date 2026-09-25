# Agent B 记录（追加式，不要改历史行）

格式：时间 / 结论 / 证据（命令 + 数字）/ 改动的文件 / 留给 A 的下一步

---

## 2026-09-26 01:58
- **结论**：基线绿（412 passed）。数据已增至 115 房 / 1129 文件（简报写 112 房 / 1110 文件）。开工前摸清了简报没写的事件流结构事实，复算脚本按这些事实写。
- **证据**：
  - `uv run pytest -q` → 412 passed in 27.77s
  - `find data/auto_sessions -name '*.json' -path '*events*' | wc -l` → 1129
  - 全量事件类型统计：`timeout` 988240 / `tile_discarded` 329681 / `tile_drawn` 296941 / `pass` 261834 / `chi` 17553 / `peng` 15855 / `round_ended` 8962 / `gang` 1864 / `game_ended` 1120
  - 结构事实（简报之外，实测）：
    1. `blocks` 是按 **≤128 事件分页**的块，同一 `round_no` 常跨 2~3 个 block；仅每局**首 block** 带非空 `start_hands`（庄家 14 张，其余 13；后续 block 为 `[None×4]`）
    2. **没有 `hu` 事件类型**——胜负只在 `round_ended.data`（`draw/fan/detail/scores`）；流局时 `seat=-1`、`draw=true`
    3. `gang.data.kind ∈ {ming, bu, an}`（全量计数待我脚本复核，与 A 报的 1828 次同量级：我这里 1864，含增长部分）
    4. `tile_drawn` 的牌面**全场可见**（前 200 文件 52740 次摸牌 0 次空 tile）
    5. `game_ended` 1120 < 文件数 1129：至少 9 份是未完成对局，名次分计算时会单独剔除并计数
- **改动的文件**：`openspec/changes/majiang-ai-agent-b/`（proposal/specs/design/tasks，已过 `openspec validate`，commit 已落）
- **留给 A 的下一步**：无阻塞。我的不变量脚本（`verify/invariants.py`）正在写，守恒结果出来后追加。

### 问 A：你报的手数 8,611 对应哪个数据快照？
数据在持续增长（简报 1110 文件 → 我接手时已 1129）。我复算会附快照指纹（文件数 + 总局数），对比时好扣除漂移量。请把你跑 `measure_strength.py` 时的文件数（或时间点）写进 `notes/agent-a.md`。

---
