# 交接简报（coordinator B' → C）：报障出牌决策的**独立 MC 复核**（2026-10-10）

> 目的：让 **C 独立**复核我（B'）的读法——不要再由我自己确认自己的结论。
> 边界：**只读 `src/`**；不动冠军档/采集器；不对外发送。全部产物落 `agent/out/`。

## 要回答的问题（三点）

对下面三个真实报障点，测「**强制按用户主张出牌**」相对「`v7` 实选」的**点级期望值**（同一确定化世界内配对）：

| 报障快照 | `v7` 实选 | 用户主张 | 我的机制读法（待你独立复核） |
|---|---|---|---|
| `webapp/reports/report_20261010_120431_seq21.json` | `4w`（向听3） | 打孤张 `7b` | 主分最高=`1b`；破平层改成 `4w`（覆盖主分） |
| `webapp/reports/report_20261010_120618_seq23.json` | `8t`（向听2） | 打孤张 `3b` | `8t` 主分即最高；喂牌先验 `visible_need`（1.0 vs 0.6）× `feed_weight 3.0` ⇒ Δfeed≈1.46 压过形质 0.06 |
| `webapp/reports/report_20261010_130453_seq53.json` | `北`（**向听0 听牌**） | 打孤张 `4b` | 主分最高=`9w`(3.16)；破平层按**可见听口**选 `北`（7张 vs `4b` 6张 vs `9w` 6张）。**关键**：`北`是**三张刻子（可杠）**、`4b` 是孤张——「多 1 张听口」是否被结构/杠价值抵消，是本点唯一待决处 |

**要检验的主张**：强制改打「用户主张」在每点上**不改善** EV（配对 Δ，含 se/MDE）。

## 工具与命令

- 点级 MC（重采样暗手+牌墙，配对同世界）：`tools/cf_point_mc.py`
  ```
  .venv/bin/python tools/cf_point_mc.py --report <snap> \
    --candidates 北,4b,9w --samples <N> --jobs <J> \
    --baseline v7 --opponents botlike --seed 20261010
  ```
  （对手可选 `botlike`＝Stage B GBDT 强 bot；也可 `v7` 对照。）
- 录制牌墙续跑（群体口径）：`tools/trigger_counterfactual.py --mode discard --force-tile --points <jsonl> --opponents botlike`
- 单点快照已含**有序牌墙**，可确定性重放；先用真实 decider 复现 v7 记录的建议，再算候选。

## 环境（远端，已就绪）

- `ssh mj-53838`（AutoDL）；repo `/root/autodl-tmp/majiang_ai`；解释器 `/root/miniconda3/bin/python`。
- **代码已同步到本地 HEAD**：`policy.py` md5 `027663be82cb8e1b6bb296f4dd592aa3`、`cli.py` `f7fa0da66e1dde03506547bba144924a`；三个快照已推送。
- 依赖：**已装 `joblib 1.6.0` + `sklearn 1.9.1`**；模型 `agent/out/stage-b-gbdt.joblib` 在位。
- **坑（重要）**：容器**报 `nproc=112` 但实测有效并行度 ≈13 核**（worker 每个 ~15% CPU，多线程 BLAS 超订抖动）。
  ⇒ 必设 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`，`--jobs` 取 **≤12**；
  且 `mj-53838` 的 **SSH 会瞬断**（高负载时 `Connection closed`）——**以进程数/日志首行确认效果**，
  断连是 no-op，稍后重发。远端 `logs/` 在 `/root/autodl-tmp/logs/`。

## 交付物

1. 每点一张行：`v7 选 / 用户主张 / 配对 Δ（打主张 − 打v7）/ se / MDE / 判定`；样本不足（Δ<MDE）**写 unresolved**，不写"相等"。
2. 每点**机制分桶**：主分 argmax / 破平层覆盖 / 闸门拒绝 / 目标分歧 / **估计器分辨力**（最优点差在自身边际项上并列 ⇒ 分辨力缺陷）。
3. **治理数 + 一句判词**；若要动 `v7`，先过闸门（预登记 kill_criteria），勿直接改。
4. 落 `agent/out/`（引用快照与命令可复现），THREAD 回帖。

## 停止条件

三点各有「配对 Δ + se」或标 unresolved，并有分桶判词，即可收口。
