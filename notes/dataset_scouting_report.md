# Riichi 麻将真实高手牌谱数据源摸排报告

**调研时间**: 2026-10-04
**调研人**: agent-e-researcher (subagent)
**目的**: 为 BC（行为克隆）训练寻找真实高手牌谱数据源，替代自规则 bot 数据以抬高模型上限。

---

## TL;DR — 推荐结论

| 推荐 | 源 | 一句话理由 |
|------|----|----------|
| **Top 1** | **NikkeTryHard/tenhou-to-mjai + HuggingFace `nikketryhard/tenhou-houou-mjai`** | 现成、已转 MJAI、含天凤凤凰 250 万局 + 雀魂玉/王座合计 ~46 GB（数百万局），CC BY 4.0 / Apache-2.0 工具链，**下载即用，无自建抓取成本**。 |
| **Top 2** | **1nfrared218/tenhou-mjai-riichi-training（HF）** | 在天凤 MJAI 之上**已生成 legal-action mask、expert action index、score_delta / rank 等监督信号**，保存为 `.npz` WebDataset 分片，BC 工程改造成本最低，可直接对齐 obs/act 接口。 |

不建议作为主力的：雀魂自建抓取（违反 ToS、需账号、协议为私有 protobuf，工程与法律风险高）；Kaggle 散数据集（规模小、质量参差）。

---

## 1. 天凤 Phoenix（鳳凰卓）牌谱

### 1.1 数据源本身
- **平台**: tenhou.net 凤凰桌（最高级房间，高手浓度最高，NAGA / Suphx 都用它做评测/训练）。
- **历史规模**: 官方下载通道 `https://tenhou.net/sc/raw/scrawYYYY.zip`（2009 年起），kanachan 的 README 提到"2009–2019 十一年累计约 1700 万 *巡目/局面*"。换算到"局（hanchan）"层面，社区整理后的统计约 **~250 万局**（见下）。
- **格式**: 原始为 `mjlog` XML（gzip）。需要 `mjlog2json` → `mjai-reviewer/convlog` 两级转换到 MJAI JSON。

### 1.2 现成数据发布（**重点推荐**）
GitHub 仓库 **NikkeTryHard/tenhou-to-mjai** + HuggingFace 数据集 **nikketryhard/tenhou-houou-mjai** 已经把整条 pipeline 跑完并公开。

**Tenhou 凤凰桌部分（2009–2026.01）**:

| Year | Size | Games | Year | Size | Games |
|------|------|-------|------|------|-------|
| 2009 | 32MB | ~8k   | 2018 | 760MB | ~195k |
| 2010 | 311MB | ~80k | 2019 | 751MB | ~190k |
| 2011 | 449MB | ~115k| 2020 | 958MB | ~245k |
| 2012 | 521MB | ~135k| 2021 | 820MB | ~210k |
| 2013 | 585MB | ~150k| 2022 | 1.1GB | ~280k |
| 2014 | 625MB | ~160k| 2023 | 1.2GB | ~310k |
| 2015 | 672MB | ~170k| 2024 | 1.3GB | ~335k |
| 2016 | 702MB | ~180k| 2025 | 861MB | ~198k |
| 2017 | 743MB | ~190k| 2026 | 60MB  | ~14k (Jan) |

- **合计**: 约 **12 GB**（gzip 压缩的 `.mjson`），**>250 万局** 四人半庄凤凰桌牌谱。
- **格式**: 每局一个 `.mjai.json.zst`（MJAI 协议 JSONL），按年打包成未压缩 `.tar`。
- **下载**: `hf download nikketryhard/tenhou-houou-mjai tenhou-houou-2024.tar` 即可单年拉取；仓库也提供 BitTorrent magnet（48 GB 全量，含雀魂）。
- **许可**: 工具 Apache-2.0；牌谱本身声明"第三方历史记录，仅作 research/archival 使用"，DATA_LICENSE 文件要求下游自查 Tenhou 条款。实质上是"灰色可用"——Tenhou 官方长期默许研究用途，但商业使用存在风险。
- **限制**: 仅 4 人麻雀、仅半庄（无东风/三人）。

### 1.3 自建抓取（备选）
- **工具**: 老的 `MahjongRepository/phoenix-logs` 已 archive，推荐后继 **`Apricot-S/houou-logs`**（MIT，活跃维护）。
- **流程**: `houou-logs fetch` 拿 ID → `houou-logs download` 拿 XML → `houou-logs export` 出 `.xml` 文件 → `mjlog2json` → `convlog` → `.mjson`。
- **重要约束**: **Tenhou 单会话/单线程限制**（2024-06 官方推文明确），需要慢速抓取，不适合"短时间补大全量"，适合作为 2026+ 增量补充。
- **成本**: 已有完整工具链，工程量 ≈ 1–2 人日；带宽和时间成本为主。

### 1.4 转换到 obs 格式预估
- MJAI JSONL 是事件流（`start_kyoku / tsumo / dahai / pon / chi / ankan / reach_accepted / hora / ryukyoku`），写一遍 replay 器即可在每个 `dahai / pon / chi / reach` 决策点切出 obs，把 `actor` 实际选择当作 label。
- 可参考开源实现：`NikkeTryHard/tenhou-to-mjai`（Rust）、`1nfrared218` 的 WebDataset 流水线（见 §3）、`Equim-chan/mjai-reviewer/libriichi`（Rust+Python，含完整状态机）。
- **成本**: 若直接消费 `1nfrared218` 已经切好的 `.npz`（含 legal-actions + expert_index），**~2–5 人日**对齐我们 obs schema；如果从原始 MJAI 自己 replay，**~1–2 人周**。

---

## 2. 雀魂（Mahjong Soul）牌谱

### 2.1 "6500 万局"传闻核实
- **来源**: `Cryolite/kanachan` README 原文（2021 年口径）：
  > "I have been crawling game records from Mahjong Soul since July 2020, and the amount of game records for 4-player Mahjong played in the Gold Room (kin-no-ma) or the higher rooms has reached **about 65 million rounds as of the end of August 2021**. This number will surely surpass 100 million rounds by the end of 2021."
- **结论**: **是 Cryolite 个人爬取的私有数据集，不是公开数据集**。截至 2026 年没有证据表明这批数据被发布。kanachan 仓库明确写："this repository does not provide any crawler … any training data, nor any trained models."
- **传闻偏差**: 中文社区常说的"雀魂有 6500 万局公开数据集"是对 kanachan README 的误读。

### 2.2 实际可获取的雀魂数据

#### 2.2.1 NikkeTryHard 雀魂 MJAI 数据集（**重点推荐，与 1.2 同 repo**）
HuggingFace `nikketryhard/tenhou-houou-mjai` 同时包含雀魂部分：

| Shard | Size | Shard | Size |
|-------|------|-------|------|
| majsoul-jade-2019   | 92M  | majsoul-throne-2019 | 53M  |
| majsoul-jade-2020   | 374M | majsoul-throne-2020 | 144M |
| majsoul-jade-2021   | ~1.0G| majsoul-throne-2021 | 281M |
| majsoul-jade-2022   | 2.8G | majsoul-throne-2022 | 1.1G |
| majsoul-jade-2023   | 4.8G | majsoul-throne-2023 | 1.7G |
| majsoul-jade-2024   | 6.2G | majsoul-throne-2024 | 2.0G |
| majsoul-jade-2025   | 6.2G | majsoul-throne-2025 | 1.5G |

- **覆盖**: **玉の間（Jade）+ 王座の間（Throne）**，即雀魂段位场高端房间，2019–2025。
- **格式**: 同 Tenhou 部分，`.mjai.json.zst`，可直接共用解析器。
- **估算局数**: 按 MJAI 每局 ~50 KB（zstd 压缩）估算，Jade 2024 一年 6.2 GB ≈ **12 万+ 局/年**；合计粗估 **数十万至上百万局** 量级，**远小于传闻的 6500 万**，但已足够训练 BC（Suphx 用的天凤凤凰量级也就是数百万局）。
- **风险**: 雀魂（Yostar / 猫粮工作室）对抓取和再分发**明确比 Tenhou 更严格**。该仓库自述工具是 "multi-account Majsoul downloader"，本质上是违反 ToS 的爬取产物。**商业使用风险显著高于天凤**；学术研究/内部训练一般不踩线，但需团队法务确认。

#### 2.2.2 amae-koromo（雀魂牌谱屋）
- 网址 `https://amae-koromo.sapk.ch/`，作者 SAPikachu，第三方维护，从 2019-11-29（Jade/Throne 南场从 2019-08-23）开始持续收集金之间及以上对局。
- **不是直接的数据集发布**，而是个站点 + 内部 DB。仓库 `SAPikachu/amae-koromo-scripts` 只挂了 MIT license 文本，**脚本本身已不再公开发布**（先前版本曾公开）。
- 数据可通过站点 UI 查询单局，但**无官方批量导出 API**；要通过它拿到百万局级数据需要逆向其内部接口，工程/道德风险都高。
- 适合作为：**单局查询/抽样校验/玩家段位画像**补充，而非主数据源。

#### 2.2.3 自建抓取（不推荐）
kanachan README 给出的官方做法是：抓 `https://game.mahjongsoul.com/?paipu=...` 的 WebSocket 响应（私有 protobuf 格式），配合 mitmproxy / Wireshark / 浏览器扩展。问题：
- 需要雀魂账号 + 进入对应房间才能拿到 paipu uuid；
- 高端房间（王座）需要对局实际发生才能目击；
- 协议未公开，protobuf schema 需逆向（有 `tensoul-py-ng` 等社区项目可借鉴）；
- **明确违反雀魂 ToS**，账号有封禁风险。

### 2.3 转换成本
与天凤 MJAI 完全相同（都是 MJAI 格式）。如果统一用 MJAI 作为中间表示，雀魂部分**不需要额外转换代码**。

---

## 3. kanachan / Mortal / 其他开源 Riichi 项目用什么数据

### 3.1 Mortal（Equim-chan）
- 代码 AGPLv3，模型权重**未发布**（作者在 gist 里明确说"目前无计划公开模型参数"，担心被外挂滥用）。
- 训练数据：作者未公开具体清单，但 GitHub `Equim-chan/mjai-reviewer` 和 `libriichi` 是为天凤 XML 设计的；社区公认 Mortal 是**纯 RL 自我对弈**为主 + 用天凤牌谱做评测/可能的 SL 初始化，**作者本人没有放出训练数据集**。
- **可复用产物**: `libriichi` 的状态机 + `mjai-reviewer` 的 `convlog` 工具（XML→JSON→MJAI），是处理天凤数据的事实标准。
- 在线服务 `mjai.ekyu.moe` 提供 Mortal/Akochan 检讨，可作为 BC 模型的"老师评分"参考。

### 3.2 kanachan（Cryolite）
- 见 §2.1。**框架开源，数据不开源**。
- 自定义一套 annotation 格式（C++ 工具链，TFRecord），**与 MJAI 不直接兼容**；作者本人也在 discussion #71 里讨论要不要做 mjai ↔ kanachan 双向转换。
- 如果将来真要合作，路径是 MJAI → kanachan annotation。但短期内**不建议**作为我们 BC 的数据起点。

### 3.3 其它值得一提
- **Suphx (MSRA)**: 用天凤数据训练 + RL，但数据集本身未发布。
- **NAGA (ニワトコ/ドワンゴ)**: 商业产品，数据不公开。
- **Akochan (critter-mj)**: 规则+搜索，无训练数据。
- **Meowjong（三人麻将）**: arXiv 2202.12847，"50,000 rounds sampled from 2019 Houou table"，仅论文样本量，**未发布数据集**。
- **csci527-phoenix 项目**（USC 课程项目）: 论文 PDF 提到用 TFRecord 存训练数据，仓库未明显发布原始数据。

### 3.4 关键观察
**"开源 Riichi AI"和"开源 Riichi 数据集"是两个独立的事**。Mortal、kanachan 都没有把训练数据作为发布物。真正把数据当第一公民发布的是 **NikkeTryHard**（原始 MJAI 仓库）和 **1nfrared218**（加工后 WebDataset），这两个是 2024–2026 新出现的关键基础设施。

---

## 4. Kaggle / HuggingFace / 学术数据集

### 4.1 HuggingFace（按相关性）

| 数据集 | 规模 | 格式 | 备注 |
|--------|------|------|------|
| **`nikketryhard/tenhou-houou-mjai`** | 天凤 250 万局 + 雀魂 ~46 GB | MJAI JSONL (.zst in .tar) | **见 §1.2 / §2.2.1，Top1** |
| **`1nfrared218/tenhou-mjai-riichi-training`** | 基于上者，按年分 chunk | WebDataset `.tar` 内 `.npz` | **Top2**，已切好 legal_actions / expert_index / score_delta / rank，**最接近 BC 训练格式** |
| `mitsutani/mahjonglm-dataset` | 基于天凤 | token 序列 | 为 LLM 训练设计，**不适合直接做 BC** |
| `pjura/mahjong_board_states` | 未明确量级 | board state 快照 | 偏 hand evaluation |
| `sileixu/mahjong-winning-tiles` | benchmark 规模 | 和牌型 | 评测用，不适合训练 |

### 4.2 Kaggle
- `hphphp123321/tenhou-4-player-riichi-mahjong-dataset`：天凤 4 人麻将，**规模与 NikkeTryHard 重叠**，但更新慢。
- `trongdt` "Japanese Mahjong Board States"：手牌评估用，规模小。
- 其余多为 CV 麻将牌图片数据集，与对局无关。
- **结论**: Kaggle 上没有超过 HuggingFace 的现成 Riichi 对局数据集。

### 4.3 学术论文附带
- Suphx (arXiv 2003.13590)、Meowjong (2202.12847)、Mahjax (arXiv 2605.20577) 等论文均**未公开原始对局数据**。
- 学术界通行做法仍是"自己从天凤下载"，所以**最终都会回到 §1**。

---

## 5. 综合对比表

| 维度 | 天凤 (houou-logs 自抓) | 天凤 (NikkeTryHard HF) | 雀魂 (NikkeTryHard HF) | 雀魂 (自建) | kanachan 65M | Kaggle 散件 |
|------|----------------------|----------------------|----------------------|------------|-------------|------------|
| **规模** | ~250 万局 (2009–2026) | 同左 | 数十万–上百万局（玉+王座） | 理论无限 | 私有不可得 | 几千–几万 |
| **格式** | mjlog XML → 需转 MJAI | 已转 MJAI .json.zst | 同左 | 私有 protobuf | 私有 TFRecord | 杂 |
| **下载** | tenhou.net 单线程限速 | HF 直接拉取（快） | HF 直接拉取（快） | 需账号 + 抓包 | 无 | Kaggle API |
| **法律风险** | 低-中（Tenhou 默许研究） | 低-中 | **中-高**（雀魂明确禁止） | **高**（违反 ToS） | 不可得 | 低 |
| **转 obs 成本** | 中（需 replay） | 低 | 低 | 高 | N/A | 高（不统一） |
| **是否含高手** | ✅ 凤凰桌（最高段） | ✅ | ✅ 玉+王座（>=金之间） | 取决于房间 | ✅ | 不一定 |

---

## 6. 推荐方案（落地路径）

### 阶段 A（0–1 周，**强烈推荐**）
1. 从 HF 拉 `nikketryhard/tenhou-houou-mjai` 的 Tenhou 部分（**先 2023+2024 两年 ≈ 65 万局**，~2.5 GB）+ 雀魂 Jade/Throne 2024（~8 GB）。
2. 同时拉 `1nfrared218/tenhou-mjai-riichi-training` 的 WebDataset 分片，作为我们 obs/act 接口设计的**参考 schema**。
3. 写一个 `mjai_events → our_obs` 的转换器（Rust 参考 `libriichi`，Python 参考 `tenhou-to-mjai/dataset/mjai-validator`）。

### 阶段 B（1–4 周）
1. 跑通小规模 BC（10 万局级），对比我们自规则 bot 基线。
2. 若天花板显著提升，再决定是否：
   - 增量补 2025–2026 年天凤（用 `houou-logs` 慢速抓）；
   - 是否启用雀魂数据（**先法务评估**）。

### 阶段 C（可选）
- 用 Mortal 在线 `mjai.ekyu.moe` 给我们的 BC 模型打分做评测；
- 关注 `Cryolite/kanachan` 是否会发布任何数据子集；
- 关注 HuggingFace 上 `nikketryhard` 是否继续更新 2026 年雀魂分片。

---

## 7. 风险与不确定性

1. **法律**: 天凤数据"灰色可用"——Tenhou 官方从未对研究/AI 训练用途采取过行动，但严格来说批量抓取违反其 robots 精神；雀魂数据**违反 Yostar ToS 的证据明确**，商用风险显著。**建议内部训练可用，对外发布模型/数据集前咨询法务**。
2. **数据偏差**: 凤凰桌高手≠绝对理性，BC 会学到人类的常见失误模式（如过度副露、不合理的立直判断）。Suphx 论文给出的解法是 BC 之后接 RL 自我对弈；我们也应预留这个 pipeline。
3. **未验证**: NikkeTryHard 雀魂分片的"玉之间+王座"具体局数未在 README 中精确给出（只有 GB），上述"数十万–上百万局"是按 MJAI 平均 ~50 KB/zstd 估算得出，**实际局数需下载后统计**。
4. **时效**: 本报告结论基于 2026-10-04 抓取的公开页面；HF 数据集页面与 magnet 链接可能随时间变化。

---

## 引用链接（按文中出现顺序）

- https://github.com/MahjongRepository/phoenix-logs
- https://github.com/Apricot-S/houou-logs
- https://github.com/NikkeTryHard/tenhou-to-mjai
- https://huggingface.co/datasets/nikketryhard/tenhou-houou-mjai
- https://huggingface.co/datasets/1nfrared218/tenhou-mjai-riichi-training
- https://github.com/Equim-chan/Mortal
- https://mortal.ekyu.moe/
- https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- https://github.com/Equim-chan/mjai-reviewer
- https://github.com/Cryolite/kanachan
- https://github.com/Cryolite/kanachan/discussions/71
- https://amae-koromo.sapk.ch/
- https://github.com/SAPikachu/amae-koromo-scripts
- https://github.com/tsubakisakura/mjlog2json
- https://mjai.ekyu.moe/
- https://arxiv.org/abs/2003.13590 (Suphx)
- https://arxiv.org/abs/2202.12847 (Meowjong)

