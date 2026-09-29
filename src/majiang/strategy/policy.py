"""启发式决策（tasks.md 分组 5）。

目标函数是**本阶段总得分的期望**而非胡牌率（design.md D10）。平台三条特性决定了这套
启发式的形状：

- **只能自摸**：没有点炮，所以不存在传统「安全牌」，唯一防守手段是打财神触发抓打圈。
- **番型连乘**：收益集中在稀有高倍率局，动作链（杠/飘）的价值高于牌型微调。
- **庄闲 ×8 不对称**：庄家自摸收三家各 ×8；闲家自摸时庄家付 ×8、另两闲各付 ×1。

热路径只做负担得起的计算：出牌排序用 ``shanten``（一次决策 14 个候选合计约 11 ms），
同向听之间用 ``quick_blocks``（约 2.4 µs）做次排序。精确的 ``ukeire`` 单次约 0.1–0.2
秒，仅用于离线分析——该量级已在任务 2.15 的基准里记录。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from majiang.rules import score as score_module
from majiang.rules import shanten as shanten_module
from majiang.rules import tiles, win
from majiang.rules.action import CHI, DISCARD, GANG, HU, PASS, PENG, Action, chi_combinations
from majiang.rules.fan import FanResult, compute_fan
from majiang.rules.melds import CHI as MELD_KIND_CHI
from majiang.rules.melds import PENG as MELD_KIND_PENG
from majiang.rules.melds import Meld
from majiang.rules.situation import PHASE_DRAW, RESPONSE_PHASES, Situation
from majiang.rules.tiles import GOD

from . import risk, routes

OPPONENT_FAN_PRIOR = 2.0
ANGANG = "angang"
# 会改变打法、且**必须**在 configure() 里保留下来的开关。新增实验档位时记得加进来，
# 否则真机上会静默跑成默认档（踩过一次，见 configure 的说明）。
VARIANT_FIELDS = (
    "meld_tolerance",
    "tiebreak",
    "safe_tiebreak",
    "unified_score",
    "feed_ready_increment",
    "win_table_correction",
    "ukeire_candidates",
    "ukeire_max_shanten",
    "ukeire_order",
    "wait_aware_tenpai",
    "two_ply_shanten1",
    "shape_value",
    "feed_visibility",
    "dealer_feed_scale",
    "chase_baotou",
    "route_aware",
    "preserve_god",
    "natural_route",
    "pair_route_pairs",
    "shanten_weight",
    "feed_weight",
    "god_discard_penalty",
    "value_weight",
    "piao_threshold_scale",
)
# 精确进张只在这些条件下启用：向听越低越接近胜负、且候选越多越值得算
# 实测开销（60 局自对弈）：heuristic 176 ms/局、廉价 ukeire 400 ms/局、精确 2510 ms/局。
# 收紧到「打完仍是 1 向听 + 并列候选前 2 张」后降到可接受范围——1 向听正是
# 「中段落后」出现的地方（见 tools/analyze_hand_progress.py 的 n=7~8 曲线）。
EXACT_UKEIRE_MAX_SHANTEN = 1
EXACT_UKEIRE_DEFAULT_CANDIDATES = 2
EXACT_UKEIRE_BUDGET_SEC = 0.6


class Mode(StrEnum):
    """策略模式（tasks.md 5.13）。"""

    QUALIFIER = "qualifier"
    FINAL = "final"


class Commitment(StrEnum):
    """路线承诺：只用于标定「某一条路线单独打时」的真实成牌率。

    共用一张「向听 → 胜率」表是 5.6 首版失败的原因：七对不能吃碰，同向听的实际成牌率
    低于一般形，共用表会系统性高估七对。把两条路线各自**单独打**一批对局，就能得到
    各自无混淆的实测表。
    """

    NONE = "none"
    PAIR = "pair"
    MELD = "meld"


class MeldTolerance(StrEnum):
    """吃碰闸门的松紧（tasks.md 5.5）。

    **由真实对局统计决定的档位**（1129 份事件流 / 8969 局，见 `tools/meld_census.py`
    与 `tools/analyze_meld_gate.py`）：

    - 我们每局**持有副露 0.591 次**，对手 **1.093 次**（1.85 倍）
    - 2372 次碰/吃机会中，闸门②「向听必须严格下降」拒绝了 1504 次（63%），
      其中**向听不变**的有 844 次（36%），且这些里面只有 132+410 次发生在 0 向听
    - 对手的副露率恰好落在「几乎接受全部『向听不变且未听牌』的机会」那一档上

    0 向听（已听牌）时副露只会换掉听口，不会让你更接近胡牌，因此放宽档**仍拒绝**它。
    """

    STRICT = "strict"
    EQUAL = "equal"
    # 更保守的放宽：只在**离听牌还远**（向听 ≥2）时接受向听不变的副露。
    # 依据是 agent B 的全量进度曲线——差距从第 2 摸起就单调扩大（n=3 +3.0pp、
    # n=5 +7.4pp、n=8 +10.9pp），说明吃碰的价值在**早段**；而 1 向听已接近听牌，
    # 那时副露的边际收益低、还会牺牲听口质量。
    EQUAL_EARLY = "equal-early"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    mode: Mode = Mode.QUALIFIER
    base_score: int = 1
    you_cai_bi_kao: bool = False
    # 路线分叉开关（tasks.md 5.6）。**实测为负收益，故默认关闭**：
    # 960 局 A/B 中，开启路线分叉的座位胡率 9.7%、总得分 −2185，而关闭的座位
    # 胡率 28.6–29.3%、总得分 +546~+841（均番仅从 1.1 升到 1.3，不抵胜率损失）。
    #
    # 根因已定位：`routes.WIN_RATE_BY_SHANTEN` 是**路线无关**的实测表，而它的样本
    # 来自旧策略的对局——那些手牌可以靠吃碰推进。七对路线无法吃碰，同样是「4 向听」，
    # 七对的实际到听概率显著低于一般形。于是估值模型系统性高估七对路线
    # （p_win(4)×番2 ≈ 6.9 > p_win(2)×番1 ≈ 5.6），策略就一路去做七对而放弃速度。
    #
    # 修法明确：按路线分别测「向听 → 到听/自摸」表（`tools/calibrate` 可扩展），
    # 而不是共用一张。在此之前保持关闭。
    route_aware: bool = False
    commitment: Commitment = Commitment.NONE
    # 吃碰闸门松紧（tasks.md 5.5）。默认 strict 是**改动前的行为**，放宽档需先过 A/B。
    meld_tolerance: MeldTolerance = MeldTolerance.STRICT
    # 精确进张要评估几张候选。**候选面本身是个缺陷来源**：
    # 候选按 ``total`` 排序，而同向听时 ``total`` 被喂牌代价主导（`-3*feed`，可达 9 分），
    # 于是「听口明显更好但喂牌稍多」的那张会在进入精确比较之前就被截掉。
    # 实测我们的听口窄于对手约 21%（听口张数 11.53 vs 14.66，
    # `tools/analyze_wait_quality.py`），与这个截断方向一致。
    ukeire_candidates: int = EXACT_UKEIRE_DEFAULT_CANDIDATES
    # 精确进张次排序适用的**最大向听**。默认 1 = 只在接近听牌时才比较进张，
    # 意味着 shanten >=2 的整个前中期，出牌由「骨架厚度 + 喂牌代价」决定、**完全不看进张**。
    # 而实测我们在第 4 摸时均向听就落后对手 0.16（同财神数下亦然，
    # `tools/analyze_god_usage.py`）——那段差距正是由前中期出牌决定的。
    ukeire_max_shanten: int = EXACT_UKEIRE_MAX_SHANTEN
    # 并列候选**按什么排序后再截断**。默认 `total`，而同向听时 total 被喂牌代价主导
    # （-3*feed 可达 9 分），于是「听口明显更好但喂牌稍多」的那张会在进入精确比较前
    # 就被截掉。改成 `blocks`（骨架厚度，微秒级）可让候选面**按手牌质量**取，
    # 再看进张——零成本地换掉那个被喂牌污染的入口顺序。
    ukeire_order: str = "total"
    # 听牌（向听 0）时改用**可见听口张数**比较候选，并且**不截断候选面**。
    #
    # **为什么需要它（这是一个静默退化的 bug 级发现）**：`shanten.ukeire` 在
    # `current == 0` 时**直接返回空元组**（向听不能再降，`shanten.py:296`）。而调用方
    # `_break_ties_by_ukeire` 把「每个候选 copies 都是 0」当成平局，于是
    # `best` 停在 `total` 排序的第一名上——**听牌时那个「精确进张次排序」什么都没做**，
    # 出牌完全由 `total = -10×向听 + 骨架厚度 - 3×喂牌 - 财神罚` 决定，**根本不看听口**。
    # 这也解释了为什么 `ukeire-wide`（候选面 2→6）测出来是平的：候选数在听牌时无关紧要。
    #
    # 实测依据（`tools/analyze_wait_ceiling.py`，v2 时代 150 文件 / 1298 个听牌出牌点）：
    # 我们的听口可见张数离同一手牌的上限平均差 **0.96 张（6.7%，分位 27.6%）**，
    # 而对手只差 **0.20 张（1.1%，分位 5.2%）**——对手几乎总选最宽的听口。
    #
    # 只在 shanten 0 生效；向听 1 以上仍走 `ukeire`（那里它是有效的）。默认关闭，
    # 因为打开它就是改冠军档（v2 的行为必须逐位可复现）。
    #
    # 成本实测：`winning_draws` **8.8 ms/次**，听牌时并列候选约 2.8 张 ⇒ 整次决策
    # 15.5 → 23.9 ms（+8.4 ms）。对比出牌预算 1800 ms、平台 DiscardTimeoutSec=3 s，
    # 余量充裕，所以这条路**不需要**墙钟上限（`ukeire` 那条 0.1–0.2 s/次才需要）。
    wait_aware_tenpai: bool = False
    # 向听 1 时改用**两拍值**排序（见 `_two_ply_value`）。
    #
    # **与 `wait_aware_tenpai` 是同一条思路的两层**：那个修的是「听牌那一刻选哪个听口」
    # （实测 regret 6.7% → 0，已由 v3 上线）；这个想修的是**它的上游**——
    # 向听 1 选哪张牌，决定了你以后能拿到多宽的听口。
    #
    # 实测依据（真机 1 向听出牌点）：**39.2% 的局面我们的选择不是两拍最优，
    # 相对 regret 均值 11.6%**（n=74）。注意这个量**有预测力**：按两拍值分三分位，
    # 本局胡牌率 14.3% / 45.0% / 68.4%，后续实际听口宽度 8.14 / 7.80 / 12.08。
    #
    # 成本实测约 **150 ms/候选**（共享 memo；`ukeire` 热跑 33.6 ms、冷跑 120 ms），
    # 向听 1 并列候选均约 3 张 ⇒ 约 450 ms/决策，仍在 1800 ms 预算内但会顶到 0.6 秒墙钟上限。
    # **默认关闭**：11.6% 落在「10–25% 灰区」，所以先做成档位，
    # 用一个**非循环论证**的机制量（自对弈里实际拿到的听口宽度）判它该不该进 A/B。
    two_ply_shanten1: bool = False
    # 同向听的次排序改用**加权形质值**（`shanten.shape_value`）替代 `2×面子 + 搭子`。
    #
    # **依据（今晚实测，这是当前最大的一个缺口）**：`quick_blocks` 的
    # `partials = min(partials, 4 - sets)` 恒饱和，导致同向听候选之间取值高度集中——
    # 真机 3493 个决策点上 **77.2% 的决策里所有候选的 `2×面子+搭子` 完全相同**
    # （副露 1 组 89.2%、2 组 93.5%）。次排序一旦并列，`_choose_discard` 就只剩
    # 「喂牌」一项在起作用，**中段等于完全没有形质概念**。
    # 这也解释了 `ukeire-wide/hand/early` 全族为何测平：候选池是按这个退化键排序的，
    # 加宽池子等于随机采样。
    #
    # 而对手侧的行为指纹显示缺口正在中段：强 bot **首次到听摸序 4.96~5.48、序号 10 到听率 75~80%**，
    # 我们是 **6.23 / 54.2%**（副露 0.62 vs 他们 1.20~1.30，听口宽 11.56 vs 12.2~12.7）。
    #
    # 设计成窄改动：形质修正幅度 < 1，**只打破并列、不覆盖「块数差 1」**。
    # 默认关闭（v1/v2/v3 行为逐位不变）。
    shape_value: bool = False
    # 喂牌项乘上「该牌种还剩几张未现」因子 `(4 − 已见)/4`。
    #
    # **依据（agent-c 独立复核确证的一条缺陷）**：`risk.visible_need(tile)` 是**牌种静态表**
    # （字牌 0.4 / 中张 1.0 / 边张 0.6），**不含已见张数**——于是**已见 3 张、几乎喂不出去的牌
    # 仍按满值计罚**，而 4 张全现的牌（不可能喂）也一样罚。正确口径应是
    # 「还剩几张可能被对手拿到」：`seen` 含本方暗手 + 四家副露 + 四家弃牌，
    # `4 − seen` 正是仍藏在别处（对手手牌或牌墙）的张数。
    #
    # 这条与「喂牌项被放大 1.5 倍」（`tools/calibrate_threat.py`，按房 30 房实测 1.49×）
    # 是**两个独立缺陷**：那条是 `threat` 水平偏高，这条是**逐牌种的分辨力**缺失。
    # 默认关闭（v1/v2/v3 行为逐位不变）。
    feed_visibility: bool = False
    # 同向听候选项之间的次排序键。**默认 "exact-ukeire"**。
    #
    # 历史：原默认是 "blocks"（骨架厚度）。5.4 曾试过 "ukeire" 并记为「无增益」，
    # 但那个档位用的是 `cheap_ukeire`（quick_shanten 贪心近似），实测它与精确口径
    # **只在 14.7% 的手牌上选出同一张打牌**、不一致时平均放弃 15.4 张精确进张
    # （tools/analyze_ukeire_fidelity.py），所以那次 A/B 测的是噪声。
    #
    # 改用精确进张后**两个独立种子、共 2000 配对场**一致显著为正（合并估计）：
    #   名次分 +0.285/场（t 4.03）· 胡次数 +0.088/场（t 4.46）· 总得分 +1.525/场（t 2.55）
    #   白板数不受损（+0.014，不显著）；胡率 25.6% vs 24.5%
    # 故切为默认。旧行为保留在 `--decider blocks` 供后续对照。
    tiebreak: str = "exact-ukeire"
    # 实验档位：**只在完全打平**的候选中，优先打「已见张最多」的那张。
    #
    # **为什么需要它**：`_choose_discard` 的最终兜底是 `sorted(..., reverse=True)` 的稳定性，
    # 于是同分时永远选**最小牌索引**——万 0~8 / 筒 9~17 / 条 18~26 / 字 27~33，即系统性地
    # 偏向打万。B' 独立量到这条偏置的分布（万 0.54 / 筒 0.28 / 字 0.22 / 条 0.08），
    # 而我们的弃牌结构与强 bot 差得最大的一列正是中张占比（我们 19.6% vs 对手 29.7%，
    # 逐房配对差 −11.2pp ± 0.4）。**这个偏置是排序实现的副作用，不是任何设计决定**。
    #
    # 换成「同分优先打已见张最多的那张」，依据是**安全性有确定方向**：一张已经被别人打掉
    # 多张的牌，别人持有的同牌更少 ⇒ 更不可能是他等的那张。它同时天然抹掉花色索引偏置
    # （已见张数与花色无关）。
    #
    # 与已被判死的 `feed_visibility` 的区别（**这条很重要**）：那个把 `(4−seen)/4` 乘进
    # **喂牌项本身**，于是能压过形质、改变 total 的排序（实测总得分/名次分 2/2 反向）；
    # 这里**只在 total 完全相等的候选之间**起作用，结构上不可能覆盖任何一项，改动面严格更小。
    safe_tiebreak: bool = False
    # 实验档位：**统一期望得分**（B' 2026-09-29 的设计说明）。
    #
    # 把出牌评分从启发式 `total = −10×向听 + 形质 − 3×喂牌 − 财神罚` 换成
    # **「打完这张牌后本局期望得分」的显式估计**，所有候选在**同一量纲（分）**上可比：
    #
    #     score(tile) = P_win × E_pay − (1 − P_win) × P_opp × E_loss − feed_cost
    #     feed_cost   = ΔP_opp(tile) × E_loss
    #
    # **为什么**：现在的 `total` 有两处实测硬伤——① 量纲不可比：同向听组内 total 极差
    # **97.25% 来自喂牌项**、形质项贡献均值仅 **0.026**（B' 14:05，三路独立复核），
    # 于是「−10×向听」这个拍出来的 10 和喂牌项谁大谁说了算，形质形同虚设；
    # ② 并列靠隐式次序：**59.5%** 决策的可见前 4 名 total 完全相同。
    # 换成分的量纲后，喂牌从「旋钮」变成**推导量**（`ΔP_opp × E_loss`），
    # `feed_weight` 这个旋钮被消灭——此前 5 个 feed 系臂全是调这个旋钮，全测平或反向。
    #
    # **刻意留作第二组件的部分（不在本开关里）**：B' 设计里 `P_win` 的**形质修正**
    # （「同向听、形质更好 ⇒ P_win 更高」）。它的定标依据是 `shape-blocks` 的 +0.350(t3.19)，
    # 而那条臂的 **n=10 确认批此刻还在跑** —— 用未结的证据定标，等于把「看着有戏就采信」
    # 再犯一次（今天 `seen-tiebreak` ≈0 已经教过一次）。所以先落地三因子结构，
    # 形质修正等 n=10 落地后再作为**独立开关**加进来，两次改动各自可归因。
    unified_score: bool = False
    # 喂牌增量：一张「对对手最有吸引力」的牌（中张）被吃碰后，对手**听牌概率的绝对增量**。
    #
    # 0.05 的来历（刻意写清楚，免得以后被当成标定值引用）：它不来自拟合，而是**量级对齐**——
    # 旧的 `feed_weight=3.0` 在典型场面下给 `3.0 × 0.6 × 0.6 ≈ 1.1 分` 的喂牌代价，
    # 而这里 `ΔP_opp × E_loss ≈ 0.05 × 20 ≈ 1.0 分`。对齐量级是为了让这条臂**只改结构、
    # 不叠加量级跳变**，否则一次 A/B 里两个变化混在一起无法归因。
    # 真正的取值应由真机数据标定（`ΔP_opp` 是可测的：见 `tools/verify_occupancy_ceiling.py` 的同族口径）。
    feed_ready_increment: float = 0.05
    # `P_win` 表的**按向听乘性修正**（`None` = 不修正）。索引 = 向听，值 = 乘到
    # `routes.win_probability` 输出上的因子。
    #
    # **来历（`tools/calibrate_win_table.py` 的实测，25 房 / 17167 个决策点）**：
    # `WIN_RATE_MELD` 那两张表是**自对弈标定**的（对手＝我们自己），而真机实测**门 2 不通过**——
    # 斜率（实际~预测）**1.694**（判据 ∈[0.8,1.2]）、整体高估 5.5 个百分点，且**不是均匀偏高**：
    #   听牌态（s=0）实际 0.339 vs 预测 0.308 ⇒ 轻微**低估**（+0.031）
    #   s=2/3/4 实际 0.140/0.111/0.098 vs 预测 0.193/0.193/0.188 ⇒ **高估 5~9pp（t −6~−10）**
    # 病灶是「表太扁平 + `draws_left` 缩放过头」：表从 s=0 到 s≥2 只差 2.3×，实测差 2.9~3.5×；
    # 而 `scale = clip(draws_left/45, 0.40, 1.35)` 把**早局的高向听手**抬到 1.35×，
    # 那正是 43% 决策发生的地方。
    #
    # 下表就是实测的 `实际/预测` 比值（s=6 无样本，沿用 s=5）。
    # **只用在我方出牌这一支**（见 `_unified_route_value`），不碰 v3 的听牌机制，
    # 也不碰碰吃闸门——那两处仍走未修正的表，避免让冠军档静默漂移。
    win_table_correction: tuple[float, ...] | None = None
    # 实验档位：绝不打出财神（只在无其他可打牌时才打）。
    # 用途是验证一条尚未测过的假设——爆头需要「4 组**自然**面子 + 1 张闲余财神」，
    # 而此前的 0 次爆头是被动观测到的（现有策略会把财神当百搭用掉）。若把财神硬留，
    # 看能不能把链路制造出来。
    preserve_god: bool = False
    # 实验档位：按「纯真牌」算向听（财神只当将牌兜底），逼策略去做 4 组自然面子，
    # 从而检验「爆头可以被制造出来」这条假设。
    natural_route: bool = False
    shanten_weight: float = 10.0
    pair_route_pairs: int = 5
    feed_weight: float = 3.0
    # 庄家局对「喂牌代价」的额外缩放（tasks.md 5.3）。
    #
    # 动机来自实测（`tools/analyze_dealer.py`，300 文件）：我们作为庄家的胜率 27.07%
    # 而对手 30.21%，**庄家局每局净分差 −1.59，是闲家局差（−0.90）的 1.8 倍**——
    # 因为庄家的赔付是 ×8。而对手把「庄家结构性优势」兑现成 +5.3 点（30.21% vs 公平 24.9%），
    # 我们只兑现 +2.2 点。
    #
    # 根因是**出牌与响应决策里没有任何庄闲项**（`_score_discard` 只有向听/骨架/喂牌/财神），
    # 而赔付方向是 8 倍不对称。这个旋钮是最小侵入的试验口子：只缩放庄家局的喂牌权重。
    # 1.0 = 现状（不分庄闲）。
    dealer_feed_scale: float = 1.0
    god_discard_penalty: float = 25.0
    value_weight: float = 10.0
    piao_threshold_scale: float = 1.0
    # 在能胡时主动弃胡、把手牌打成爆头（听任意）——通往财飘链的唯一入口。
    # 关掉它是为了 A/B 量化这条决策的期望值。
    chase_baotou: bool = True

    @classmethod
    def for_mode(cls, mode: Mode, **overrides: Any) -> PolicyConfig:
        """晋级轮稳健、决赛激进：决赛以总得分为唯一目标，因此更愿意追高番。"""
        if mode is Mode.FINAL:
            # 用 setdefault 而不是直接传：原来写成 `cls(mode=mode, feed_weight=1.5, **overrides)`，
            # 只要 overrides 里也带 feed_weight（例如 `--decider feed-low`）就会
            # **TypeError: got multiple values for keyword argument**，
            # 而且只在 final 模式启动时才炸——比赛当天才会暴露。
            overrides.setdefault("piao_threshold_scale", 0.85)
            overrides.setdefault("feed_weight", 1.5)
            return cls(mode=mode, **overrides)
        return cls(mode=mode, **overrides)


# 变体命名的比较基准。注意它与 `for_mode` 无关：命名的目的是让日志能区分档位，
# 而不是判断「这个档位该不该叫变体」。
_DEFAULT_POLICY = PolicyConfig()


@dataclass(frozen=True, slots=True)
class DiscardScore:
    tile: int
    shanten: int
    blocks: int
    route: str
    pair_value: float
    meld_value: float
    route_value: float
    feed_cost: float
    god_penalty: float
    total: float

    def describe(self) -> str:
        return (
            f"{tiles.to_code(self.tile)} 向听={self.shanten} 路线={self.route} "
            f"七对值={self.pair_value:.1f} 副露值={self.meld_value:.1f} "
            f"喂牌={self.feed_cost:.1f} 财神={self.god_penalty:.0f} 合计={self.total:.2f}"
        )


def _shanten_or_none(counts: Sequence[int], meld_count: int) -> int | None:
    try:
        return shanten_module.shanten_any(counts, meld_count)
    except shanten_module.ShantenError:
        return None


def _best_after_drop(counts: list[int], meld_count: int) -> int | None:
    """兼容旧调用点：``shanten_any`` 已统一处理「恰好／多一张」两种张数。"""
    return _shanten_or_none(counts, meld_count)


def _current_shanten(situation: Situation) -> int | None:
    """本人当前可达的最小向听。

    摸牌阶段手牌是 14 张，而 ``shanten`` 只接受未摸牌的 13 张，因此必须走
    「打出一张后取最小」这条路径——直接用 ``shanten`` 会抛错并被当成「无向听信息」，
    结果整个杠决策被静默跳过。
    """
    return _shanten_or_none(situation.hand.counts, situation.hand.meld_count)


def cheap_ukeire(
    counts: Sequence[int], visible: Sequence[int] | None = None
) -> tuple[int, int]:
    """廉价进张估计，返回 ``(牌种数, 剩余张数)``。

    用 ``quick_shanten`` 代替精确向听，结构上与 :func:`majiang.rules.shanten.ukeire`
    同构（同样对每个可摸牌种取「打出任意一张后的最小」）。实测精确 ``ukeire`` 为
    42–157 ms、本函数 1.1–1.6 ms（约 40–100 倍），且两者量级一致。
    """
    base = _cheap_best_shanten(list(counts))
    kinds = copies = 0
    work = list(counts)
    for tile in range(tiles.TILE_KINDS):
        remaining = tiles.COPIES_PER_KIND - (visible[tile] if visible else work[tile])
        if remaining <= 0:
            continue
        work[tile] += 1
        if _cheap_best_shanten(work) < base:
            kinds += 1
            copies += remaining
        work[tile] -= 1
    return kinds, copies


def _two_ply_value(
    counts: Sequence[int],
    meld_count: int,
    visible: Sequence[int],
    memo: dict,
) -> float:
    """两拍值：``Σ(每个进张的剩余张数) × (摸到它之后、打完一张能拿到的最宽听口)``。

    **为什么是它**：向听 1 的决策只用一拍（``ukeire``能到听牌的张数），而``到的听口有多宽``
    要到下一拍才知道。实测这个量**有预测力**（真机 74 个向听 1 点按两拍值分三分位：
    本局胡牌率 14.3% / 45.0% / 68.4%，后续实际听口宽度 8.14 / 7.80 / 12.08）。
    而我们**同位置内的 regret 是 11.6%**（39.2% 的局面不是两拍最优）——
    这就是 `total` 在向听 1 那一层留下的可兑现空间。

    **成本**（决定它能不能上真机）：实测 **约 150 ms/候选**（共享 memo 下 `ukeire` 热跑 33.6 ms，
    冷跑 120 ms；memo 加速 3.6 倍）。向听 1 的并列候选平均约 3 张 ⇒ **约 450 ms/决策**，
    在出牌预算 1800 ms 内，但会顶到 0.6 秒墙钟上限——所以调用方**必须**保留超时回退。

    ``visible`` 是**按打出前**算的，摸到 ``tile`` 之后要把它自己也记为已见，否则会高估一张。
    """
    base = list(visible)
    total = 0.0
    for tile, left in shanten_module.ukeire(counts, meld_count, memo=memo):
        if left <= 0:
            continue
        drawn = list(counts)
        drawn[tile] += 1
        seen = list(base)
        seen[tile] += 1
        widest = 0
        for drop in range(tiles.TILE_KINDS):
            if drawn[drop] <= 0:
                continue
            after = list(drawn)
            after[drop] -= 1
            try:
                if shanten_module.shanten_any(after, meld_count, memo=memo) != 0:
                    continue
            except shanten_module.ShantenError:
                continue  # 张数不符的候选跳过，不当成 0 参与比较
            waits = win.winning_draws(after, meld_count)
            if not waits:
                continue
            widest = max(
                widest,
                sum(max(0, tiles.COPIES_PER_KIND - seen[wait]) for wait in waits),
            )
        total += left * widest
    return total


def _wait_copies(
    counts: Sequence[int], meld_count: int, visible: Sequence[int], discarded: int
) -> int | None:
    """打完 ``discarded`` 之后，**还剩几张可见的牌能直接完成胡牌**。

    这是听牌时唯一正确的候选比较口径。不能用 `shanten.ukeire`：它在 `current == 0`
    时返回空元组（向听不能再降），老调用方把「全是 0」当平局，于是听牌时的次排序
    静默失效、出牌退化成按 `total`（骨架厚度 + 喂牌）选——**完全不看听口**。

    ``visible`` 是按**打出前**的 14 张手牌算的，所以这里要减掉即将打出的那一张，
    否则该牌种会被少算一张可摸牌。

    张数不符时返回 ``None``（调用方会跳过该候选并计数），**不返回 0**——
    返回 0 等于把「算不出来」伪装成「这个候选的听口是空的」，那正是本项目里
    反复出现过的「非法输入变成看起来合理的结论」那一类错误。
    """
    try:
        waits = win.winning_draws(counts, meld_count)
    except ValueError:
        return None
    if not waits:
        return 0
    remaining = list(visible)
    if 0 <= discarded < len(remaining):
        remaining[discarded] = max(0, remaining[discarded] - 1)
    return sum(max(0, tiles.COPIES_PER_KIND - remaining[wait]) for wait in waits)


def _cheap_best_shanten(counts: list[int]) -> int:
    best = shanten_module.quick_shanten(counts)
    for tile in range(tiles.TILE_KINDS):
        if counts[tile] == 0:
            continue
        counts[tile] -= 1
        value = shanten_module.quick_shanten(counts)
        counts[tile] += 1
        if value < best:
            best = value
    return best


class HeuristicDecider:
    """启发式决策器，满足运行时依赖的 ``Decider`` 窄接口。"""

    # 实例会在 __init__ 里按 config 覆盖成 `heuristic[<变体>]`，见 _refresh_name()
    name = "heuristic"

    def __init__(self, config: PolicyConfig | None = None, *, risk_model: object | None = None) -> None:
        self.config = config or PolicyConfig()
        # 对手听牌模型（tasks.md 6B）：为 None 时 risk.assess 用手写启发式兜底
        self.risk_model = risk_model
        self.last_reason = ""
        self.last_detail: dict[str, Any] = {}
        self._refresh_name()

    def _risks(self, situation: Situation):
        """对手风险：注入了模型就用模型，否则用手写启发式（`risk.assess` 内部兜底）。"""
        if self.risk_model is None:
            return risk.assess(situation)
        return risk.assess(situation, model=self.risk_model)  # type: ignore[arg-type]

    def configure(self, tournament) -> None:
        """由运行时注入赛事配置：底分与「有财必拷响」必须取自服务端，不能写死。

        **必须用 ``replace`` 而不是重建 ``PolicyConfig``。** 原先的实现只把 6 个字段带上，
        把下面这些**静默丢弃**并回落到默认值：

        ``meld_tolerance`` / ``tiebreak`` / ``chase_baotou`` / ``commitment`` /
        ``preserve_god`` / ``natural_route`` / ``pair_route_pairs`` / ``shanten_weight``

        实测后果（严重）：``configure`` 只在**真机**路径被调用（`engine.py` 的
        `Runtime.run`），自对弈路径不调用它。于是真机上所有实验档位
        （no-chase / ukeire / meld-equal）**都跑成了默认档**，而自对弈跑的是真档位——
        表现为「自对弈有差异、真机无差异」，看起来像噪声，实际是档位根本没生效。
        """
        self.config = replace(
            self.config,
            base_score=int(tournament.base_score),
            you_cai_bi_kao=bool(tournament.you_cai_bi_kao),
        )
        self._refresh_name()

    def _refresh_name(self) -> None:
        """把与默认档不同的开关拼进 ``name``。

        ``name`` 是类属性且原本恒为 ``"heuristic"``，而日志记的是
        ``decision.made ... decider=<name>``，所以**所有变体在日志里长得一模一样**，
        事后无法从日志分辨某场会话跑的是哪个档位。
        """
        differences = [
            f"{field.replace('_', '-')}={getattr(self.config, field)}"
            for field in VARIANT_FIELDS
            if getattr(self.config, field) != getattr(_DEFAULT_POLICY, field)
        ]
        if self.config.commitment is not Commitment.NONE:
            differences.append(f"commitment={self.config.commitment.value}")
        self.name = "heuristic" if not differences else "heuristic[" + ",".join(differences) + "]"

    # ---- 入口 -------------------------------------------------------------

    def choose(
        self,
        situation: Situation,
        actions: Sequence[Action],
        *,
        budget_ms: int,
    ) -> Action | None:
        self.last_reason = ""
        self.last_detail = {}
        if not actions:
            return None
        if situation.phase in RESPONSE_PHASES:
            return self._choose_response(situation, actions)
        if situation.phase == PHASE_DRAW:
            return self._choose_turn(situation, actions)
        return actions[0]

    # ---- 本人回合 ---------------------------------------------------------

    def _choose_turn(self, situation: Situation, actions: Sequence[Action]) -> Action | None:
        if any(action.kind == HU for action in actions):
            chosen = self._choose_win_or_piao(situation, actions)
            if chosen is not None:
                return chosen
        gang = self._best_gang(situation, actions)
        if gang is not None:
            return gang
        return self._choose_discard(situation, actions)

    def _choose_win_or_piao(
        self, situation: Situation, actions: Sequence[Action]
    ) -> Action | None:
        """胡牌与弃胡飘的取舍（tasks.md 5.9 / 5.10）。

        设立即胡的得分为 ``V``、失败时承担的赔付为 ``L``、本圈无人自摸的生存概率为
        ``q``。飘一次使番翻倍，故比较 ``2V·q − L·(1−q)`` 与 ``V``，等价阈值
        ``q > (V + L) / (2V + L)``（``L = 0`` 时退化为 ``q > 0.5``）。
        """
        hu_action = next(action for action in actions if action.kind == HU)
        hand = situation.hand
        drawn = situation.drawn_tile
        if drawn is None:
            self.last_reason = "能胡但快照未给摸牌信息，直接胡"
            return hu_action

        waiting = list(hand.counts)
        waiting[drawn] -= 1
        current = self._fan_of(situation, waiting, drawn)
        if not current.hu:
            return hu_action

        if self._piao_candidate(situation, waiting, drawn) is None:
            # 尚未爆头：看看能否靠**弃胡**打成爆头（通往财飘链的唯一入口）
            target = self._baotou_by_discard(situation, waiting, drawn) if self.config.chase_baotou else None
            if target is None:
                self.last_reason = (
                    f"胡：{current.fan} 番（{'爆头' if current.baotou else '常规'}），无续飘机会"
                )
                return hu_action
            risks = self._risks(situation)
            survival = risk.lap_survival(risks)
            gain = float(self._win_points(current.fan * 2, situation))
            loss = float(self._loss_points(situation))
            threshold = self._piao_threshold(gain, loss)
            self.last_detail = {
                "fan_now": current.fan,
                "fan_if_baotou": current.fan * 2,
                "survival": round(survival, 3),
                "threshold": round(threshold, 3),
                "baotou_discard": tiles.to_code(target),
            }
            if survival > threshold and not situation.is_restricted:
                self.last_reason = (
                    f"弃胡求爆头：打 {tiles.to_code(target)} 做成听任意，"
                    f"番 {current.fan}→{current.fan * 2}，生存 {survival:.2f} > 阈值 {threshold:.2f}"
                )
                return Action(DISCARD, tile=target)
            self.last_reason = (
                f"胡牌：番 {current.fan}，弃胡求爆头不划算（生存 {survival:.2f} ≤ 阈值 {threshold:.2f}）"
            )
            return hu_action

        risks = self._risks(situation)
        survival = risk.lap_survival(risks)
        gain = float(self._win_points(current.fan, situation))
        loss = float(self._loss_points(situation))
        threshold = self._piao_threshold(gain, loss)
        self.last_detail = {
            "fan_now": current.fan,
            "survival": round(survival, 3),
            "threshold": round(threshold, 3),
            "gain": gain,
            "loss": loss,
            "opponent_risk": [round(r.self_draw_probability, 3) for r in risks],
        }
        # **规则合规守卫（不是策略参数）**：抓打圈内出牌只能打刚摸到的那张
        # （``rules/action.py:_turn_actions`` 用 ``tile != drawn`` 过滤候选），而这里是
        # **自建** `Action(DISCARD, tile=GOD)`、绕过了 ``legal_actions``，所以必须自己确认。
        # 缺这道守卫时会返回非法弃牌：实测（40 万副随机手牌筛出 6 例）例如
        # `6w6w8w8w1b1b4b4b7b7b3t3t白白`＋摸 `6w`，续飘成立、生存 0.98 > 阈值 0.54，
        # 而该局面的合法集只有 `[discard:6w, hu]` ⇒ 真机必被 409 INVALID_ACTION 拒掉、
        # 白费一个出牌窗口（B' 的 D1）。
        # 注意**不能**照抄 :562 那种一票否决：摸到的正好是财神时弃财神是合法的，
        # 而 :562 的目标牌是别人。
        can_piao = current.chain_count < 6 and (drawn == GOD or not situation.is_restricted)
        if survival > threshold and can_piao:
            self.last_reason = (
                f"弃胡飘：番 {current.fan}→{current.fan * 2}，生存 {survival:.2f} > 阈值 {threshold:.2f}"
            )
            return Action(DISCARD, tile=GOD)
        if survival > threshold and drawn != GOD and situation.is_restricted:
            # 通过阈值却飘不成：必须说清是**被规则挡住**而不是「没算过阈值」，
            # 否则真机日志会把这个局面记成一次「保守胡牌」。
            self.last_detail["piao_blocked_by"] = "catch-play"
            self.last_reason = (
                f"胡牌：番 {current.fan}。生存 {survival:.2f} > 阈值 {threshold:.2f}，"
                "但本座受抓打圈限制、只能打刚摸到的那张，无法弃财神续飘"
            )
            return hu_action
        self.last_reason = f"胡牌：番 {current.fan}，生存 {survival:.2f} 未过阈值 {threshold:.2f}"
        return hu_action

    def _piao_candidate(
        self, situation: Situation, waiting: Sequence[int], drawn: int
    ) -> int | None:
        """续飘条件：当前爆头，且打出一张财神后仍听任意。"""
        meld_count = situation.hand.meld_count
        if not win.is_baotou(waiting, meld_count):
            return None
        after = list(waiting)
        after[drawn] += 1
        if after[GOD] <= 0:
            return None
        after[GOD] -= 1
        if not win.is_baotou(after, meld_count):
            return None
        return GOD

    def _baotou_by_discard(
        self, situation: Situation, waiting: Sequence[int], drawn: int
    ) -> int | None:
        """找出能把手牌打成**爆头**的那张弃牌（通往财飘链的唯一入口）。

        为什么需要「弃胡」：若 13 张手牌 H 是听任意，则 H 加任意一张都成胡，因此上一个
        回合玩家手里 `H + 被打出的那张` 本已成胡——**不弃胡就走不进 H**。所以爆头只能靠
        「在能胡的时候选择不胡」到达，而这是财飘链的前提。
        """
        meld_count = situation.hand.meld_count
        if win.is_baotou(waiting, meld_count):
            return None  # 已经爆头，走续飘路径
        hand = list(waiting)
        hand[drawn] += 1  # 当前 14 张
        for tile in range(tiles.TILE_KINDS):
            if hand[tile] <= 0:
                continue
            hand[tile] -= 1
            reachable = win.is_baotou(hand, meld_count)
            hand[tile] += 1
            if reachable:
                return tile
        return None

    def _fan_of(self, situation: Situation, waiting: Sequence[int], drawn: int) -> FanResult:
        return compute_fan(
            waiting,
            drawn,
            situation.hand.meld_count,
            chain_count=situation.god.chain_count,
            piao_count=situation.god.piao_count,
            you_cai_bi_kao=self.config.you_cai_bi_kao,
        )

    def _win_points(self, fan: int, situation: Situation) -> int:
        settlement = score_module.settle(fan, self.config.base_score)
        is_dealer = situation.table.dealer_seat == situation.seat
        return settlement.line(dealer_hu=is_dealer).win

    def _loss_points(self, situation: Situation) -> int:
        """对手自摸时本人要付的点数（取最坏位：由庄家自摸、或本人是庄时付 ×8）。"""
        seat = situation.seat
        dealer = situation.table.dealer_seat
        winner = (seat + 1) % 4
        deltas = score_module.seat_deltas(
            int(OPPONENT_FAN_PRIOR), self.config.base_score, winner, dealer
        )
        return abs(min(deltas[seat], 0))

    def _piao_threshold(self, gain: float, loss: float) -> float:
        if gain <= 0:
            return 1.0
        base = (gain + loss) / (2 * gain + loss)
        return min(0.95, base * self.config.piao_threshold_scale)

    # ---- 杠与出牌 ---------------------------------------------------------

    def _best_gang(self, situation: Situation, actions: Sequence[Action]) -> Action | None:
        """杠决策（tasks.md 5.7 / 5.8）。

        杠是「无暴露的 ×2」：链 +1，且自带补牌——补牌不经过任何对手回合，而飘需要熬过
        一整圈。因此只要不让手牌倒退就执行，暗杠额外优先（对手无从利用）。
        """
        gangs = [action for action in actions if action.kind == GANG]
        if not gangs or not situation.table.can_gang:
            # 牌墙尾部禁杠由规则引擎保证，这里再挡一道：不假设传入动作一定合法
            return None
        current = _current_shanten(situation)
        if current is None:
            return None
        best: tuple[int, Action] | None = None
        for action in gangs:
            after = self._shanten_after_gang(situation, action)
            if after is None or after > current:
                continue
            score = 2 if after < current else 1
            if action.gang_kind == ANGANG:
                score += 1
            # 杠会把牌移入副露、手牌回到「13 张等效听牌态」且不经过弃牌——因此它能
            # **创造**一个弃牌路径到不了的爆头态（服务端也正是在杠动作时重算该标记）。
            # 杠完即听任意 ⇒ 补牌自摸就是杠爆 = 爆头 × 杠开 = ×4，比普通杠开翻倍。
            if self._gang_reaches_baotou(situation, action):
                score += 4
            if best is None or score > best[0]:
                best = (score, action)
        if best is None:
            return None
        chosen = best[1]
        if self._gang_reaches_baotou(situation, chosen):
            self.last_reason = f"杠求爆头：杠后即听任意，补牌自摸为杠爆 ×4（{chosen.describe()}）"
        else:
            self.last_reason = f"杠：链 +1 且自带补牌，手牌不倒退（{chosen.describe()}）"
        return chosen

    def _gang_reaches_baotou(self, situation: Situation, action: Action) -> bool:
        """该杠动作是否让手牌进入爆头（听任意）态——杠爆（×4）的前提。"""
        tile = action.tile
        if tile is None:
            return False
        hand = situation.hand
        counts = list(hand.counts)
        if action.gang_kind == "bugang":
            if counts[tile] < 1:
                return False
            counts[tile] -= 1
            meld_count = hand.meld_count
        else:
            if counts[tile] < tiles.COPIES_PER_KIND:
                return False
            counts[tile] -= tiles.COPIES_PER_KIND
            meld_count = hand.meld_count + 1
        if tiles.total_tiles(counts) != tiles.HAND_SIZE - tiles.MELD_SLOTS * meld_count:
            return False
        return win.is_baotou(counts, meld_count)

    def _shanten_after_gang(self, situation: Situation, action: Action) -> int | None:
        hand = situation.hand
        tile = action.tile
        if tile is None:
            return None
        counts = list(hand.counts)
        if action.gang_kind == "bugang":
            if counts[tile] < 1:
                return None
            counts[tile] -= 1
            meld_count = hand.meld_count
        else:
            if counts[tile] < tiles.SET_LENGTH:
                return None
            counts[tile] -= tiles.SET_LENGTH
            meld_count = hand.meld_count + 1
        return _best_after_drop(counts, meld_count)

    def _choose_discard(self, situation: Situation, actions: Sequence[Action]) -> Action | None:
        candidates = [action for action in actions if action.kind == DISCARD]
        if self.config.preserve_god:
            without_god = [action for action in candidates if action.tile != GOD]
            if without_god:
                candidates = without_god
        if not candidates:
            return next((action for action in actions if action.kind != PASS), None)
        # 同分（total 完全相等）时的次序：默认靠 `sorted` 的稳定性落到**最小牌索引**上，
        # 那是一个没被设计过的花色偏置（见 `safe_tiebreak` 的说明）。开启后改成
        # 「已见张多的优先」——只影响完全打平的候选，动不了任何一项的排序。
        if self.config.safe_tiebreak:
            seen = shanten_module.visible_counts(
                situation.hand.counts,
                [meld.tiles for meld in situation.all_melds],
                situation.discards,
            )
            scores = sorted(
                (self._score_discard(situation, action) for action in candidates),
                key=lambda item: (item.total, seen[item.tile]),
                reverse=True,
            )
        else:
            scores = sorted(
                (self._score_discard(situation, action) for action in candidates),
                key=lambda item: item.total,
                reverse=True,
            )
        best = scores[0]
        # `exact-ukeire` / `ukeire`：整层进张次排序。`tenpai-only`：**只保留听牌态那一层**
        # （即 v3 的 `wait_aware_tenpai`），把向听 ≥1 的排序交还给主排序键。
        #
        # 为什么要这个中间档：`exact-ukeire` 会拿进张去**重排** top 2，
        # 于是「统一期望得分」算出来的次序大部分被它覆盖（实测与 v3 的分歧只有 5.5%），
        # 而统一得分本来就是要**取代**这层补丁——它存在只是因为旧量纲下 59.5% 的决策并列。
        # 但直接关掉整层会连 v3 唯一被证过的机制（听牌按可见听口选牌）一起丢掉，
        # 那样测的就不是「统一得分 vs v3」，而是「统一得分 减 一个已知有效组件」。
        ties = self.config.tiebreak
        if ties in ("ukeire", "exact-ukeire") or (ties == "tenpai-only" and best.shanten == 0):
            best = self._break_ties_by_ukeire(situation, scores) or best
        self.last_detail["discards"] = [score.describe() for score in scores[:4]]
        self.last_reason = f"出牌：{best.describe()}"
        return Action(DISCARD, tile=best.tile)

    def _break_ties_by_ukeire(
        self, situation: Situation, scores: Sequence[DiscardScore]
    ) -> DiscardScore | None:
        """在向听相同的候选项里，选出**进张最多**的那张。

        两种口径（`tools/analyze_ukeire_fidelity.py` 量化了差别）：

        - ``ukeire``：廉价估计，每张 1.1–1.6 ms。**实测不可用** —— 在 150 副 1 向听手牌上，
          它与精确口径选出同一张打牌的比例只有 **14.7%**，不一致时平均放弃 **15.4 张**
          精确进张（同位置最佳与最差候选平均差 29.5 张）。所以旧档位测出「无增益」是噪声。
        - ``exact-ukeire``：精确 ``shanten.ukeire``，42–157 ms/张。只算**向听 ≤ 2 且并列
          候选最多前 3 张**，并加 0.6 秒墙钟上限——出牌预算 1800 ms，当前实测决策耗时
          仅 0.5–4 ms，余量足够；超时即用已算出的最优，绝不阻塞窗口。
        """
        top_shanten = scores[0].shanten
        tied = [score for score in scores if score.shanten == top_shanten]
        if len(tied) < 2:
            return None
        exact = self.config.tiebreak in ("exact-ukeire", "tenpai-only")
        if exact and top_shanten > self.config.ukeire_max_shanten:
            return None
        # 听牌档：改比**可见听口张数**，并且不截断候选面。
        # 判据见 `PolicyConfig.wait_aware_tenpai`——`ukeire` 在向听 0 时返回空元组，
        # 老代码把「全是 0」当平局，于是听牌时的次排序其实什么都没做。
        wait_aware = exact and top_shanten == 0 and self.config.wait_aware_tenpai
        # 两拍只用在向听 1：向听 0 已有 `wait_aware`（那一层用的是同一条思路的终点），
        # 向听 ≥2 时**连一拍都还没走稳**，两拍的噪声会压过信号（且成本翻倍）。
        two_ply = exact and top_shanten == 1 and self.config.two_ply_shanten1
        if exact:
            if self.config.ukeire_order == "blocks":
                tied = sorted(tied, key=lambda item: -item.blocks)
            if not wait_aware and not two_ply:
                tied = tied[: max(1, self.config.ukeire_candidates)]
        visible = shanten_module.visible_counts(
            situation.hand.counts,
            [meld.tiles for meld in situation.all_melds],
            situation.discards,
        )
        deadline = time.monotonic() + EXACT_UKEIRE_BUDGET_SEC if exact else None
        # **一次决策共用一个 memo**：`ukeire` 内部对 34 个牌种各做一次 best_shanten，
        # 而同一决策里的各个候选共用同一副手牌，子问题大量重叠。`ukeire` 早就支持
        # `memo` 形参，但决策层一直没传——等于每个候选都从空表重算。
        # 这是**纯提速、零行为变化**，直接换来更多可用候选与更大的真机安全余量
        # （出牌预算 1800 ms，精确进张单次 42–157 ms）。
        memo: dict = {}
        best: tuple[int, DiscardScore] | None = None
        for score in tied:
            counts = list(situation.hand.counts)
            counts[score.tile] -= 1
            if wait_aware:
                copies = _wait_copies(counts, situation.hand.meld_count, visible, score.tile)
                if copies is None:
                    # 算不出来就跳过该候选并留痕——不能当成 0 参与比较
                    self.last_detail["wait_copies_failed"] = (
                        self.last_detail.get("wait_copies_failed", 0) + 1
                    )
                    continue
            elif two_ply:
                # 两拍值只在**超时回退之外**算：deadline 已到就停在当前最优上，
                # 绝不为了算完而拖过墙钟上限（出牌预算 1800 ms / 平台 3 s）。
                if deadline is not None and time.monotonic() > deadline and best is not None:
                    self.last_detail["tiebreak_timeout"] = True
                    break
                copies = int(_two_ply_value(counts, situation.hand.meld_count, visible, memo))
            elif exact:
                if deadline is not None and time.monotonic() > deadline and best is not None:
                    self.last_detail["tiebreak_timeout"] = True
                    break
                entries = shanten_module.ukeire(
                    counts, situation.hand.meld_count, visible=visible, memo=memo
                )
                copies = sum(copy for _, copy in entries)
            else:
                _, copies = cheap_ukeire(counts, visible)
            if best is None or copies > best[0]:
                best = (copies, score)
        if best is not None:
            metric = (
                "可见听口" if wait_aware
                else ("两拍值" if two_ply else ("精确" if exact else "廉价") + "进张")
            )
            self.last_detail["tiebreak"] = (
                f"向听 {top_shanten} 并列 {len(tied)} 张，{metric} {best[0]} 张"
            )
        return None if best is None else best[1]

    def _score_discard(self, situation: Situation, action: Action) -> DiscardScore:
        """出牌评分（tasks.md 5.4 / 5.6）。
        主项是**打完这一张之后两条路线各自的期望得分**，取较优者——这直接实现了「两条
        路线独立估值再比较」，而不是用一个手调打分函数隐式混合。期望得分 = 该向听的
        实测胜率 × 该路线可及的番 × 庄闲赔付，再减去失败时承担的赔付。

        另减去「喂牌」与「打出财神」的代价。本平台没有点炮，所以不存在传统安全牌，
        喂牌代价按「对手手牌推进被加速」计价。
        """
        tile = action.tile
        assert tile is not None
        hand = situation.hand
        counts = list(hand.counts)
        counts[tile] -= 1
        if self.config.shape_value:
            block_value = shanten_module.shape_value(counts, hand.meld_count)
        else:
            blocks = shanten_module.quick_blocks(counts)
            block_value = 2 * blocks[0] + blocks[1]

        risks = self._risks(situation)
        threat = sum(item.ready_probability for item in risks)
        feed = risk.visible_need(tile) * threat
        if self.config.feed_visibility:
            # 乘「还剩几张未现」因子：`seen` = 本方暗手 + 四家副露 + 四家弃牌，
            # `4 − seen` 是仍可能被对手拿到的张数（藏在对手手牌或牌墙里）。
            # 这张牌打出去之后自己也会成为可见牌，所以 `seen` 用**打出前**的口径即可——
            # 我们手里那张本来就在 `hand.counts` 里。
            seen = shanten_module.visible_counts(
                situation.hand.counts,
                [meld.tiles for meld in situation.all_melds],
                situation.discards,
            )
            feed *= max(0.0, tiles.COPIES_PER_KIND - seen[tile]) / tiles.COPIES_PER_KIND
        god_penalty = self.config.god_discard_penalty if tile == GOD else 0.0
        # 庄家局的喂牌权重单独缩放：庄闲赔付是 8 倍不对称，而决策层此前完全不分庄闲。
        feed_scale = self.config.feed_weight
        if (
            self.config.dealer_feed_scale != 1.0
            and situation.table.dealer_seat == situation.seat
        ):
            feed_scale *= self.config.dealer_feed_scale

        if not self.config.unified_score and not self.config.route_aware and self.config.commitment is Commitment.NONE:
            if self.config.natural_route:
                shanten_value = shanten_module.natural_shanten(counts, hand.meld_count)
            else:
                value = _shanten_or_none(counts, hand.meld_count)
                shanten_value = shanten_module.SHANTEN_MAX if value is None else value
            total = (
                -self.config.shanten_weight * shanten_value
                + block_value
                - feed_scale * feed
                - god_penalty
            )
            return DiscardScore(
                tile=tile,
                shanten=shanten_value,
                blocks=block_value,
                route="none",
                pair_value=0.0,
                meld_value=0.0,
                route_value=0.0,
                feed_cost=feed,
                god_penalty=god_penalty,
                total=total,
            )

        pair_route, meld_route = routes.evaluate(
            situation, base_score=self.config.base_score, counts=counts
        )
        if self.config.commitment is Commitment.PAIR:
            best = pair_route
        elif self.config.commitment is Commitment.MELD:
            best = meld_route
        else:
            best = pair_route if pair_route.value >= meld_route.value else meld_route
        god_penalty += self._god_route_penalty(tile, pair_route)

        # **统一期望得分**（`unified_score`）在这一支只做两处替换，结构其余部分与
        # `route_aware` 完全相同——这是刻意的：`routes.evaluate` 的 `value` **本来就是**
        # `P_win × E_pay − (1−P_win) × P_opp × E_loss`（见 `routes.py:181/187`），
        # 也就是说 B' 设计说明的前两项**早已存在**，缺的只是下面这两处：
        #
        #   ① `value_weight=10.0` 是拍出来的倍数（与 `−10×向听` 同一个病：用一个任意常数
        #      决定「路线期望」和「喂牌」谁大谁小）⇒ 统一得分里取 **1.0**，直接用量纲=分的原值；
        #   ② `feed` 从 `feed_weight × visible_need × threat`（无量纲的旋钮）换成
        #      **`ΔP_opp(tile) × E_loss`**（量纲=分）——喂牌从旋钮变成推导量，`feed_weight` 被消灭。
        value_scale = self.config.value_weight
        if self.config.unified_score:
            value_scale = 1.0
            feed_scale = 1.0
            feed = (
                self.config.feed_ready_increment
                * risk.visible_need(tile)
                * meld_route.loss
            )
            best_value = self._corrected_route_value(best, situation, risks)
            total = value_scale * best_value - feed_scale * feed - god_penalty
            return DiscardScore(
                tile=tile,
                shanten=meld_route.route_shanten,
                blocks=block_value,
                route=best.route.value,
                pair_value=pair_route.value,
                meld_value=meld_route.value,
                route_value=best_value,
                feed_cost=feed,
                god_penalty=god_penalty,
                total=total,
            )

        total = (
            value_scale * best.value
            - feed_scale * feed
            - god_penalty
        )
        return DiscardScore(
            tile=tile,
            shanten=meld_route.route_shanten,
            blocks=block_value,
            route=best.route.value,
            pair_value=pair_route.value,
            meld_value=meld_route.value,
            route_value=best.value,
            feed_cost=feed,
            god_penalty=god_penalty,
            total=total,
        )

    def _corrected_route_value(
        self,
        route: routes.RouteValuation,
        situation: Situation,
        risks: Sequence[risk.OpponentRisk],
    ) -> float:
        """把 `P_win` 的按向听乘性修正施加到一条路线的期望值上（无修正时原值返回）。

        **为什么是「重算」而不是「改表」**：`win_probability` 的输出同时被三处消费——
        出牌评分（本支）、碰吃闸门（`_choose_response`）、路线比较。改表 = 让**冠军档的碰吃行为
        一起漂移**，那是一次无人察觉的行为变更（本仓踩过：`configure` 丢字段导致真机全跑默认档）。
        所以修正只在本支重算，其余两处仍走未修正的表。

        重算用 `evaluate` 已经算好的分解量，不重复调 `routes`：
        `gain = P_win × E_pay` ⇒ `E_pay = gain / reach`；`loss` 就是 `E_loss`。
        """
        correction = self.config.win_table_correction
        if not correction or route.reach <= 0:
            return route.value
        index = max(0, min(route.route_shanten, len(correction) - 1))
        p_win = min(0.95, route.reach * correction[index])
        e_pay = route.gain / route.reach
        opponent_win = 1.0 - risk.lap_survival(risks)
        return p_win * e_pay - (1.0 - p_win) * opponent_win * route.loss

    def _god_route_penalty(self, tile: int, pair_route: routes.RouteValuation) -> float:
        """打财神的额外机会成本：它是本平台唯一的链货币，也是七对路线的唯一入口。"""
        if tile != GOD:
            return 0.0
        return max(0.0, pair_route.value) * 0.5

    # ---- 响应窗口 ---------------------------------------------------------

    def _choose_response(self, situation: Situation, actions: Sequence[Action]) -> Action:
        """碰/吃/明杠的取舍（tasks.md 5.5 / 5.6）。

        副露能加速手牌，但不进动作链（只有杠与飘进链），而且**副露会让七对路线永久作废**。
        因此判据是路线比较：把「吃完/碰完之后的副露路线期望」与「保留七对路线的期望」
        相比，只有在副露确实更优、且确实改善向听时才吃碰。
        """
        fallback = next((action for action in actions if action.kind == PASS), actions[0])
        offered = situation.offered_tile
        if offered is None:
            return fallback
        hand = situation.hand
        current = shanten_module.shanten_any(hand.counts, hand.meld_count)

        if self.config.commitment is Commitment.PAIR:
            self.last_reason = "路线承诺（标定用）：七对路线不吃不碰"
            return fallback

        gang = next((action for action in actions if action.kind == GANG), None)
        if gang is not None:
            after = self._shanten_after_meld(situation, offered, PENG)
            if after is not None and after <= current:
                self.last_reason = "明杠：链 +1 且自带补牌，手牌不倒退"
                return gang

        if self.config.commitment is Commitment.MELD:
            best_meld: tuple[int, Action] | None = None
            for action in actions:
                if action.kind not in (PENG, CHI):
                    continue
                after_shanten = self._shanten_after_meld(situation, offered, action.kind)
                if after_shanten is None or after_shanten >= current:
                    continue
                if best_meld is None or after_shanten < best_meld[0]:
                    best_meld = (after_shanten, action)
            if best_meld is None:
                self.last_reason = f"路线承诺（标定用）：向听 {current} 无改善"
                return fallback
            self.last_reason = f"路线承诺（标定用）：副露 {best_meld[1].describe()}"
            return best_meld[1]

        if not self.config.route_aware:
            if self._is_pair_route(situation):
                self.last_reason = "保留七对路线：不吃不碰"
                return fallback
            best_legacy: tuple[int, Action] | None = None
            # 放宽档允许「向听不变」的副露，但**排除已听牌**（0 向听）——那时副露只会
            # 换掉听口，不会让你更接近胡牌。见 MeldTolerance 的实测依据。
            # **必须用 == 而不是 is**：`PolicyConfig.for_mode(mode, meld_tolerance="equal")`
            # 传进来的是普通 str，而 StrEnum 成员与它的值是 `==` 相等但 `is` 不等，
            # 用 `is` 会让这个档位**静默退化成 strict**（造档时踩过一次）。
            tolerance = self.config.meld_tolerance
            threshold = {
                MeldTolerance.EQUAL: 1,
                MeldTolerance.EQUAL_EARLY: 2,
            }.get(tolerance)  # type: ignore[arg-type]
            accept_equal = threshold is not None and current >= threshold
            for action in actions:
                if action.kind not in (PENG, CHI):
                    continue
                after_legacy = self._shanten_after_meld(situation, offered, action.kind)
                if after_legacy is None or after_legacy > current:
                    continue
                if after_legacy == current and not accept_equal:
                    continue
                # 同向听时优先碰：吃全局最多 2 副，碰不占这个额度
                rank = (after_legacy, 0 if action.kind == PENG else 1)
                if best_legacy is None or rank < (
                    best_legacy[0],
                    0 if best_legacy[1].kind == PENG else 1,
                ):
                    best_legacy = (after_legacy, action)
            if best_legacy is None:
                self.last_reason = f"不副露：向听 {current} 无改善"
                return fallback
            suffix = "（向听不变，放宽档）" if best_legacy[0] == current else ""
            self.last_reason = (
                f"副露 {best_legacy[1].describe()}：向听 {current}→{best_legacy[0]}{suffix}"
            )
            return best_legacy[1]

        pair_route, meld_route = routes.evaluate(situation, base_score=self.config.base_score)
        self.last_detail["routes"] = [pair_route.describe(), meld_route.describe()]

        best: tuple[int, Action, float] | None = None
        for action in actions:
            if action.kind not in (PENG, CHI):
                continue
            after_shanten = self._shanten_after_meld(situation, offered, action.kind)
            if after_shanten is None or after_shanten >= current:
                continue
            plan = self._meld_plan(situation, offered, action)
            if plan is None:
                continue
            hypothetical, after_counts = plan
            _, after_value = routes.evaluate(
                situation,
                base_score=self.config.base_score,
                counts=after_counts,
                melds=hypothetical,
            )
            if pair_route.feasible and after_value.value <= pair_route.value:
                continue
            if best is None or after_value.value > best[2]:
                best = (after_shanten, action, after_value.value)
        if best is None:
            self.last_reason = (
                f"不副露：向听 {current} 无改善，或保留七对路线更优"
                f"（七对 {pair_route.value:.1f} / 副露 {meld_route.value:.1f}）"
            )
            return fallback
        self.last_reason = (
            f"副露 {best[1].describe()}：向听 {current}→{best[0]}，"
            f"副露路线期望 {best[2]:.1f} 超过七对 {pair_route.value:.1f}"
        )
        return best[1]

    def _meld_plan(
        self, situation: Situation, offered: int, action: Action
    ) -> tuple[tuple[Meld, ...], list[int]] | None:
        """假设执行该副露之后的（副露列表, 暗手牌计数）——用于反事实估值。

        两者必须同时给出：副露使暗手少两张，只改副露不改手牌会让张数与副露数不相容。
        """
        hand = situation.hand
        counts = list(hand.counts)
        if action.kind == PENG:
            if counts[offered] < 2:
                return None
            counts[offered] -= 2
            return hand.melds + (Meld(kind=MELD_KIND_PENG, tiles=(offered,) * 3),), counts
        combos = chi_combinations(counts, offered)
        if not combos:
            return None
        first, second = combos[0]
        counts[first] -= 1
        counts[second] -= 1
        return (
            hand.melds + (
                Meld(kind=MELD_KIND_CHI, tiles=tuple(sorted((first, second, offered)))),
            ),
            counts,
        )

    def _is_pair_route(self, situation: Situation) -> bool:
        """粗糙的七对门槛（仅在 ``route_aware`` 关闭时使用）。"""
        if situation.hand.meld_count:
            return False
        pairs = sum(amount // 2 for amount in situation.hand.counts if amount >= 2)
        return pairs >= self.config.pair_route_pairs

    def _shanten_after_meld(self, situation: Situation, offered: int, kind: str) -> int | None:
        """吃/碰后（并打出多余一张）能达到的最小向听。"""
        hand = situation.hand
        counts = list(hand.counts)
        if kind == PENG:
            if counts[offered] < 2:
                return None
            counts[offered] -= 2
        else:
            combos = chi_combinations(counts, offered)
            if not combos:
                return None
            first, second = combos[0]
            counts[first] -= 1
            counts[second] -= 1
        return _best_after_drop(counts, hand.meld_count + 1)


__all__ = ["DiscardScore", "HeuristicDecider", "Mode", "PolicyConfig", "cheap_ukeire"]
