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

# shanten_fast Cython 加速（条件导入，无 .so 时回退纯 Python）
# 覆盖 5 个重函数：shanten/best_shanten/shanten_any/seven_pairs_shanten/ukeire
# 通过 monkey-patch 挂到 shanten_module 上：所有调用点（本文件及 routes/rollout/features/oracle）
# 自动走 fast 版，无需改调用代码。ShantenError 统一为纯 Python 版类对象，except 不裂化。
try:
    from majiang.rules import shanten_fast as _shanten_fast_mod
    _shanten_fast_mod.ShantenError = shanten_module.ShantenError
    for _fn in ("shanten", "best_shanten", "shanten_any", "seven_pairs_shanten", "ukeire"):
        setattr(shanten_module, _fn, getattr(_shanten_fast_mod, _fn))
    _HAS_SHANTEN_FAST = True
except ImportError:
    _HAS_SHANTEN_FAST = False
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
    "meld_conditional",
    "god_wait_boost",
    "standing_feed_apply",
    "god_penalty_behind",
    "standing_remaining_max",
    "standing_delta_max",
    "standing_behind_scale",
    "standing_lead_scale",
    "tiebreak",
    "safe_tiebreak",
    "unified_score",
    "feed_ready_increment",
    "win_table_correction",
    "goodshape_tolerance",
    "ukeire_preselect",
    "edge_partial_weight",
    # **保留多余对子**（`v7-keappairs`，2026-10-08）：>0 时给被 `shape_value` 的
    # `weights[:need]` 裁剪掉的多余对子一个折扣正值（字牌 0.5×、数牌 0.3×）。默认 0.0 ⇒ 逐位等于 v5。
    "keep_extra_pairs",
    # **多选择吃法：按具体吃法算向听 + 同向听按进张宽度打破平局**（`v7-keepchi`，2026-10-08）。
    # 默认 False ⇒ 逐位等于 v5。
    "meld_chi_best",
    "meld_chi_tiebreak",
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
    # **按 cell 条件化闸门**（`v7m` v1，A 2026-10-03 22:55 立）：只在对手**副露密度高**的
    # 「巡目 × 向听」cell 里允许「向听不下降」的副露，其余 cell 维持 STRICT。
    #
    # **为什么是条件化而不是一律放开**：`v6-equal`（一律放开）在现代基座上实测为负
    # （平台口径每场名次分 −0.216(t−2.39)、胡次数 −0.067(t−2.64)）。C 的 1a（`agent/verify/meld_cond_dist_probe.py`，
    # 7056 房 / 3190 个副露事件）给出一致解释：**头部 bot 的副露密度集中在早中盘 × 浅向听**——
    # 1-4 巡 0/1/2/3 向听 = 30.5%/41.7%/34.7%/26.3%；5-8 巡 = 15.5%/25.8%/26.8%/24.7%；
    # 9-12 巡 = 8.5%/17.0%/20.1%/0%；**13+ 巡 ≤11%（3 向听仅 0.4%）**。
    # 也就是「一律放开」的伤害正来自它在 **bot 自己也不副露**的 cell（尾盘、深向听）里放开。
    #
    # **cell 表**（照抄 1a 的密度形状，不做插值/拟合）：
    #   - 打出次数 ≤ 8（早中盘）且 副露前向听 ≤ 3 ⇒ 允许（该区 bot 密度 24.7%~41.7%）
    #   - 打出次数 9~12 且 副露前向听 ≤ 2 ⇒ 允许（该区 bot 密度 8.5%~20.1%）
    #   - 其余（13+ 巡、或深向听见 0.4%）⇒ **仍然 STRICT**，与 bot 一致
    # 仍**拒绝 0 向听**的副露（与 `MeldTolerance` 的既有依据一致：听牌后副露只换听口，
    # 「换听口」是**另一个变量**，留给 `v7m` v2，避免一档测两件事）。
    #
    # **风险如实记录**：这条轴已被驳倒两次（旧时代 `meld-equal` 三配置一致为负、今晚 `v6-equal`），
    # 且「副露率 × 胜率 r=+0.839」**可能是共因**（强 bot 两者都高）——因果方向已被 `v6-equal` 反证。
    # 因此本档只动「cell 条件化」这一个变量，并配预登记的 kill_criteria 与机制门（见 THREAD 22:55）。
    meld_conditional: bool = False
    # **G1 持财神听口加权系数**（`v5-godwait`，A 2026-10-04 落地 B' 的实现稿）。
    # 0 = 关闭（逐位等于 v5）；>0 = 出牌后手牌含财神时，候选键加 `boost × 财神数 × 听口种数`。
    # 依据与不变量见 `_break_ties_by_ukeire` 里那段注释（含「为什么不能只放在 wait_aware 分支」）。
    god_wait_boost: float = 0.0
    # **S3 局况姿态**（`v5-standing`，A 2026-10-04）。三个条件同时成立才生效：
    # 名次在边界（1 或 ≥3）、与相邻名次分差 ≤ `standing_delta_max`（默认 29 分）、余局 ≤ `standing_remaining_max`（默认 1）。
    # 语义：`standing_lead_scale` > 1 = 守（阈值升、见胡就收）；`standing_behind_scale` < 1 = 搏（阈值降、追大牌）。
    # **两个都是 1.0（默认）⇒ 逐位等于 v5**；比分拿不到或并列也 ⇒ 1.0（安全兜底）。
    # 依据与定义见 `_standing_scale`；窄臂占比 7.28%（B' 11:42 实测）。
    standing_lead_scale: float = 1.0
    standing_behind_scale: float = 1.0
    # **财神线（机制件）**：v1 的形态（缩放惩罚）已被实测证伪（2026-10-05 22:05 铁律），v2 见下方 `god_penalty_behind`。
    # 0 = 关闭（逐位等于 v5）；>0 = 当「手里有财神且到听进度落后」时，把**打出财神的惩罚**乘这个系数。
    #
    # **现象与依据**（三处独立证据交汇在同一点）：
    #  ① C 19:32 分歧对拍：触发我预登记判据的 5 个大桶**全是有财神桶**（我方胡率 −3.0~−6.1pp、
    #     对手同手更高；无财神桶全不触发），且「有财神时先到听率」我们低于 bot（1 向听 35.2% vs 37.9%）；
    #  ② S2：爆头率只有 bot 的 1/3、平胡率 −0.16 fan/局；
    #  ③ B' 19:35：**财飘次数 21 vs bot 41（约 1/2）**——对手比我们更常把财神打出去博链。
    #  ⇒ 解释：我们把财神当 buffer **囤着**，错过转化时机（代价：更晚听、更少胡）。
    #     所以 v1 只做一件事：**进度落后时，降低「留财神」的隐性偏好**（更愿意打财神博财飘链）。
    #
    # **不变量**：手里无财神 ⇒ 打出财神的候选根本不存在 ⇒ 该系数无从生效 ⇒ **逐位等于 v5**（结构性保证，非巧合）。
    # **财神线 v2**（A 2026-10-05 23:20 更正后的形态）：**把恒定的「打财神禁忌」条件化**。
    #
    # **为什么不是「缩放」而是「换值」**（读代码实测）：`god_discard_penalty = 25.0`，
    # 而候选间的其他差异范围是 **个位数**（`shanten_weight=10.0` 一个向听步 10 分、`feed` 项 ~1~2 分、
    # `shape_value` 的形质项总幅度 **≤0.36 分**）⇒ 25 分 = **2.5 个向听步 ≈ 其他差异范围的 12~20 倍**
    # ⇒ **打财神在评分层被结构性禁止**（缩放它、或抠 `shape_value` 里财神那 1.4 个搭子，都碰不到尾巴）。
    # ⇒ 正确做法是**换一个值**：在「有财神且到听进度落后」时改用**与其他项同量级**的禁忌值（个位数），
    # 让「打财神」**进入正常竞争**（由 chain/进张/形质去裁决）；进度正常时**保持 25.0**（不无脑放开）。
    #
    # **证据三处交汇**：C 19:32 分歧对拍（触发桶全在「有财神」）+ B' 19:35（**财飘 21 vs bot 41 ≈ 一半**）+ S2（爆头率 1/3）。
    # 结构性事实：我们的「打财神」**只从 `_choose_win_or_piao` 的「弃胡求爆头」一条窄路进来**（那边直接构造 Action、绕过本函数）
    # ⇒ 不是「打得少」，是**几乎没有入口**。
    #
    # 默认值 = `25.0`（与 `god_discard_penalty` 相同）⇒ **逐位等于 v5**（关闭态）。
    god_penalty_behind: float = 25.0
    standing_feed_apply: bool = False
    standing_delta_max: int = 29
    standing_remaining_max: int = 1
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
    # **破平层的作用面收口**：`_break_ties_by_ukeire` 的候选面原本是「**同向听**的全部候选」，
    # 胜者整张替换 `scores[0]` ⇒ 主分 `total` 里除向听以外的全部信息（形质/喂牌/财神罚）
    # 在该层被覆盖。本旋钮把候选面收口到「主分不低于 `最高分 − slack`」的那些候选。
    #
    # **缺陷证据（B' 2026-10-09 `cc5338d3` 立案 + 用户报障总表 v2 第一节「硬缺陷」4 点）**：
    # `tools/trigger_census_tiebreak.py` 试跑 120 房 → 可比点 1,706，破平层换掉主分最高者
    # 400（23.4%），其中**真·覆盖（gap>0）197 = 11.5%**。4 个报障点
    # （seq53/55/113/197）在 `tiebreak` 跳过破平层后**逐点给出报障总表的「最高分候选」列**
    # （发/2t/1w/1b）——即用户主张的那张；且四例的差异**全部来自 `喂牌` 项**（形质/七对/副露全同）。
    #
    # **`float("inf")` = 关闭收口 = 逐位等于 v5**（默认档、冠军档保持不动）。
    # `0.0` = 只在**主分完全相等**的候选间破平（字面意义的 tie-break；`gap==0` 那一层仍在，
    # 实测占决策 11.9%，是这层的正当作用面）。
    tiebreak_total_slack: float = float("inf")
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
    # 实验档位：**进张数一级键、好型率二级键（带容差）**（B' 调研的候选③）。
    #
    # 依据 `research/qualitative-leap-survey.md` §2.1.1（RiichiBook ch3 §3.4）：
    # 「perfect 1-away（2 両面搭子 + 2 対子）无论怎么进张，最终听口一定是**两面**」
    # ⇒ **同为 1 向听，好型听牌率可以是 100% 或 50~70%**，同向听层内的质量差主要就在这里。
    # 调研给的读法是「进张枚数一级键、好型变化二级键」。
    #
    # 实现口径：把并列候选按 `copies` 排序后，**只保留 copies >= tolerance × max(copies) 的那些**，
    # 在其中取形质结构最好的（`shanten.shape_mix` 的结构性代理）——
    # 这样「多 1 枚进张但变愚形」不会被它翻掉（一级键优先），而「进张差不多时」才比好型。
    # `0.0` = 关闭（默认）。
    #
    # **为什么用代理而不是精确好型率**：精确版要对每个进张枚举一次听口类型，
    # 成本会压到 v4 已有的 **p99 851ms / 1800ms** 之上（`docs/ops.md` 的红线是 p99 ≤1000ms）。
    # 结构性代理（両面/対子/坎张的计数）与 `shape_value` 同源、微秒级。
    goodshape_tolerance: float = 0.0
    # 实验档位：**候选面的廉价预筛**（2026-10-01 19:20，用户批准的 ①）。
    #
    # **要解决的问题**：候选面加宽的价值已证（`v5-cand5` +0.639 > `v5-cand3` +0.451），
    # 但**真机延迟卡住**——候选面每 +1 张多算一次精确进张（42–157ms），
    # v5(cand3) 的真机 p99 已 **860ms / 预算 1800ms** ⇒ cand7/cand10 上不了平台。
    # ⇒ 离线收益进不了冠军档，这条线就停在这儿。
    #
    # **做法**：用**廉价代理**（`cheap_ukeire`，微秒级）先把并列候选排序，只取前
    # `ukeire_preselect` 张去算精确进张。这样「有效候选面」可以是 8~10 张，
    # 而精确计算只做 N 次 ⇒ **成本回到 cand-N 的量级，收益接近 cand-10 的量级**。
    #
    # **为什么代理够用**：`cheap_ukeire` 与精确口径的「选出同一张」只有 14.7% 一致
    # （`tools/analyze_ukeire_fidelity.py`），但预筛不需要它**排序正确**，
    # 只需要它把**真·最优圈进前 N 名**（召回率）。这两个要求差很远——
    # 这也是为什么当年「用廉价口径替代精确」失败、而「用廉价口径预筛」可能成立。
    # **取值待 B' 的「召回率 @ N」表定**；启用本开关时 `ukeire_candidates` 会被绕过。
    ukeire_preselect: int = 0
    # 实验档位：**边张搭（12 / 89）单独计权**（agent-c 22:20 的机制信号，2026-10-02 量化确认）。
    # 实测：`8w9w` 与真两面 `4w5w` 在 `shape_value` 里**同值**（都 1.2），而进张是 **4 vs 8 张**；
    # 坎张 `1w3w`（同样 4 张）只有 0.7 ⇒ **边张搭被高估一倍**。
    # `0.0` = 关闭（保持旧行为）；0.7 = 与坎张同权；0.8 = 介于坎张与真两面之间。
    edge_partial_weight: float = 0.0
    # **保留多余对子**（`v7-keappairs`，2026-10-08）：`shape_value` 的 `weights[:need]` 裁剪
    # 把「排不进前 `need` 名的块」当 0，于是**多余的对子**在估值里完全消失——拆掉它不掉分。
    # 实测（1200 副「唯一对子=东东」手牌）唯一对子不会被拆；但当手里有 ≥2 个对子时，
    # 多余的那个被当 0，会与孤立字牌在**平局规则**里被重排 ⇒ 偶尔拆掉对子（含字牌对子，
    # 而它不能被吃、是最好的碰材）。本开关给被裁掉的对子一个折扣正值，使「多留一个对子」
    # 优于「留一张孤张」。幅度 < 1 ⇒ 仍只打破并列、不覆盖块数差（见 `shape_value` 注释）。
    # 0.0 = 关闭（**逐位等于 v5**）。默认档、冠军档在采纳前**一律保持 0.0**（需人拍板）。
    keep_extra_pairs: float = 0.0
    # **①根因修复：按具体吃法算向听**（`v7-chibest`，2026-10-08；A 22:12 裁定为该臂）。
    # `_shanten_after_meld`（及 `_meld_plan`）旧实现无视 `action.tiles`，对每个 CHI 选项都取
    # `chi_combinations()[0]`。同一张 `offered` 的两种吃法可以有**不同**的「吃后最小向听」
    # （实测 4000 副：≥2 种吃法的 2904 个局面里 **1155 个（39.8%）** 不同）。
    # 后果**包括漏吃**：手牌 `1w6w6w8w9w2b3b9b1t北北白白`，上家出 `7w`，吃 `8w9w` 真能到向听 1
    # （连默认 STRICT 也该吃），但 `combos[0]=(6w,8w)` 只到向听 2 ⇒ 误判「无改善」而 PASS。
    # `False` = 关闭（**逐位等于 v5**，走旧 `combos[0]`）；`True` = 吃牌按 `action.tiles` 算。
    meld_chi_best: bool = False
    # **②次排序：同向听时按进张宽度打破吃法平局**（`v7-keepchi`，2026-10-08；A 22:12 裁定：
    # 这是**根因修复的下游**、须**单独成第二臂**，故与 ① 拆开）。
    # 只有 ① 修好后才会出现「同降幅的多种吃法」这个比较（此前不存在）。
    # `5w 4b5b6b7b7b8b 1t4t6t7t8t9t` 吃 `7t`：吃 `8t9t` 留 `6t7t` 两面（`shape_value=4.18`）
    # 优于吃 `6t8t`（`3.955`）；精确进张 65 vs 71 张；旧代码恒选 `combos[0]`。
    # 判据 `shape_value`（打哪张 + 吃哪两张都按它取最优），开销微秒级。
    # `False` = 关闭（**逐位等于 v5**）。
    meld_chi_tiebreak: bool = False
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


def _goodshape_proxy(counts: Sequence[int], meld_count: int) -> float:
    """好型率的**结构性代理**（微秒级）：両面权重最高，坎张扣分。

    与 `shape_value` 同源分解（`shanten.shape_mix`），但刻意保留**结构计数**而非加权和——
    前者能区分「2 両面 + 2 対子」（perfect 1-away）与「1 両面 + 3 対子」，后者在加权和里会撞车。
    """
    runs, pairs, kanchan = shanten_module.shape_mix(counts, meld_count)
    return 3.0 * runs + 1.0 * pairs - 0.5 * kanchan


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


# `meld_conditional` 的 cell 边界（见 `PolicyConfig.meld_conditional` 的完整依据）。
MELD_CELL_EARLY_TURNS = 8
MELD_CELL_MID_TURNS = 12


# 各巡目「应该到几向听」的**粗基准**（`god_penalty_behind` 的判据用）。
# 取法是经验值而非拟合：它只需要把「明显落后」与「正常」分开，不需要精确
# （精确化会引入新的可调参数，而这个机制件的 v1 刻意只动一个系数）。
GOD_PROGRESS_EXPECTED_SHANTEN = ((4, 2), (8, 1))  # 巡目 ≤4 ⇒ 期望 ≤2；≤8 ⇒ ≤1；更晚 ⇒ 0


def _god_progress_behind(situation: Situation) -> bool:
    """手里有财神、且「到听进度落后于巡目」—— `god_penalty_behind`（财神线 v2）的触发条件。

    **巡目用「自己已打出的张数」近似**（与 `_meld_cell_allows` 同一口径，理由见那里）。
    落后 = 当前向听 > 该巡目的期望向听。
    """
    hand = situation.hand
    if hand.counts[tiles.GOD] < 1:
        return False
    seat = situation.seat
    discards = situation.discards
    played = len(discards[seat]) if 0 <= seat < len(discards) else 0
    expected = 0
    for limit, value in GOD_PROGRESS_EXPECTED_SHANTEN:
        if played <= limit:
            expected = value
            break
    current = _shanten_or_none(hand.counts, hand.meld_count)
    if current is None:
        return False
    return current > expected


def _meld_cell_allows(situation: Situation, current: int) -> bool:
    """这个「巡目 × 副露前向听」cell 是否在对手的**高副露密度区**里。

    **巡目用「自己已打出的张数」近似**（`len(situation.discards[seat])`）：在响应吃碰的时刻，
    我们已经打过多少张牌是最容易拿到、且随局progress单调的量。它与「摸牌次数」在无副露时相等，
    有副露时会略微超前（每次副露后仍要打一张但不摸牌）——对这个**只有两三档**的粗分桶足够，
    且误差方向是「把偏早的时点算成偏晚」，即**更保守**（不会因为近似而额外放开闸门）。
    """
    seat = situation.seat
    discards = situation.discards
    played = len(discards[seat]) if 0 <= seat < len(discards) else 0
    if played <= MELD_CELL_EARLY_TURNS:
        return current <= 3
    if played <= MELD_CELL_MID_TURNS:
        return current <= 2
    return False


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
            threshold = self._piao_threshold(gain, loss, situation)
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
        threshold = self._piao_threshold(gain, loss, situation)
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

    def _piao_threshold(self, gain: float, loss: float, situation: Situation) -> float:
        if gain <= 0:
            return 1.0
        base = (gain + loss) / (2 * gain + loss)
        scale = 1.0 if self.config.standing_feed_apply else self._standing_scale(situation)
        return min(0.95, base * self.config.piao_threshold_scale * scale)

    def _standing_scale(self, situation: Situation) -> float:
        """**S3 局况姿态**（A 2026-10-04；窄臂形态，只在「差距可竞争」的边界态生效）。

        **为什么是窄臂而不是换目标**：C 9-30 的审计（`standings_prior_audit.py`，4090 场）已确认
        **平台排序键是总得分** ⇒ 线性效用下逐手 E-max 已最优 ⇒ 「全局换成 P(首名)」预判无效。
        真正的价值只在**边界态**：P(首名 | 名次×余局) 的跨度在边界处最大（名次1余1 = 79.8%、
        名次4余1 = 1.6%），而那正是「一把大牌就能翻盘/被翻盘」的位置。

        **定义（B' 11:42 出数后定稿，占比 7.28%）**：
        - **守**（`standing_lead_scale` > 1）：名次 1 **且** 与第 2 名分差 ≤ `standing_delta_max`；
        - **搏**（`standing_behind_scale` < 1）：名次 ≥3 **且** 与上一名分差 ≤ `standing_delta_max`；
        - **余局 > `standing_remaining_max` ⇒ 中性**；
        - **比分拿不到（真机 `Snapshot.scores` 历史上恒为空）或并列（分差 0）⇒ 中性（1.0）** ——
          这条是**安全兜底**：拿不到局况时行为**逐位等于 v5**，绝不靠猜姿调。

        语义：`scale > 1` ⇒ 阈值升 ⇒ 见胡就收（守）；`scale < 1` ⇒ 阈值降 ⇒ 更易弃胡追大牌（搏）。
        """
        lead = self.config.standing_lead_scale
        behind = self.config.standing_behind_scale
        if lead == 1.0 and behind == 1.0:
            return 1.0
        table = situation.table
        scores = table.scores
        if not scores or len(scores) < 4 or table.rounds_total <= 0:
            return 1.0
        remaining = table.rounds_total - table.round_no + 1
        if remaining > self.config.standing_remaining_max:
            return 1.0
        me = situation.seat
        if not 0 <= me < len(scores):
            return 1.0
        mine = scores[me]
        if any(score == mine for index, score in enumerate(scores) if index != me):
            # **并列 ⇒ 中性**：与最近对手同分时「分差」方向不明确（谁领先要看谁先摸），
            # 按 A 11:35 的定义「分差不明 ⇒ 中性」。缺这条会把「并列第一」误判成「领先 20 分」。
            return 1.0
        above = [score for index, score in enumerate(scores) if index != me and score > mine]
        below = [score for index, score in enumerate(scores) if index != me and score < mine]
        if not above:
            if not below:
                return 1.0
            gap = mine - max(below)
            return lead if 0 < gap <= self.config.standing_delta_max else 1.0
        if len(above) >= 2:
            gap = min(above) - mine
            return behind if 0 < gap <= self.config.standing_delta_max else 1.0
        return 1.0

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
        # 作用面收口（默认 `inf` ⇒ 不生效，逐位等于 v5）：把「主分明显更低」的候选剔出破平。
        # 依据见 `PolicyConfig.tiebreak_total_slack`——不这么做，一个叫 tie-break 的层会覆盖
        # 主分里除向听以外的全部信息（喂牌项是主要受害者）。
        if self.config.tiebreak_total_slack < float("inf"):
            ceiling = scores[0].total - self.config.tiebreak_total_slack
            narrowed = [score for score in tied if score.total >= ceiling]
            if len(narrowed) < 2:
                # 收口后不足两张 ⇒ 没有可破的平局，交还给主分（`_choose_discard` 的 `or best`）。
                return None
            self.last_detail["tiebreak_narrowed"] = (len(tied), len(narrowed))
            tied = narrowed
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
            if not wait_aware and not two_ply and self.config.ukeire_preselect <= 0:
                tied = tied[: max(1, self.config.ukeire_candidates)]
        visible = shanten_module.visible_counts(
            situation.hand.counts,
            [meld.tiles for meld in situation.all_melds],
            situation.discards,
        )
        if self.config.ukeire_preselect > 0 and len(tied) > self.config.ukeire_preselect:
            # 廉价代理预筛：只保留代理口径的前 N 张去算精确进张。
            #
            # **代理必须真的是零成本**（2026-10-01 19:40 实测教训）：初版用 `cheap_ukeire`，
            # 实测单场墙钟 **158s vs cand10 的 144s** ⇒ **比不算预筛还贵**，核心指标不达标
            # （`cheap_ukeire` 内部对每个牌种跑一次 `quick_shanten`，是 ms 级、不是 μs 级）。
            # 现在改用 **`blocks`**——它是 `_score_discard` **早就算好**的骨架厚度，
            # 所以排序**零额外计算**。代价是代理更粗，但预筛只需要「把真·最优圈进前 N」（召回率），
            # 不需要代理排序正确。
            tied = sorted(tied, key=lambda item: -item.blocks)[: self.config.ukeire_preselect]
            self.last_detail["preselect"] = (
                f"廉价预筛到 {len(tied)} 张（原并列 {len(scores)} 张同级）"
            )
        deadline = time.monotonic() + EXACT_UKEIRE_BUDGET_SEC if exact else None
        # **一次决策共用一个 memo**：`ukeire` 内部对 34 个牌种各做一次 best_shanten，
        # 而同一决策里的各个候选共用同一副手牌，子问题大量重叠。`ukeire` 早就支持
        # `memo` 形参，但决策层一直没传——等于每个候选都从空表重算。
        # 这是**纯提速、零行为变化**，直接换来更多可用候选与更大的真机安全余量
        # （出牌预算 1800 ms，精确进张单次 42–157 ms）。
        memo: dict = {}
        best: tuple[int, DiscardScore] | None = None
        # 二级键用：每个候选的 (进张数, 形质代理, 候选)
        goodshape_rows: list[tuple[int, float, DiscardScore]] = []
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
            # **G1 持财神听口加权**（`v5-godwait`，A 2026-10-04 00:15 落地 B' 的实现稿）。
            #
            # 依据（B' 23:40 财神专项 + C 的标定 23:57）：持财神时 bot 的**听口种数** 6.08 vs 我们 4.53（+34%）、
            # 可见张数 +33%；且 `1523/1523 一致（100%）` 的暴力枚举对拍证明**这不是计算口径 bug**，
            # 是**策略行为**——我们的听口选择在「张数」与「种数」之间偏向张数（`_wait_copies` 只看可见张数），
            # 而财神是百搭 ⇒ **种数的边际价值被放大**（每个种数多一张可重指派的空间）。
            #
            # **口径**：出牌后手牌**含财神**（`god_n ≥ 1`）时，把候选键加上 `boost × god_n × 听口种数`。
            # **`god_n == 0` 时逐位等于 v5**（本改动被 `if god_n >= 1` 完全闸住，单测钉住这一点）——
            # 这条不变量是判读的前提：否则「任何改动都扰动」会伪装成信号。
            #
            # **我相对 B' 稿的修正**：B' 稿把这个加权只放在 `wait_aware` 分支里，于是消融臂
            # `v5-godwait-noWA`（`wait_aware_tenpai=False`）**根本不进那个分支** ⇒ 加权无效 ⇒ 那会是一个**空臂**，
            # 什么也测不出来。现在把它放在**分支之后**：`wait_aware` 时加在「可见张数」键上，
            # `exact` 时加在「精确进张」键上 ⇒ `noWA` 臂变成有意义的消融（同一个加权、不同的基础键）。
            if self.config.god_wait_boost > 0 and counts[tiles.GOD] >= 1:
                try:
                    kinds = len(win.winning_draws(counts, situation.hand.meld_count))
                except ValueError:
                    kinds = 0
                if kinds:
                    god_n = counts[tiles.GOD]
                    copies = copies + self.config.god_wait_boost * god_n * kinds
                    self.last_detail["god_wait_boost"] = (god_n, kinds, copies)
            goodshape_rows.append((int(copies), _goodshape_proxy(counts, situation.hand.meld_count), score))
            if best is None or copies > best[0]:
                best = (copies, score)
        if best is not None and self.config.goodshape_tolerance > 0 and goodshape_rows:
            top = max(row[0] for row in goodshape_rows)
            pool = [row for row in goodshape_rows if row[0] >= self.config.goodshape_tolerance * top]
            pick = max(pool, key=lambda row: (row[1], row[0]))
            if pick[2] is not best[1]:
                self.last_detail["goodshape_pick"] = pick[2].describe()
            best = (pick[0], pick[2])
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
            block_value = shanten_module.shape_value(
                counts,
                hand.meld_count,
                edge_partial_weight=(
                    self.config.edge_partial_weight or None
                ),
                keep_extra_pairs=self.config.keep_extra_pairs,
            )
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
        god_penalty = 0.0
        if tile == GOD:
            # 财神线 v2：**落后态换值**（见 `PolicyConfig.god_penalty_behind`）；正常态保持恒定禁忌。
            # **覆盖口径（A 2026-10-05 23:35 修正）**：不加「进度落后」这个门——
            # 因为空干预门实测该门在真机 200 点上**触发 0 次**（分歧 0.0%），
            # 而 C 19:32 的证据桶是**「有财神」本身就分歧**（1 向听 / 2 向听 / 3+ 向听、1-5 巡与 6-10 巡都有），
            # 与 B' 19:35「财飘只有 bot 的一半」同向 ⇒ **禁忌应当在「持财神」时整体降低**，而不是只在落后时降低。
            god_penalty = (
                self.config.god_penalty_behind
                if situation.hand.counts[GOD] >= 1
                else self.config.god_discard_penalty
            )
        # 庄家局的喂牌权重单独缩放：庄闲赔付是 8 倍不对称，而决策层此前完全不分庄闲。
        feed_scale = self.config.feed_weight
        if (
            self.config.dealer_feed_scale != 1.0
            and situation.table.dealer_seat == situation.seat
        ):
            feed_scale *= self.config.dealer_feed_scale
        # **S3 v2：局况姿态接在「喂牌代价」上**（A 2026-10-05 19:30 裁决）。
        #
        # 为什么从 v1 的接入点（弃胡阈值）搬到这里：v1 的触发基数只有 **≈0.05 次/座·局**
        # （插桩实测：48 局里只调用 10 次、非中性 0 次）⇒ 在那个点上测 1% 量级的效应**功效为零**，
        # 那是一次「没有功效的实验」。而 `_score_discard` 对**每个出牌候选**都会跑（每决策 ~14 次调用），
        # 边界态（余局≤1 + 分差≤29）占出牌决策点 ~13% ⇒ **可测**。
        #
        # 语义：**守（领先，scale > 1）⇒ 喂牌惩罚加重**（不喂人、稳名次）；
        # **搏（落后，scale < 1）⇒ 喂牌惩罚减轻**（敢打危险牌换速度）。其余/并列/拿不到比分 ⇒ 1.0（逐位等于 v5）。
        if self.config.standing_feed_apply:
            feed_scale *= self._standing_scale(situation)

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
            self._meld_trace(gate="commitment-pair", current=current)
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
                after_shanten = self._meld_after_shanten(situation, offered, action)
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
                self._meld_trace(gate="pair-route", current=current)
                return fallback
            options: list[dict] = []
            best_legacy: tuple[tuple[int, int, int], Action] | None = None
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
            if accept_equal and self.config.meld_conditional:
                accept_equal = _meld_cell_allows(situation, current)
            for action in actions:
                if action.kind not in (PENG, CHI):
                    continue
                after_legacy = self._meld_after_shanten(situation, offered, action)
                entry: dict = {
                    "kind": action.kind,
                    "tiles": list(action.tiles),
                    "after_shanten": after_legacy,
                }
                options.append(entry)
                if after_legacy is None:
                    entry["reject"] = "no_shanten"
                    continue
                if after_legacy > current:
                    entry["reject"] = f"after{after_legacy}>current{current}"
                    continue
                if after_legacy == current and not accept_equal:
                    entry["reject"] = "equal-not-allowed"
                    continue
                # 同向听时优先碰：吃全局最多 2 副，碰不占这个额度
                peng_rank = 0 if action.kind == PENG else 1
                # 同向听的多种吃法之间：按进张宽度次排序（默认关 ⇒ ukeire_rank 恒 0，
                # 逐位等于旧行为）。仅对 CHI 生效（PENG 的结构固定，无需次排序）。
                ukeire_rank = 0
                if self.config.meld_chi_tiebreak and action.kind == CHI:
                    ukeire_rank = -self._meld_ukeire_copies(
                        situation, offered, action, after_legacy
                    )
                    # **注意字段名**：`_meld_ukeire_copies` 返回的是
                    # `shape_value × 1000`（不是「张数」）——它用形质值而非精确进张宽度
                    # 做次排序。命名成 `meld_shape` 以免读日志的人把它当进张数。
                    entry["meld_shape"] = -ukeire_rank
                rank = (after_legacy, peng_rank, ukeire_rank)
                entry["rank"] = list(rank)
                if best_legacy is None or rank < best_legacy[0]:
                    best_legacy = (rank, action)
            if best_legacy is None:
                self.last_reason = f"不副露：向听 {current} 无改善"
                self._meld_trace(
                    gate="strict-pass",
                    current=current,
                    options=options,
                    extra={"tolerance": str(tolerance), "accept_equal": accept_equal},
                )
                return fallback
            chosen_shanten = best_legacy[0][0]
            suffix = "（向听不变，放宽档）" if chosen_shanten == current else ""
            self.last_reason = (
                f"副露 {best_legacy[1].describe()}：向听 {current}→{chosen_shanten}{suffix}"
            )
            self._meld_trace(
                gate="accept",
                current=current,
                options=options,
                chosen=best_legacy[1],
                extra={"tolerance": str(tolerance), "accept_equal": accept_equal},
            )
            return best_legacy[1]

        pair_route, meld_route = routes.evaluate(situation, base_score=self.config.base_score)
        self.last_detail["routes"] = [pair_route.describe(), meld_route.describe()]

        best: tuple[int, Action, float] | None = None
        route_options: list[dict] = []
        for action in actions:
            if action.kind not in (PENG, CHI):
                continue
            after_shanten = self._meld_after_shanten(situation, offered, action)
            route_entry: dict = {
                "kind": action.kind,
                "tiles": list(action.tiles),
                "after_shanten": after_shanten,
            }
            route_options.append(route_entry)
            if after_shanten is None or after_shanten >= current:
                route_entry["reject"] = "after>=current"
                continue
            plan = self._meld_plan(situation, offered, action)
            if plan is None:
                route_entry["reject"] = "no_plan"
                continue
            hypothetical, after_counts = plan
            _, after_value = routes.evaluate(
                situation,
                base_score=self.config.base_score,
                counts=after_counts,
                melds=hypothetical,
            )
            route_entry["after_value"] = round(after_value.value, 2)
            if pair_route.feasible and after_value.value <= pair_route.value:
                route_entry["reject"] = "pair-route-better"
                continue
            if best is None or after_value.value > best[2]:
                best = (after_shanten, action, after_value.value)
        if best is None:
            self._meld_trace(gate="route-aware-pass", current=current, options=route_options)
            self.last_reason = (
                f"不副露：向听 {current} 无改善，或保留七对路线更优"
                f"（七对 {pair_route.value:.1f} / 副露 {meld_route.value:.1f}）"
            )
            return fallback
        self.last_reason = (
            f"副露 {best[1].describe()}：向听 {current}→{best[0]}，"
            f"副露路线期望 {best[2]:.1f} 超过七对 {pair_route.value:.1f}"
        )
        self._meld_trace(
            gate="route-aware-accept", current=current, options=route_options, chosen=best[1]
        )
        return best[1]

    def _meld_after_shanten(self, situation: Situation, offered: int, action: Action) -> int | None:
        """某个具体吃/碰动作执行后的最小向听。

        **与旧实现的差别**：开启 `meld_chi_best` 时把 ``action.tiles``（具体吃哪两张）
        传下去，而不是让每个 CHI 选项共用 `chi_combinations()[0]`。关闭时保持旧行为（逐位等于 v5）。
        """
        chi_tiles = action.tiles if (action.kind == CHI and self.config.meld_chi_best) else None
        return self._shanten_after_meld(situation, offered, action.kind, chi_tiles)

    def _meld_ukeire_copies(
        self, situation: Situation, offered: int, action: Action, after_shanten: int
    ) -> int:
        """某个具体吃法之后、打出最优一张时的**剩下张数**（`cheap_ukeire`，次排序用）。

        只在同级向听的多种吃法之间打破平局（见 `PolicyConfig.meld_chi_tiebreak`），
        不做进动作链、不改变是否副露。开销 ~1.5ms/候选。
        """
        plan = self._meld_plan(situation, offered, action)
        if plan is None:
            return 0
        _, after_counts = plan
        meld_after = situation.hand.meld_count + 1
        # 吃后是**未打牌**态（张数 = 14 − 3×副露）：先选出「打哪张能留下最好的形」——
        # 判据即 `shape_value`（已含向听项 `2×sets + len(taken)`，并正则化到块均权重）。
        # 用 `shape_value` 而非 `cheap_ukeire`：后者内部 `quick_shanten` 贪心，实测连
        # 「两面 vs 愚形」这种量级都分不出（两种吃法都给同样的 29 张）。
        base = max(
            (self._drop_one(after_counts, tile) for tile in range(tiles.TILE_KINDS) if after_counts[tile] > 0),
            key=lambda candidate: shanten_module.shape_value(
                candidate, meld_after, keep_extra_pairs=self.config.keep_extra_pairs
            ),
            default=after_counts,
        )
        return round(1000 * shanten_module.shape_value(
            base, meld_after, keep_extra_pairs=self.config.keep_extra_pairs
        ))

    @staticmethod
    def _drop_one(counts: list[int], tile: int) -> list[int]:
        """打出一张后的暗手计数。"""
        out = list(counts)
        out[tile] -= 1
        return out

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
        # **多选择吃法**：开启 `meld_chi_best` 时按 `action.tiles`（具体吃哪两张）算；
        # 否则保持旧行为（`combos[0]`）——旧代码在这里也有同一个 `combos[0]` 缺陷。
        if self.config.meld_chi_best and action.tiles:
            first, second = action.tiles
            if counts[first] < 1 or counts[second] < 1:
                return None
        else:
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

    def _meld_trace(
        self,
        *,
        gate: str,
        current: int,
        options: list[dict] | None = None,
        chosen: Action | None = None,
        extra: dict | None = None,
    ) -> None:
        """把**响应窗口的候选明细**写进 ``last_detail``（随 `logs/*.jsonl` 的 `decision.made` 落盘）。

        **为什么需要它**（2026-10-08 23:35 用户交办）：吃法类改动的盲点全在这里。今晚要回答
        「这手为什么不吃了 `8t9t`」，必须**重放整局**才能拿到「各吃法的吃后向听」——
        因为日志只记了 `reason` 的自由文本（如「不副露：向听 2 无改善」），
        **看不出每个吃法各自算出来是多少**。补上后 `grep decision.made` 直接可查：
        每个候选的 `after_shanten`、吃后进张宽度、以及闸门为什么放行/否决。

        **成本与安全**：记的全部是**已经算出来**的量（`_meld_after_shanten` 的返回值、
        已算过的 `ukeire_rank`），**不新增任何决策计算、不改变出牌**；
        只往 `last_detail` 加一个键，老消费者（`audit_response.py` / `read_decision_traces.py`）
        读的是 `reason`/`candidates`，不受影响。
        """
        item: dict = {"gate": gate, "current_shanten": current}
        if options is not None:
            item["options"] = options
        if chosen is not None:
            item["chosen"] = {"kind": chosen.kind, "tiles": list(chosen.tiles)}
        if extra:
            item.update(extra)
        self.last_detail["meld"] = item

    def _is_pair_route(self, situation: Situation) -> bool:
        """粗糙的七对门槛（仅在 ``route_aware`` 关闭时使用）。"""
        if situation.hand.meld_count:
            return False
        pairs = sum(amount // 2 for amount in situation.hand.counts if amount >= 2)
        return pairs >= self.config.pair_route_pairs

    def _shanten_after_meld(
        self,
        situation: Situation,
        offered: int,
        kind: str,
        chi_tiles: tuple[int, ...] | None = None,
    ) -> int | None:
        """吃/碰后（并打出多余一张）能达到的最小向听。

        ``chi_tiles``：**具体吃法**（两张手牌）。给了就用它算——同一张 `offered` 可能有多
        种吃法，不同吃法的「吃后最小向听」可以**不同**（实测 39.8% 的多选择局面如此），
        不给则退回旧行为（取 `chi_combinations()[0]`）。
        """
        hand = situation.hand
        counts = list(hand.counts)
        if kind == PENG:
            if counts[offered] < 2:
                return None
            counts[offered] -= 2
        else:
            if chi_tiles is not None:
                first, second = chi_tiles
            else:
                combos = chi_combinations(counts, offered)
                if not combos:
                    return None
                first, second = combos[0]
            if counts[first] < 1 or counts[second] < 1:
                return None
            counts[first] -= 1
            counts[second] -= 1
        return _best_after_drop(counts, hand.meld_count + 1)


__all__ = ["DiscardScore", "HeuristicDecider", "Mode", "PolicyConfig", "cheap_ukeire"]
