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
    "ukeire_candidates",
    "ukeire_max_shanten",
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
        if survival > threshold and current.chain_count < 6:
            self.last_reason = (
                f"弃胡飘：番 {current.fan}→{current.fan * 2}，生存 {survival:.2f} > 阈值 {threshold:.2f}"
            )
            return Action(DISCARD, tile=GOD)
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
        scores = sorted(
            (self._score_discard(situation, action) for action in candidates),
            key=lambda item: item.total,
            reverse=True,
        )
        best = scores[0]
        if self.config.tiebreak in ("ukeire", "exact-ukeire"):
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
        exact = self.config.tiebreak == "exact-ukeire"
        if exact and top_shanten > self.config.ukeire_max_shanten:
            return None
        if exact:
            tied = tied[: max(1, self.config.ukeire_candidates)]
        visible = shanten_module.visible_counts(
            situation.hand.counts,
            [meld.tiles for meld in situation.all_melds],
            situation.discards,
        )
        deadline = time.monotonic() + EXACT_UKEIRE_BUDGET_SEC if exact else None
        best: tuple[int, DiscardScore] | None = None
        for score in tied:
            counts = list(situation.hand.counts)
            counts[score.tile] -= 1
            if exact:
                if deadline is not None and time.monotonic() > deadline and best is not None:
                    self.last_detail["tiebreak_timeout"] = True
                    break
                entries = shanten_module.ukeire(
                    counts, situation.hand.meld_count, visible=visible
                )
                copies = sum(copy for _, copy in entries)
            else:
                _, copies = cheap_ukeire(counts, visible)
            if best is None or copies > best[0]:
                best = (copies, score)
        if best is not None:
            self.last_detail["tiebreak"] = (
                f"向听 {top_shanten} 并列 {len(tied)} 张，"
                f"{'精确' if exact else '廉价'}进张 {best[0]} 张"
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
        blocks = shanten_module.quick_blocks(counts)
        block_value = 2 * blocks[0] + blocks[1]

        risks = self._risks(situation)
        threat = sum(item.ready_probability for item in risks)
        feed = risk.visible_need(tile) * threat
        god_penalty = self.config.god_discard_penalty if tile == GOD else 0.0
        # 庄家局的喂牌权重单独缩放：庄闲赔付是 8 倍不对称，而决策层此前完全不分庄闲。
        feed_scale = self.config.feed_weight
        if (
            self.config.dealer_feed_scale != 1.0
            and situation.table.dealer_seat == situation.seat
        ):
            feed_scale *= self.config.dealer_feed_scale

        if not self.config.route_aware and self.config.commitment is Commitment.NONE:
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

        total = (
            self.config.value_weight * best.value
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
