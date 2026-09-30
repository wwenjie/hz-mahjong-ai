"""决策器版本库（champion / challenger）。

**用途**：保留历史版本，使**任意两版可以直接对打**。做法是把每个「冠军」的**变体开关**
冻结成一个具名版本，`--decider v1` / `--decider v2` 即可指名：

    uv run python tools/ab_test.py --treatment v3 --baseline v2

只冻结**变体开关**（`tiebreak` / `meld_tolerance` / `chase_baotou` / `feed_weight` …），
**不冻结** `base_score` / `you_cai_bi_kao` —— 那两个由运行时从服务端注入
（`HeuristicDecider.configure()`）。冻结它们会让版本在真机上与服务端实际规则不符。

**纪律（重要）**：每次改默认档**必须**同时新增一个版本快照。否则「旧默认档」的行为就此丢失，
以后再也无法与新默认直接对打，只能拿一个近似的档位当替身——而 `blocks` 那次就是这么补救的。
新增时写清日期与改动理由。

**它不替代 `DECIDERS`**：`DECIDERS` 是「尚未胜出的实验档位」，本文件是「已胜出并冻结的版本」。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .policy import HeuristicDecider, Mode, PolicyConfig


@dataclass(frozen=True, slots=True)
class Version:
    id: str
    date: str
    note: str
    knobs: Mapping[str, Any] = field(default_factory=dict)


VERSIONS: tuple[Version, ...] = (
    Version(
        id="v1",
        date="2026-09-26",
        note=(
            "本轮之前的线上基线：同向听用骨架厚度（blocks）做次排序、"
            "chase_baotou 开启、喂牌权重 3.0、吃碰闸门 strict。"
        ),
        knobs={"tiebreak": "blocks"},
    ),
    Version(
        id="v2",
        date="2026-09-27",
        note=(
            "exact-ukeire：同向听改用**精确**进张次排序（只在打完仍 1 向听的前 2 候选上算，"
            "带 0.6 秒墙钟上限）。两个独立种子共 2000 配对场复现显著："
            "名次分 +0.285（t 4.03）· 胡次数 +0.088（t 4.46）· 总得分 +1.525（t 2.55）。"
            "真机侧机制指标同向：出牌后到听率 22.0% → 23.8%（对手 +0.7，差分 +1.1）。"
            "**注（09-28 追记）**：这条增益**只来自向听 1 的那些局面**——向听 0 时 "
            "`shanten.ukeire` 返回空元组、调用方把「各候选都是 0」当平局，"
            "所以听牌时的次排序其实一直没生效（由 v3 修掉，见下）。"
        ),
        knobs={},
    ),
    Version(
        id="v3",
        date="2026-09-28",
        note=(
            "wait-aware tenpai：**听牌时按「可见听口张数」选牌**（并且不截断候选面），"
            "修掉 v2 里那个静默失效——`shanten.ukeire` 在 `current == 0` 时返回空元组"
            "（向听不能再降），而 `_break_ties_by_ukeire` 把「每个候选 copies 都是 0」"
            "当成平局，于是出牌退化成按 `total`（骨架厚度 + 喂牌）选，**完全不看听口**。"
            "这也解释了为什么 `ukeire-wide`（候选面 2→6）测出来是平的：听牌时候选数无关紧要。"
            "\n机制证据（先于 A/B，`tools/analyze_wait_ceiling.py`）："
            "我们听牌时的听口可见张数离同一手牌的上限平均差 **0.96 张（6.7%，分位 27.6%）**，"
            "对手只差 **0.20 张（1.1%，分位 5.2%）**；修后自对弈实测 regret **6.7% → 0.0%**、"
            "听口 **+12.3%**（14.03 → 15.76 张）、且**从未在听牌时打财神**（0.0%）。"
            "\nA/B（四座位旋转、同批牌、两种子）："
            "名次分 **+0.235（t 1.9）/ +0.446（t 3.8）**· 胡次数 **+0.077（t 2.2）/ "
            "+0.117（t 3.5）**· 总得分 +0.229（t 0.2）/ **+3.367（t 3.3）**。"
            "**当前冠军档。**"
        ),
        knobs={"wait_aware_tenpai": True},
    ),
    Version(
        id="v4",
        date="2026-09-30",
        note=(
            "shape-blocks：同向听次排序改用**有分辨力的加权形质值**（`shape_value`）+ 候选按"
            "**形质**取面（`ukeire_order=\"blocks\"`）+ 精确进张的适用向听**从 1 开到 3**。"
            "\n**三项各自的贡献（2026-09-30 10:30 拆解后更正，原先写成「合取」是错的）**："
            "把缺失的两个格子补齐后看，三项是**大致可加**的，而且**门是主项**——"
            "`ukeire-early`（只开门到 3、键与排序都不变）+0.154/+0.106（n=2, 功效弱）；"
            "`blocks-gate3`（门 3 + blocks 排序、**无** shape_value）**+0.205(t1.93, n2)**；"
            "`shape-gate3`（门 3 + shape_value）**+0.235(t2.31, n3)**；"
            "三者叠加即本条 **+0.2732（t+4.66, n=10, 10/10 同向）**。"
            "**原先的「合取」说法来自一个不公平的比较**：`shape`(+0.119) 与 `ukeire-hand`(+0.086) "
            "都只开**门 1**，也就是把主项拿掉了再比 ⇒ 当然显得「单用不行」。"
            "⇒ 正确的读法是「**门 3 贡献约 +0.15，两个排序键各再加 +0.05~0.08**」，"
            "这也解释为什么再开门（`ukeire-deep` 门 5）没有额外收益。"
            "病根是 `shanten.quick_blocks` 的裁剪 `partials = min(partials, slots − sets)` **恒饱和**"
            "（同向听内全并列占比 77.2%/89.2%/93.5%，按副露 0/1/2）⇒ 形质项在候选之间没有分辨力，"
            "中段出牌实际由写死的喂牌表（字 0.4/边 0.6/中 1.0）单独决定（三路独立测量一致："
            "97.7%/97.5%/97.25%）。`shape_value` 是给这一项**恢复分辨力**的药。"
            "\n**A/B（对照 v3，四座位旋转、同批牌、10 个种子 × 120 场 = 4800 配对场）**："
            "名次分 **+0.2732（t +4.66，10/10 为正）**· 胡次数 **+0.0870（t +5.27）**"
            "≈ **胡率 +1.09 个百分点**· 番数总和 +0.0823（t +3.41）· 总得分 +1.156（t +2.37）·"
            "白板数 +0.0035（t +0.20，**没动** ⇒ 不是靠攒财神换分）。"
            "\n**真机（2026-09-30 起）**：与 v3 **交错轮换**采集（`--decider v3,v4`），"
            "不是时代对比——时代对比会被房间/对手池/时段混杂污染（v2-vs-v3 那份分析里已注明）。"
            "**已知缺口（诚实记录）**：`shape_value` 的**机制门尚未独立复算**"
            "（该仪器由 agent-c 承担，2026-09-29 17:05 派下、换档时未交付）。"
            "所以本档的采纳依据是**自对弈 A/B 的强证据**，机制解释仍是待验状态——"
            "这与 v3 当年的顺序相反（v3 是机制先行：regret 6.7%→0 才谈 A/B）。"
            "若机制门显示 `shape_value` 近乎不生效，需回来重审本档。"
        ),
        knobs={
            "wait_aware_tenpai": True,
            "shape_value": True,
            "ukeire_order": "blocks",
            "ukeire_max_shanten": 3,
        },
    ),
    Version(
        id="v5",
        date="2026-10-01",
        note=(
            "cand3：把**精确进张比较的候选面从 2 张放宽到 3 张**（`ukeire_candidates=2→3`），叠在 v4 之上。"
            "\n**为什么是这个开关**：`_break_ties_by_ukeire` 先按 `total` 排序、再**截断到前 "
            "`ukeire_candidates` 张**才做精确进张比较 ⇒ 「进张更多但排序稍后」的牌**从来没进过比较**。"
            "这正是 agent-c 2026-09-29 复核的 ② 条；但当年 `ukeire-wide`（候选面 2→6）测平——"
            "因为那时 v4 的形质分辨率（`shape_value`）与门 3 还没装上，**候选池是按一个退化的键排序的，"
            "加宽池子等于随机采样**。装上之后这条老主张才兑现。"
            "\n**证据（对照 v3=`tenpai-wait`，四座位旋转、同批牌）**："
            "`v4-cand3` **n=8 种子、合并名次分 +0.538（t +7.69）、8/8 种子为正**；"
            "胡次数 +0.1623（t +8.24）≈ **胡率 +2.03pp**（v4 是 +1.09pp）。"
            "\n**与 v4 的增量（配对口径，同 seed ⇒ 同一副牌）**：8 个共同种子逐种子差 "
            "`−0.011 +0.254 +0.240 +0.414 +0.528 +0.348 +0.139 +0.129` ⇒ "
            "**配对均值 +0.255（se 0.061、t +4.18、7/8 为正）**，胡率口径 **≈ +0.94pp**。"
            "与「两个实验对同一基准作差」的间接估计（+0.265、t2.90）吻合。"
            "\n`v4-cand4`（2→4）的边际与 cand3 打平（+0.256、t3.53）⇒ **取 cand3，少算一张候选、延迟更低**。"
            "\n**门 3（空干预防线）**：与 v4 在真机决策点上分歧 **10.0%**（12/120），且是实质不同的牌 ⇒ 过门。"
            "\n**⚠ 已知风险与护栏（换档时写下）**：候选面 +1 张 = 每张候选多算一次精确进张（42–157ms）。"
            "v4 的真机 p99 已是 **851ms / 预算 1800ms**，v5 大概率升到 ~1000–1200ms。"
            "**承诺：换档后立即实测真机 `elapsed_ms` 分布，若 p99 > 1500ms 或出现任何超预算事件，"
            "立即回退成 `--decider v3,v4`（一条命令），不等授权。**"
        ),
        knobs={
            "wait_aware_tenpai": True,
            "shape_value": True,
            "ukeire_order": "blocks",
            "ukeire_max_shanten": 3,
            "ukeire_candidates": 3,
        },
    ),
)

BY_ID: dict[str, Version] = {version.id: version for version in VERSIONS}


def build(version_id: str, mode: Mode) -> HeuristicDecider:
    """按版本号构造决策器。未知版本号抛 ``KeyError``（由调用方转成启动错误）。"""
    return HeuristicDecider(PolicyConfig.for_mode(mode, **dict(BY_ID[version_id].knobs)))


def is_version(name: str) -> bool:
    return name in BY_ID


def describe() -> str:
    return "\n".join(
        f"  {version.id}  {version.date}  {version.note.splitlines()[0]}"
        for version in VERSIONS
    )
