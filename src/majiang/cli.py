"""命令行入口。

令牌来源优先级：``--token-env`` 指定的环境变量 > ``--env-prefix`` 收集到的多个环境变量
> ``--token`` 直接传参。推荐用环境变量，避免令牌出现在命令行与进程列表里。

单身份::

    uv run python -m majiang --token-env TEST_TOKEN_QINGLONG --duration 600

多身份（联调用，一个进程内各身份各自独立限速）::

    uv run python -m majiang --env-prefix TEST_TOKEN_ --duration 600
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from .client.api import KNOWN_GUIDE_VERSION, GuideVersion, PlatformApi
from .client.errors import ApiError
from .client.transport import HttpTransport
from .runtime.decider import Decider, FirstLegalDecider
from .runtime.engine import Runtime, RuntimeOptions
from .runtime.logging import DEFAULT_LOG_DIR
from .strategy.policy import HeuristicDecider, Mode, PolicyConfig
from .strategy import versions
from .strategy.opponent import load_or_none
from .strategy.search import SearchConfig, SearchDecider
from .strategy.value import ValueDecider, ValueModel, ValueModelError

DEFAULT_SERVER = "https://10.240.169.190:18080"
# 价值模型默认路径；缺失或版本不兼容时 ValueDecider 会退回启发式并给出提示
VALUE_MODEL_PATH = "models/value_model.json"
OPPONENT_MODEL_PATH = "models/opponent_model.json"


def _value_decider(mode: Mode, path: str) -> Decider:
    """价值模型驱动；模型不可用时退回启发式（绝不因模型缺失而中止）。"""
    heuristic = HeuristicDecider(PolicyConfig.for_mode(mode))
    try:
        return ValueDecider(model=ValueModel.load(path), heuristic=heuristic)
    except ValueModelError as exc:
        print(f"[警告] 价值模型不可用（{exc}），退回启发式", file=sys.stderr)
        return heuristic


def guide_version_warning(info: GuideVersion, known: int = KNOWN_GUIDE_VERSION) -> str | None:
    """给定指南版本信息，返回告警文本；不需要告警时返回 ``None``（纯函数，便于测试）。

    三种需要人工注意的情况：版本号无法识别、平台版本比代码依据的新且含破坏性变更、
    平台版本更新但未标记破坏性变更（仍值得核对端点与快照字段）。
    """
    if info.version <= 0:
        return "指南版本响应无法识别（version 缺失或非正数），可能是破坏性变更，请人工核对端点与快照字段"
    if info.version <= known:
        return None
    head = (
        f"平台接入指南已更新到 v{info.version}（代码依据 v{known}，"
        f"更新于 {info.updated_at or '未知时间'}）"
    )
    breaking = info.breaking_since(known)
    if not breaking:
        return head + "\n  （未标记破坏性变更，但仍建议核对端点与快照字段）"
    return head + "".join(f"\n  [破坏性变更] {line}" for line in breaking)


def check_guide_version(server: str, *, known: int = KNOWN_GUIDE_VERSION) -> None:
    """启动自检（tasks.md 1.4）：核对平台指南版本并对破坏性变更告警。

    **绝不因自检失败而中止启动**——比赛不能因为一个诊断端点不可达就放弃。
    """
    try:
        info = PlatformApi(HttpTransport(server)).guide_version()
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 指南版本自检失败（{type(exc).__name__}: {exc}），已跳过", file=sys.stderr)
        return
    warning = guide_version_warning(info, known)
    if warning is None:
        print(f"指南版本自检通过：平台 v{info.version}，代码依据 v{known}", file=sys.stderr)
        return
    print(f"[警告] {warning}", file=sys.stderr)
    print(f"  核对后请更新 majiang/client/api.py 的 KNOWN_GUIDE_VERSION", file=sys.stderr)


# 平台 /state 限速为 16/s。取 14 留出窗口余量：实测发现突发容量也会影响判定，
# 因此不仅压速率，令牌桶的突发上限也压到 2。
PLATFORM_STATE_RATE_PER_SEC = 14.0

DECIDERS: dict[str, Callable[[Mode], Decider]] = {
    # 默认档位：`PolicyConfig` 的默认 tiebreak 已是 "exact-ukeire"（见其注释）
    "heuristic": lambda mode: HeuristicDecider(PolicyConfig.for_mode(mode)),
    # 对照：同向听用**骨架厚度**次排序（2026-09-26 之前的默认行为）。
    # 2000 配对场显示它显著差于精确进张，保留仅为后续对照。
    "blocks": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="blocks")
    ),
    # 同向听改用「进张最多」做次排序（tasks.md 5.4 的优化方向）
    # **注意**：这个档位用的是廉价估计，实测与精确进张只有 14.7% 的选择一致
    # （tools/analyze_ukeire_fidelity.py），因此它当年测出的「无增益」不可采信。
    "ukeire": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="ukeire")
    ),
    # 同向听改用**精确**进张做次排序。只在向听 ≤2 且并列候选前 3 张上计算，
    # 带 0.6 秒墙钟上限；出牌预算 1800 ms。
    "ukeire-exact": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire")
    ),
    # 扩大精确进张的候选面（2 -> 6 张）。实测我们听口窄于对手约 21%，
    # 而候选按 ``total`` 排序、同向听时被喂牌代价主导，好听的牌可能被提前截掉。
    # 墙钟上限仍是 0.6 秒，故成本有界。
    "ukeire-wide": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", ukeire_candidates=6)
    ),
    # 确定化前瞻搜索（tasks.md 5.15）。samples/top_k 越小越快
    "search": lambda mode: SearchDecider(
        HeuristicDecider(PolicyConfig.for_mode(mode)),
        SearchConfig(samples=6, top_k=2),
    ),
    "search-deep": lambda mode: SearchDecider(
        HeuristicDecider(PolicyConfig.for_mode(mode)),
        SearchConfig(samples=16, top_k=3),
    ),
    # 开启路线分叉（tasks.md 5.6）。实测为负收益，仅在实验时使用
    "route-aware": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, route_aware=True)
    ),
    # 实验档位：绝不打出财神（验证能否把爆头/财飘链制造出来）
    "preserve-god": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, preserve_god=True)
    ),
    # 实验档位：按纯真牌算向听，逼策略做 4 组自然面子以谋求爆头（＋留财神）
    "natural": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, preserve_god=True, natural_route=True)
    ),
    # 对照：关闭「弃胡求爆头」（用于量化该决策的期望值）
    "no-chase": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, chase_baotou=False)
    ),
    # 吃碰闸门放宽档（tasks.md 5.5）：接受「向听不变且未听牌」的吃碰。
    # 依据是实测我们副露 0.591/局 vs 对手 1.093/局（1.85 倍）。
    "meld-equal": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, meld_tolerance="equal")
    ),
    # 只在向听 >=2（离听牌还远）时接受向听不变的副露。依据是 agent B 的进度曲线：
    # 差距从第 2 摸起就单调扩大，吃碰的价值集中在早段。
    "meld-equal-early": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, meld_tolerance="equal-early")
    ),
    # 喂牌权重下调档。本平台**没有点炮**，喂牌只让对手吃碰加速，不该压过手牌质量；
    # 默认 3.0 是从「有点炮」的直觉来的。
    "feed-low": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, feed_weight=1.0)
    ),
    # **符号镜像档**：与 `feed-low` 反方向的同一个旋钮。存在的唯一目的是做**仪器检定**——
    # 已完成的 16 个单旋钮档位里 14 个都是「总得分小幅为正」（合并均值约 +0.40，
    # 而单种子标准误约 0.4–0.7），这形状要么说明「我们的自对弈场地（三个自己的复制品）
    # 对任何偏离都给正分」，要么说明这些旋钮都真的有效。
    # `feed-low` 与 `feed-high` 同时为正 ⇒ 是前者（场地效应），那么**所有 ±0.5 量级的
    # 自对弈正号都不能作为采纳依据**；一正一负 ⇒ 符号有意义，可以继续按效应量筛。
    # 这与「dealer-soft 与 dealer-hard 两个反方向扰动都为正」的形状一致，需独立确认。
    "feed-high": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, feed_weight=6.0)
    ),
    # 价值模型驱动（tasks.md 5.15）：用自对弈学到的价值函数给出牌打分
    "value": lambda mode: _value_decider(mode, VALUE_MODEL_PATH),
    # 对手听牌模型驱动的风险（tasks.md 6B）：模型不可用时自动回退手写启发式
    "risk": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode), risk_model=load_or_none(OPPONENT_MODEL_PATH)
    ),
    # 决赛特化（tasks.md 5.13 + agent B 的复核）。**计分口径**：三轮的排序键依次是
    # 总得分 -> 名次分(+3/+1/-1/-3) -> 白板获取数，而**决赛只用总得分**（同分无限加赛）。
    # 也就是说名次分连晋级轮都只是**次级键**，而我们最近的改动都在优化名次分/胡次数这类
    # 低方差代理量——对「测量功效」是对的，对「决赛目标函数」是偏保守的。
    # 番数连乘、分布重尾（可达 512），故决赛应更偏向追高番。
    "final": lambda mode: HeuristicDecider(PolicyConfig.for_mode(Mode.FINAL)),
    "final-plus": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(Mode.FINAL, piao_threshold_scale=0.7, feed_weight=1.0)
    ),
    # 庄闲敏感档位（tasks.md 5.3）。实测（tools/analyze_dealer.py）我们庄家胜率 27.07%
    # vs 对手 30.21%，庄家局每局净分差 -1.59 是闲家局差（-0.90）的 1.8 倍（x8 赔付），
    # 而决策层此前完全不分庄闲。两个方向都测——我对符号没有先验把握：
    # 庄家的「收益:风险」是 24:8=3:1，闲家是 10:1，比值更低反而说明庄家更该谨慎；
    # 但庄家局的绝对收益也最大，两种论证方向相反，交给数据。
    "dealer-soft": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, dealer_feed_scale=0.5)
    ),
    "dealer-hard": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, dealer_feed_scale=2.0)
    ),
    # 追番更激进（依据 agent B 的 P2 定量）：前 10% 的局贡献 27.3% 总分，
    # **追番翻倍只要「输掉的概率 < 51%」就净赚**，而我们的弃胡自补率实测 88.7%，
    # 远在盈亏线上方 —— 说明当前阈值（final 档也只缩到 0.85）过于保守。
    # 这一档把阈值缩到 0.5，用于同时检验晋级轮与决赛。
    "chase-more": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, piao_threshold_scale=0.5)
    ),
    # 把精确进张的适用向听从 1 放宽到 3：**前中期也按进张选牌**。
    # 现状是 shanten >=2 时出牌完全由「骨架厚度 + 喂牌代价」决定，而
    # 「第 4 摸均向听落后 0.16」正是这段决定的。这是从未测过的结构性缺口
    # （`ukeire-wide` 只改了候选数、没改适用向听，所以它测的不是这件事）。
    "ukeire-early": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", ukeire_max_shanten=3)
    ),
    # 适用向听包围测试（gate=2）：与 ukeire-g3 一起定位「进张该从几向听起生效」。
    "ukeire-g2": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", ukeire_max_shanten=2)
    ),
    # 适用向听包围测试（gate=5）：与 ukeire-g3 一起定位「进张该从几向听起生效」。
    "ukeire-g5": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", ukeire_max_shanten=5)
    ),
    # 候选面**按手牌质量排序后再截断**（默认按 total 排，而同向听时 total 被喂牌代价主导）。
    # 与 ukeire-wide 互补：那个改的是「取几张」，这个改的是「按什么顺序取」。
    "ukeire-hand": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", ukeire_order="blocks",
                              ukeire_max_shanten=3)
    ),
    # **这个家族的最强形态**：适用向听 5 + 候选按手牌质量排序 + 候选面 4。
    # 意义在于判据是一次性的：若最强形态也无效，则「适用门」家族死掉，
    # 直接转统一期望得分；若有效，再逐项拆解是哪个因子在起作用。
    "ukeire-deep": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", ukeire_max_shanten=5,
                              ukeire_order="blocks", ukeire_candidates=4)
    ),
    # 听牌后按**可见听口张数**选牌（并放开候选面）。**这是一个 bug 级发现的修补**：
    # `shanten.ukeire` 在向听 0 时返回空元组，而 `_break_ties_by_ukeire` 把「各候选都是 0」
    # 当平局，于是**听牌时的精确进张次排序其实什么都没做**，出牌完全按
    # `total = -10×向听 + 骨架厚度 - 3×喂牌 - 财神罚` 决定，**根本不看听口**。
    # 这也解释了为什么 `ukeire-wide`（候选面 2→6）测出来是平的：听牌时候选数无关紧要。
    # 实测（`tools/analyze_wait_ceiling.py`，v2 时代 150 文件 / 1298 个听牌点）：
    # 我们的听口可见张数离上限平均差 0.96 张（6.7%，分位 27.6%），对手只差 0.20 张（1.1%）。
    "tenpai-wait": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True)
    ),
    # 对照：同一条改动 + **同时放开向听 1 的候选面**（2 → 6）。用于分离
    # 「听牌口径修正」与「向听 1 算得更多」两个因子。
    "tenpai-wait-6": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              ukeire_candidates=6)
    ),
    # 向听 1 按两拍值排序（**叠在 v3 之上**，所以这条臂直接与冠军档对比）。
    # 实测依据：真机 1 向听出牌点里 **39.2% 的局面我们的选择不是两拍最优、相对 regret 11.6%**
    # （n=74）；而两拍值有预测力（三分位 → 本局胡牌率 14.3%/45.0%/68.4%，
    # 后续实际听口宽度 8.14/7.80/12.08）。成本约 150 ms/候选 ⇒ 约 450 ms/决策。
    "two-ply": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire",
                              wait_aware_tenpai=True, two_ply_shanten1=True)
    ),
    # 对照：只开两拍、不开听牌口径修正（用来把两层分开归因）。
    "two-ply-only": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", two_ply_shanten1=True)
    ),
    # 同向听次排序改用**加权形质值**（叠在 v3 之上，所以直接与冠军档对比）。
    # 依据：真机 3493 个决策点上 **77.2% 的决策里所有同向听候选的 `2×面子+搭子` 完全相同**
    # （副露 1 组 89.2%、2 组 93.5%）⇒ 次排序退化、中段只剩「喂牌」在起作用。
    # 修法给两面/对子/坎张分级并按副露数裁剪，且修正幅度 <1（只打破并列）。
    "shape": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire",
                              wait_aware_tenpai=True, shape_value=True)
    ),
    # 对照：只换形质值、不叠听牌修复（用来把两层分开归因）。
    "shape-only": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", shape_value=True)
    ),
    # **形质修好之后才谈得上降喂牌权重**：实测「全量 total」的 argmax 与「只看喂牌」
    # 的前 1 一致率 = 97.7%（形质项退化时），打开 shape 后降到 92.3%——因为
    # `3×feed` 的差幅（最大 1.8）仍大于形质项分辨率（<1）。而 `feed` 本身只是
    # 一张按牌种写死的表（字牌 0.4 / 中张 1.0 / 边张 0.6），**完全不看自己手牌需要什么**。
    # 所以 `feed-low` 当年测平是可解释的：**形质项退化 ⇒ 权重降了也没有东西可让位**。
    # 这条臂就是「修好形质 + 降权重」的组合。
    "shape-feed-low": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, feed_weight=1.0)
    ),
    # **副露闸门的重测**（带一个今晚才成立的理由）：当年 `meld-equal` 三个配置一致为负，
    # 但那时**副露手的候选并列最严重**（1 组 89.2%、2 组 93.5%）⇒ 副露之后的出牌近乎随机，
    # 于是「多副露」看起来只有代价、没有收益。修好形质项后这条混淆才被拆掉。
    # 而对手侧指纹显示副露是**最大的行为差异**：强 bot 1.20~1.30 副露/局 vs 我们 0.62。
    "shape-meld-equal": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, meld_tolerance="equal")
    ),
    "shape-meld-early": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, meld_tolerance="equal-early")
    ),
    # v3 + **训练过的**对手模型（`models/opponent_model.json`）。存在理由有两条：
    # ① 该模型挂在 `risk` 臂上**从未被对拍过**（与当年 `search` 同样的疏忽）；
    # ② 同批位置标定（n=3907）显示它比手写模型准得多：
    #    手写 `HeuristicReadyModel` 高估实测 **2.6 倍**（34.2% vs 13.2%），GBDT 只高估 **1.6 倍**（21.3%）。
    #    而 `threat = Σ ready_probability` 决定喂牌项的权重（真机实测中段 **97.7%** 由喂牌决定）。
    "risk-v3": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, wait_aware_tenpai=True),
        risk_model=load_or_none(OPPONENT_MODEL_PATH),
    ),
    # 喂牌项乘「还剩几张未现」(4−seen)/4 —— agent-c 复核确证的一条缺陷：
    # `visible_need` 是牌种静态表、**不含已见张数**，于是已见 3 张（几乎喂不出）的牌仍按满值计罚。
    # 这条与「threat 水平被放大 1.5 倍」是两个独立缺陷（那条是水平、这条是逐牌种分辨力）。
    "seen-feed": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              feed_visibility=True)
    ),
    # 两条一起：修正后的喂牌项（水平 × 分辨力）叠加修好的形质项
    "seen-shape-feed": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, feed_visibility=True, feed_weight=2.0)
    ),
    # **复测 `ukeire_order="blocks"`**（agent-c 复核的 ② 条）。它当年在 `ukeire-hand` 上
    # 测平，但那时 `block_value` 恒饱和（77~93% 并列）⇒ **按一个退化键排序 ≈ 随机排序**，
    # 那次测平不能说明「按形质排序无效」。现在 `shape_value` 给了这个键分辨率，
    # 这条臂才第一次真正测到「先按形质取候选面、再比进张」这件事。
    "shape-blocks": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3)
    ),
    # **显式的并列规则**（B' 的 D2）。依据：`shape-blocks` 的机制门实测显示，它对真机决策的
    # 改动 19.5% 全是「同喂牌值、同听口」的**等价候选换花色**（v3 打 1w、它打 1b／1t）
    # —— 也就是它实际兑现出来的是「并列怎么破」，而不是 ② 说的「更宽的听口被截掉」。
    # 而现在的并列兜底是 `sorted` 的稳定性 ⇒ 永远选最小牌索引 ⇒ 系统性偏向打万
    # （B' 量到万 0.54 / 筒 0.28 / 字 0.22 / 条 0.08）。这条臂把规则**显式**写成
    # 「同分优先打已见张最多的那张」（已被人打过 ⇒ 更不可能是他等的），
    # 与 v3 只差这一个开关，且结构上不可能覆盖任何一项的排序。
    "seen-tiebreak": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              safe_tiebreak=True)
    ),
    # **统一期望得分**（B' 18:45 设计说明；A 19:20 实现）。三因子：
    #   score = P_win×E_pay − (1−P_win)×P_opp×E_loss − feed_cost
    # 前两项**早已存在**于 `routes.evaluate` 的 `value`；本档位只接两处替换：
    # `value_weight 10.0→1.0`（去掉拍出来的倍数）、`feed_cost` 由旋钮改为
    # `ΔP_opp(tile) × E_loss`（量纲=分，`feed_weight` 被消灭）。
    # 刻意**不含** `P_win` 的形质修正：其定标依据 `shape-blocks` 的 n=10 还在跑。
    "unified": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              unified_score=True)
    ),
    # **重标后的统一期望得分**（A 19:35 的 `tools/calibrate_win_table.py` 门 2 实测）：
    # 门 2 不通过（斜率 1.694、s=2/3/4 高估 5~9pp、听牌态反轻微低估）⇒ 先用修正因子把
    # `P_win` 拉到实测水平，再谈结构。修正值 = 实测「实际/预测」比值（s=6 无样本沿用 s=5）。
    # 只在我方出牌这一支生效，v3 的听牌机制与碰吃闸门不受影响。
    "unified-recal": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="tenpai-only", wait_aware_tenpai=True,
                              unified_score=True,
                              win_table_correction=(1.10, 0.96, 0.73, 0.58, 0.52, 0.61, 0.61))
    ),
    # 同上的**主测试臂**：`tiebreak="tenpai-only"` ⇒ 只保留 v3 唯一被证过的机制
    # （听牌按可见听口选牌），把向听 ≥1 的次序**交还给统一期望得分**。
    # 理由：`unified`（带完整 `exact-ukeire` 层）与 v3 在真机决策点上分歧只有 **5.5%**
    # （`tools/divergence_gate.py`，200 点）——那一层把前 2 名又按进张重排、盖住了统一得分。
    # 而它是旧量纲下 59.5% 并列的**补丁**，统一得分的本意就是取代它。
    "unified-pure": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="tenpai-only", wait_aware_tenpai=True,
                              unified_score=True)
    ),
    # **v4 的缺失格子**（2026-09-30 01:10 登记）。v4 = 三个开关的合取（`shape_value` +
    # `ukeire_order=blocks` + 门开到 3），而它的两个「子集」都测平了：
    # `shape`（只开 shape_value，门 1）+0.119(t1.45)、`ukeire-hand`（只换排序键，门 1）+0.086(t0.66)、
    # `ukeire-early`（只开门 3，键与排序都不变）+0.154/+0.106 ≈ 0。
    # 于是还剩两个格子没测——**「门 3」与另外两个开关的任意两两组合**：
    #   `shape-gate3`  = shape_value + 门 3（排序键仍是 total）
    #   `blocks-gate3` = blocks 排序 + 门 3（**没有** shape_value）
    # 这两格决定「合取里到底哪一项不可少」：若 `blocks-gate3` 单独就能拿到 +0.27，
    # 说明**门才是关键**、`shape_value` 是可有可无（那 v4 的版本注释就要改）；
    # 若两者都明显低于 +0.27，则证实是**三项合取**、且 `shape_value` 确实不可少。
    "shape-gate3": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_max_shanten=3)
    ),
    "blocks-gate3": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              ukeire_order="blocks", ukeire_max_shanten=3)
    ),
    # **进张一级键 + 好型率二级键（带容差）**（B' 调研候选③，2026-09-30 登记）。叠在 **v4** 之上：
    # 并列候选里只保留 `copies >= 0.95 × max` 的，在其中取 `shape_mix` 结构代理最好的一张。
    # 依据 RiichiBook ch3 §3.4：perfect 1-away（2 両面 + 2 対子）的两面听牌率 100%，
    # 而含愚形搭子的同向听只有 50~70% ⇒ 同向听层内的质量差主要在这里。
    "goodshape": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              goodshape_tolerance=0.95)
    ),
    # **v4 + two-ply**（2026-09-30 13:30）：测两个**独立的正面组件**是否可加。
    # `two-ply`（v3 + 两拍值，n=2 时 +0.293 t2.26）与 v4 的形质层是不同机制：
    # 前者改**向听 1 的进张口径**（看「下一步还能进多少张」），后者改**形质分辨率与候选面**。
    # 刚被验证有效的做法就是「拆开算边际、再把边际加起来」——这条就是那个「加」。
    # ⚠ **成本**：两拍值单候选约 150 ms，叠在 v4 之上（v4 实测 p99 851ms/1800ms）⇒
    # 本臂**只用于离线 A/B**；若它胜出，上真机前必须先做候选预筛（`docs/ops.md` 红线 p99≤1000ms）。
    "v4-twoply": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              two_ply_shanten1=True)
    ),
    # **v4 的门/候选面单开关扫描**（2026-09-30 15:45，用户 7 小时自主窗口）。
    # 依据两条已证的事实：① shape 族拆解显示**门 3 是主项**（`shape-gate3` +0.214/t3.13，
    # 与 v4 的差在噪声内）；② C 14:15 的分解显示**缺口 62% 在「到听速度」**、集中在
    # n=4~10 中巡，而门控制的正是「中段是否按精确进张选牌」。
    # ⇒ 该试的是「门再开一点会不会更好」（`ukeire-deep` 的门 5 是在**另一种组合**下测的，
    # 不能直接外推到 v4 之上），以及「候选面放宽到 3/4 张」。
    # 四个都是**单开关**，零实现风险。
    "v4-gate4": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=4)
    ),
    "v4-gate6": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=6)
    ),
    "v4-cand3": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3)
    ),
    "v4-cand4": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=4)
    ),
    # **v5 上的两个单开关组合**（2026-10-01 01:10，用户 10 小时自主窗口）。
    # `v5-twoply`：候选面 + 两拍值的可加性——v4 上测过 two-ply 与 v4「重叠」（纯边际 +0.111 t1.15），
    #   但候选面是**另一个机制**（改的是「谁进比较」而不是「怎么比」），所以值得在 v5 上重测一次。
    # `v5-cand5`：候选面 2→5，检查「再宽是否更好」——cand3 与 cand4 打平 ⇒ 边际递减的迹象已出现，
    #   5 是用来**定位拐点**的（若 5 不优于 3，则 3 就是拐点）。
    "v5-twoply": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, two_ply_shanten1=True)
    ),
    "v5-cand5": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=5)
    ),
    # **候选面继续放宽 + 与两拍值组合**（2026-10-01 18:58，用户「继续推进」）。
    # 依据当天的结果：`v5-cand5` (+0.639 t6.33 n4) 明显高于 `v5-cand3` (+0.451 t8.68 n14)
    # ⇒ **候选面在 3 处还没到拐点**；而 `v5-twoply` (+0.651 t4.48 n4) 高于 `v5` (+0.451)
    # ⇒ 两拍值在候选面之上**是可加的**（此前在 v4 上测出「重叠」，那是被 field 混淆咬的：
    # 那条的 `--field` 默认＝baseline＝v4，量的是「v4 场」里的增量）。
    # `cand5-twoply` 是两者的组合，用来测「更宽的面 + 两拍值」是否继续叠加。
    # ⚠ 延迟：候选面每 +1 张多算一次精确进张（42–157ms）。v5(cand3) 实测 p99 860ms/1800ms
    # ⇒ cand7/cand10 大概率需要候选预筛才能上真机。**这几条只用于离线 A/B。**
    "v5-cand7": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=7)
    ),
    "v5-cand10": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=10)
    ),
    "v5-cand5-twoply": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=5, two_ply_shanten1=True)
    ),
    # **候选面 + 廉价预筛**（① 的实现）：有效候选面 10 张，但只对代理口径的前 5 张算精确进张。
    # 目的＝「拿到 cand10 量级的收益、付出 cand5 量级的成本」，以便上真机。
    "v5-presel5": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=10, ukeire_preselect=5)
    ),
    # **预筛的 N 由 B' 的召回率表定**（2026-10-01 21:40，n=1703 真机决策点）：
    #   N=5 → recall@3 0.799（**向听3 只有 0.422，过半概率丢掉「真·前 3」**）⇒ 太激进
    #   N=6 → 0.890（向听3 0.705）；N=8 → **0.975**（向听3 0.910）⇒ 达标
    # ⇒ `v5-presel5` 保留作「省但丢」的参照，另加 6/8 两档。
    # **注意**：预筛开启时 `ukeire_candidates` 被绕过 ⇒ **有效候选面＝全部同向听并列候选**，
    # 只对 `blocks` 排序的前 N 张算精确进张。所以 presel8 的成本≈cand8，
    # 但它的**有效面比 cand8 宽得多**（不被 8 截断）——这才是预筛的真正价值形态。
    "v5-presel6": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=10, ukeire_preselect=6)
    ),
    "v5-presel8": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=10, ukeire_preselect=8)
    ),
    # **边张搭单独计权**（agent-c 22:20 机制信号 → 2026-10-02 量化确认）：
    # `shape_value` 原先把 12/89 也当两面（1.2），而它们只等到 4 张（与坎张同）。
    # 两档分别给 0.7（与坎张同权）与 0.8（介于坎张与两面之间）——用来定位该给多少。
    "v5-edge7": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, edge_partial_weight=0.7)
    ),
    "v5-edge8": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, edge_partial_weight=0.8)
    ),
    # **弃胡阈值方向性测试**（B' 01:59 第 4 层）：`piao_threshold_scale` 调大＝更保守。
    # 目标口袋：弃胡后别人胡的 165 次让出 1696 番（整体弃胡仍是赚的，所以这是一次「条件化」的前置方向测试，
    # 不是一刀切收紧）。判读要看**弃胡次数与让出番数两个量**，不能只看总得分。
    # **弃胡阈值的反方向：`piao_threshold_scale=0.5`（更激进地追胡）**（A 2026-10-03 22:40 立）。
    #
    # **为什么现在补这一档**：① 这条轴的单调性已被两个端点夹住——
    # `piao13`(1.3) 在**平台口径**下是 **−0.042(t−3.80)** 且番数 −0.086(t−5.88)，
    # 而 legacy 基座上的 `chase-more`(0.5) 是**番数正、逐局名次负** ⇒ 两个端点符号相反；
    # ② S2（`agent/out/s2-fan-gap.txt`）显示我们的缺口 **87% 来自「胡得少」**（率差项 +0.273 / 总 +0.314），
    # 爆头率只有头部 bot 的 1/3（0.0287 vs 0.0879/局）⇒ **「更敢追胡」正好打这个缺口**。
    #
    # **注意**：已注册的 `chase-more` 是 `for_mode` 默认档 + 0.5，即 **legacy 基座**
    # （无 `wait_aware_tenpai`/`shape_value`/`exact-ukeire`）⇒ 拿它跟 v5 比是**换了基座**，
    # 归因不了阈值（与 `meld-equal` 同一个坑）。本档 = **v5 全部旋钮 + `piao_threshold_scale=0.5`**。
    # **`v7m` v1 = cell 条件化闸门**（A 2026-10-03 22:55 立；输入= C 的 1a，`agent/out/meld-cond.log`）。
    #
    # 与 `v6-equal` 的**唯一**差别：`v6-equal` 在**所有** cell 放开「向听不下降也可副露」，
    # `v7m` 只在**对手副露密度高**的 cell 放开（打出次数 ≤8 且向听 ≤3；9~12 且向听 ≤2），
    # 13+ 巡与深向听维持 STRICT —— 因为 1a 显示头部 bot 在那些 cell 里自己也不副露（≤11%、3 向听 0.4%）。
    # 其余旋钮与 `v5` 完全一致（唯一差别就是 `meld_tolerance=equal` + `meld_conditional=True`）。
    #
    # **预登记判据见 THREAD 22:55**：`v7m vs v5` 同场 4 种子，每场名次分 >0 且 t≥2 ⇒ 采纳；
    # ≤0 或 t<2 ⇒ **整条副露轴关闭**。机制门：副露率 0.9~1.25、胡次数不下降、番数不显著下降。
    "v7m": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, meld_tolerance="equal",
                              meld_conditional=True)
    ),
    # 对照：**同一基座 + 一律放开**（= `v6-equal` 的复刻，用于确认「条件化」确实比「一律」好）。
    "v7m-all": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, meld_tolerance="equal")
    ),
    # **G1 持财神听口加权**（A 2026-10-04 落地 B' 的实现稿；依据=财神专项 23:40 + 口径标定 1523/1523）。
    # `god_wait_boost=2.0` 初值来自「财神百搭 ⇒ 听口种数边际价值翻倍」的量级判断（B' 稿）。
    # `noWA` 是**消融**：同一个加权、但基础键换成「精确进张」（`wait_aware_tenpai=False`），
    # 用来分清「加权有效」还是「加权覆盖了原本有效的听口排序」。
    # **`v5-standing` = S3 局况姿态**（A 2026-10-04；窄臂形态，前置=表 C 11:10 + 接线 B' 11:35 + 定义 B' 11:42）。
    # 只在「差距可竞争」的边界态（占比 7.28%）把**弃胡/追胡阈值**按局况缩放：落后搏（0.7）、领先守（1.4），其余中性。
    # 与 `v5` 的差别只有四个 `standing_*` 字段（已逐字段核对）；`standing_*` 全默认时逐位等于 v5。
    # **`v5-standing-feed` = S3 v2**（A 2026-10-05 19:30 裁决）：同一定义（守 1.4 / 搏 0.7、分差≤29、余局≤1），
    # 但接入点从 v1 的**弃胡阈值**（触发 ≈0.05 次/座·局 ⇒ 功效为零）搬到**出牌层的喂牌代价**
    # （每个候选都算 ⇒ 边界态占出牌决策点 ~13% ⇒ 可测）。语义：守⇒喂牌惩罚加重、搏⇒减轻。
    # `standing_feed_apply=True` 使两个接入点**互斥**（不会把局况计两次）。
    # **`v5-godprog` = 财神做牌路径机制件 v1**（A 2026-10-05 20:30 立项）：有财神且到听进度落后时，
    # 落后态把「打财神禁忌」从 25.0 换成 `god_penalty_behind=2.0`（与候选间其他差异同量级）⇒ 打财神进入正常竞争。依据见 `PolicyConfig.god_penalty_behind`。
    # **门①（自对弈的「有财神桶」到听速度/爆头/平胡）先过，才起 A/B**（`v7m` 的教训）。
    "v5-godprog": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, god_penalty_behind=2.0)
    ),
    "v5-standing-feed": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, standing_lead_scale=1.4,
                              standing_behind_scale=0.7, standing_feed_apply=True)
    ),
    "v5-standing": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, standing_lead_scale=1.4,
                              standing_behind_scale=0.7)
    ),
    "v5-godwait": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, god_wait_boost=2.0)
    ),
    "v5-godwait-noWA": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=False,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, god_wait_boost=2.0)
    ),
    "v5-piao05": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, piao_threshold_scale=0.5)
    ),
    "v5-piao13": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, piao_threshold_scale=1.3)
    ),
    "v5-piao12": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, piao_threshold_scale=1.2)
    ),
    # **组合臂**（2026-10-02 12:50）：三个单开关各自独立为正，且都落在 +0.51~+0.65 带里
    # （基准 `v5`=+0.451）⇒ 下一步是测**可加性**：`presel5`（成本与召回）∪ `piao13`（弃胡口袋）。
    # 若可加，就是 v6 的形态；若重叠（≈max 而非 sum），说明它们动的是同一处（中段出牌质量）。
    "v6a": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, ukeire_preselect=5, piao_threshold_scale=1.3)
    ),
    "v6b": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, ukeire_preselect=5, edge_partial_weight=0.8,
                              piao_threshold_scale=1.3)
    ),
    "v6a-twoply": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, ukeire_preselect=5, piao_threshold_scale=1.3,
                              two_ply_shanten1=True)
    ),
    # **EQUAL 臂的正确形态**（A 2026-10-03 13:2x 裁决）：副露闸门的对照臂必须**只差闸门**。
    # 已注册的 `meld-equal` 是 `for_mode` 默认档 + `meld_tolerance="equal"`，即 **legacy 基座**——
    # 它既没有 `wait_aware_tenpai`/`shape_value`/`exact-ukeire`，也**不是冠军**。
    # 拿它上真机做「放开闸门的行为验证」，等于同时换掉基座，副露的行为变化无法归因。
    # 本档 = **v6 的全部旋钮 + `meld_tolerance="equal"`**，逐位可复现（见 tests/test_versions.py）。
    "v6-equal": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, ukeire_preselect=5, piao_threshold_scale=1.3,
                              meld_tolerance="equal")
    ),
    # **`v7-keeppairs` = 保留多余对子**（B'/coordinator 2026-10-08 立；用户 21:10 指示修复）。
    #
    # **现象（本日实测）**：`shape_value` 的 `weights[:need]` 裁剪把「排不进前 need 名的块」
    # 当 0 ⇒「多余的对子」在估值里完全消失。构造 1200 副「唯一对子=东东」手牌，打出东 **0** 次
    # （唯一对子不会被拆）；但 1500 副「恰好 2 张东」的随机手牌里有 **9** 次打出东，
    # **全部**是「东东」与另一个孤立字牌同分、被平局规则选中——即多余对子被当 0 后才与孤张并列。
    # 这与麻将常识相背：对子（尤其**字牌对子，不能被吃**）是最优碰材，不该在平局里被拆。
    #
    # **改动（只改一处）**：`shanten.shape_value(..., keep_extra_pairs)` 给被裁掉的对子一个
    # 折扣正值（字牌 0.5×、数牌 0.3×）。幅度 < 1 ⇒ 只打破并列、不覆盖块数差（与 shape_value 同约束）。
    # 其余旋钮与 `v5` 完全一致，唯一差别就是 `keep_extra_pairs` ⇒ 可归因。
    #
    # **预登记判据（先机制门、后 A/B）**：
    #   机制门（离线）：① 在「恰好 2 张东」的随机手牌上打出东的次数**显著下降**（目标 ≤ 原来的 1/3）；
    #     ② 默认档（keep_extra_pairs=0）逐位不变；③ 单决策 `shape_value` 成本不升（仍微秒级）。
    #   A/B 门：`v7-keeppairs vs v5`，同场 4 种子，合并名次分 >0 且 t≥2 ⇒ 采纳，建 `v7` 快照；
    #     否则回退（默认档与冠军档均保持 `keep_extra_pairs=0`）。
    "v7-keeppairs": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, keep_extra_pairs=1.0)
    ),
    # 消融：同样保留多余对子，但权重减半（字牌 0.25 / 数牌 0.15）。用来分清「是有用」还是「过冲」。
    "v7-keeppairs-half": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, keep_extra_pairs=0.5)
    ),
    # **`v7-chibest` = 多选择吃法【根因修复】（B'/coordinator 2026-10-08；A 22:12 裁定为该臂）**
    #
    # **现象（本日实测）**：`HeuristicDecider._shanten_after_meld`（及 `_meld_plan`）旧实现
    # **无视 `action.tiles`**，对每个 CHI 选项都取 `chi_combinations(counts, offered)[0]`（固定第一种吃法）。
    # 同一张 `offered` 的两种吃法可以有**不同**的「吃后最小向听」：扫 4000 副 13 张手牌，
    # 上家出牌能形成 ≥2 种吃法的 2904 个局面里 **1155 个（39.8%）** 不同。
    # 后果包括**漏吃**（不是选错、是该吃不吃）：手牌 `1w6w6w8w9w2b3b9b1t北北白白`，
    # 上家出 `7w`，吃 `8w9w` 真能到**向听 1**（连默认 STRICT 也该吃），
    # 但 `combos[0]=(6w,8w)` 只到向听 2 ⇒ 误判「无改善」而 PASS。
    #
    # **改动（只一处）**：吃牌按 `action.tiles` 算「吃后最小向听」。
    # 其余旋钮与 `v5` 完全一致，唯一差别就是 `meld_chi_best` ⇒ 可归因。
    # **默认档与冠军档 `meld_chi_best=False` ⇒ 逐位等于 v5**（未改默认档，§6/§7.3）。
    #
    # **预登记判据（先机制门、后 A/B）**：
    #   机制门（离线）：① 已知漏吃局面（`7w` 那个）必须从 PASS 变为吃 `8w9w`；
    #     ② 默认档 `meld_chi_best=False` 逐位不变。
    #   A/B 门：`v7-chibest vs v5`，同场 4 种子，合并「每场名次分」>0 且 t≥2 ⇒ 采纳建 `v7` 快照；
    #     |t|<1.2 ⇒ 关闭；1.2~2 ⇒ 补到 6 种子。
    "v7-chibest": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, meld_chi_best=True)
    ),
    # **`v7-keepchi` = 多选择吃法【根因修复 + 同向听次排序】**（A 22:12 裁定：②是①的**下游**、
    # **单独成第二臂**，故 = `v7-chibest` 再叠加 `meld_chi_tiebreak=True`）。
    # 只有 ① 修好后才会出现「同降幅的多种吃法」这个比较（此前不存在）。
    # 用户局面 `5w 4b5b6b7b7b8b 1t4t6t7t8t9t` 上家出 `7t`：吃 `8t9t`（留 `6t7t` 两面，`shape_value=4.18`）
    # 优于吃 `6t8t`（`3.955`）；精确进张 65 vs 71 张（旧代码恒选前者）。
    # 判据 `shape_value`（打哪张 + 吃哪两张都按它取最优），开销微秒级。
    # **A ③ 要求两臂独立预登记、不得合并计入同一次裁决**（本文只登记机制；A/B 各自跑）。
    "v7-keepchi": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode, tiebreak="exact-ukeire", wait_aware_tenpai=True,
                              shape_value=True, ukeire_order="blocks", ukeire_max_shanten=3,
                              ukeire_candidates=3, meld_chi_best=True, meld_chi_tiebreak=True)
    ),
    # **`botlike`**：Stage B 的 bot 出牌预测器（GBDT, 77.4% top-1）包成决策器，
    # **只用于当 `ab_test --field botlike` 的对手模型**（A 2026-10-06 01:57 提出的场地修正）。
    # 见 `strategy/botlike.py` 的模块 docstring。**不作为待采纳臂**。
    "botlike": lambda mode: __import__("majiang.strategy.botlike", fromlist=["build"]).build(mode),
    "first-legal": lambda _mode: FirstLegalDecider(),
}


def collect_tokens(prefix: str, environment: dict[str, str] | None = None) -> list[tuple[str, str]]:
    source = os.environ if environment is None else environment
    found: list[tuple[str, str]] = []
    for key in sorted(source):
        if key.startswith(prefix):
            value = source[key].strip()
            if value:
                found.append((key[len(prefix) :], value))
    return found


def make_decider(name: str, mode: Mode) -> Decider:
    # 版本库优先于 `DECIDERS`：`v1`/`v2` 是**已胜出并冻结**的冠军版本，
    # `DECIDERS` 是尚未胜出的实验档位。两者同名时以版本库为准（不该发生，但要有定论）。
    if versions.is_version(name):
        return versions.build(name, mode)
    factory = DECIDERS.get(name)
    if factory is None:
        raise SystemExit(f"未知决策器 {name!r}，可选: {', '.join(sorted(DECIDERS))}")
    return factory(mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="majiang", description="杭州麻将 AI 参赛程序")
    parser.add_argument("--server", default=os.environ.get("MAJIANG_SERVER", DEFAULT_SERVER))
    parser.add_argument("--token", help="参赛令牌（不推荐：会出现在进程列表里）")
    parser.add_argument("--token-env", help="存有参赛令牌的环境变量名")
    parser.add_argument("--env-prefix", help="收集该前缀下的所有环境变量作为多身份令牌")
    parser.add_argument("--decider", default="heuristic", help="决策器名称")
    parser.add_argument(
        "--mode",
        default=Mode.QUALIFIER.value,
        choices=[mode.value for mode in Mode],
        help="策略模式：晋级轮稳健 / 决赛激进",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="运行秒数，0 表示直到赛事终态")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--quiet", action="store_true", help="不往控制台打日志")
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help=f"每身份每秒请求上限，缺省 {PLATFORM_STATE_RATE_PER_SEC}（略低于平台 16/s 以留窗口余量）",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--reopen-test-room",
        action="store_true",
        help="测试房跑完一轮后再次到位开启下一轮（正式赛事不要打开）",
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="跳过启动时的指南版本自检（离线或用假令牌测试时使用）",
    )
    parser.add_argument(
        "--auto-match",
        action="store_true",
        help="走自由匹配：先用 POST /api/match 取得房号再参赛（需全局令牌，非报名令牌）",
    )
    return parser


def resolve_tokens(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.env_prefix:
        tokens = collect_tokens(args.env_prefix)
        if not tokens:
            raise SystemExit(f"环境变量里没有以 {args.env_prefix} 开头的令牌")
        return tokens
    if args.token_env:
        value = os.environ.get(args.token_env, "").strip()
        if not value:
            raise SystemExit(f"环境变量 {args.token_env} 为空")
        return [(args.token_env, value)]
    if args.token:
        return [("argv", args.token)]
    raise SystemExit("必须提供 --token-env、--env-prefix 或 --token 之一")


def resolve_auto_rooms(tokens: list[tuple[str, str]], server: str) -> list[tuple[str, str]]:
    """每个身份先调一次 ``POST /api/match`` 取得房号，返回 ``[(名字, 房号)]``。

    逐个身份报错**不互相影响**：某个令牌类型不对（报名令牌会 400 ``TOKEN_NOT_SCOPED``）
    或身份未绑定（403 ``PORTAL_BINDING_REQUIRED``），只跳过该身份。

    自动房不会出现在 ``/api/me`` 的 ``active_games`` 里（实测为空），所以房号只能这样带入。
    """
    assigned: list[tuple[str, str]] = []
    for name, token in tokens:
        try:
            result = PlatformApi(HttpTransport(server), token).match()
        except ApiError as error:
            print(
                f"[警告] {name} 自动匹配失败（{error.status} {error.code}: {error.message}），跳过该身份",
                file=sys.stderr,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            print(
                f"[警告] {name} 自动匹配异常（{type(exc).__name__}: {exc}），跳过该身份",
                file=sys.stderr,
            )
            continue
        if not result.room_id:
            print(f"[警告] {name} 未返回房号，跳过该身份", file=sys.stderr)
            continue
        config = result.config
        print(
            f"  {name} 入席 {result.room_id}"
            f"（Kind={config.get('Kind')} M={config.get('M')} Rounds={config.get('Rounds')}"
            f" 出牌{config.get('DiscardTimeoutSec')}s"
            f" 碰{config.get('PengTimeoutSec')}s"
            f" 吃{config.get('ChiTimeoutSec')}s）",
            file=sys.stderr,
        )
        assigned.append((name, result.room_id))
    return assigned


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokens = resolve_tokens(args)
    mode = Mode(args.mode)
    decider = make_decider(args.decider, mode)
    rate = args.rate if args.rate else PLATFORM_STATE_RATE_PER_SEC
    options = RuntimeOptions(
        rate_per_sec=rate,
        max_workers=args.max_workers,
        log_dir=args.log_dir,
        log_console=not args.quiet,
        reopen_finished_test_room=args.reopen_test_room,
    )
    print(f"启动 {len(tokens)} 个身份：{', '.join(name for name, _ in tokens)}", file=sys.stderr)
    print(
        f"服务器 {args.server}  决策器 {args.decider}({mode.value})  每身份限速 {rate:.1f}/s  "
        f"日志 {Path(args.log_dir).resolve()}",
        file=sys.stderr,
    )
    if not args.skip_version_check:
        check_guide_version(args.server)

    if args.auto_match:
        print("自由匹配：为每个身份调用 POST /api/match 取得房号", file=sys.stderr)
        assigned = resolve_auto_rooms(tokens, args.server)
        if not assigned:
            print("没有身份成功入席，退出", file=sys.stderr)
            return 1
        by_name = dict(assigned)
        runtimes = [
            Runtime(
                token,
                server=args.server,
                decider=decider,
                options=options,
                tournament_id=by_name[name],
            )
            for name, token in tokens
            if name in by_name
        ]
    else:
        runtimes = [
            Runtime(token, server=args.server, decider=decider, options=options)
            for _, token in tokens
        ]

    stopping = threading.Event()

    def handle_signal(_signum: int, _frame: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        print("收到中断，正在收尾…", file=sys.stderr)
        for runtime in runtimes:
            runtime.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    duration = args.duration if args.duration > 0 else None
    summaries: list[object] = []
    threads: list[threading.Thread] = []
    failures: list[BaseException] = []

    def drive(runtime: Runtime) -> None:
        try:
            summaries.append(runtime.run(duration_sec=duration))
        except Exception as exc:  # noqa: BLE001
            # 记下来并让进程以非零退出：守护脚本靠退出码区分「崩溃该重启」与
            # 「赛事终态正常收工」。若在此吞掉异常并返回 0，崩溃后守护不会重启。
            failures.append(exc)
            print(f"运行时异常退出: {type(exc).__name__}: {exc}", file=sys.stderr)

    for runtime in runtimes:
        thread = threading.Thread(target=drive, args=(runtime,), name="runtime")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    print("\n收尾：", file=sys.stderr)
    for summary in summaries:
        print(f"  {summary}", file=sys.stderr)
    if failures:
        print(
            f"{len(failures)} 个身份异常退出，以退出码 1 结束以便守护脚本重启",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
