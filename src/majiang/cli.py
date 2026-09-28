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
