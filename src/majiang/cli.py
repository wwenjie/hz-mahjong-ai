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
    # 价值模型驱动（tasks.md 5.15）：用自对弈学到的价值函数给出牌打分
    "value": lambda mode: _value_decider(mode, VALUE_MODEL_PATH),
    # 对手听牌模型驱动的风险（tasks.md 6B）：模型不可用时自动回退手写启发式
    "risk": lambda mode: HeuristicDecider(
        PolicyConfig.for_mode(mode), risk_model=load_or_none(OPPONENT_MODEL_PATH)
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
