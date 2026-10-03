"""配对 A/B 检验（tasks.md 6B.11）。

为什么不能直接跑两遍 ``selfplay.py`` 比总数：座位与时序仍有残差偏差，且总得分是重型
右偏分布（少数大番把均值拉走），只看均值无法判断差异是否超出噪声。

设计：

1. **同种子重放**：两次运行的 ``--seed`` 相同，而 ``run_match`` 的随机源由
   ``seed * 100003 + index`` 推导，因此**同一场次的两侧拿到完全相同的牌**，构成配对。
2. **四座位旋转**：treatment 依次放在 0/1/2/3 座各跑一轮，另三座为 baseline。
   把四轮数据合并，抵消座次与时序偏置，统计功效提升 4 倍。
3. **逐场配对差分**：对每场算 ``d = treatment 分 − baseline 分``，报告均值、标准误、
   t 值与 95% 置信区间。区间跨 0 即「无显著差异」。

用法::

    uv run python tools/ab_test.py --treatment risk --baseline heuristic \\
        --matches 200 --seed 20260923
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from majiang.cli import DECIDERS, make_decider
from majiang.sim.batch import SEATS, run_match
from majiang.strategy import versions
from majiang.strategy.policy import Mode

TALLY = {"heuristic": Mode.QUALIFIER, "final": Mode.FINAL, "qualifier": Mode.QUALIFIER}
ALIASES = {"baseline": "first-legal"}

# 一场的产出：各座位依次为 (总得分, 名次分, 白板数, 胡次数, 番数总和)
MatchOutcome = tuple[tuple[int, int, int, int, int], ...]


def build(name: str):
    if name in TALLY:
        return make_decider("heuristic", TALLY[name])
    resolved = ALIASES.get(name, name)
    # 允许直接指名版本号（`--treatment v2 --baseline v1`），见 strategy/versions.py
    if resolved not in DECIDERS and not versions.is_version(resolved):
        raise SystemExit(
            f"未知策略 {name!r}，可选: {sorted(set(DECIDERS) | set(TALLY) | set(ALIASES))}"
            f" 或版本号 {sorted(versions.BY_ID)}"
        )
    return make_decider(resolved, Mode.QUALIFIER)


def play(
    names: list[str],
    match_index: int,
    *,
    rounds: int,
    base_score: int,
    seed: int,
) -> MatchOutcome:
    """跑一场并返回各座位产出。

    种子与庄家轮换的推导必须与 ``sim.batch.run_batch`` 完全一致，否则两侧不是同一副牌。
    """
    deciders = [build(name) for name in names]
    result = run_match(
        deciders,
        rounds=rounds,
        base_score=base_score,
        seed=seed * 100003 + match_index,
        labels=names,
        start_dealer=match_index % SEATS,
    )
    return tuple(
        (
            stat.total_score,
            stat.place_points,
            stat.god_count,
            stat.wins,
            stat.fan_total,
            stat.game_place_points,
        )
        for stat in result.seats
    )


def names_for(rotation: int, seat_name: str, field_list: list[str]) -> list[str]:
    """把 ``seat_name`` 放到 ``rotation`` 座，另三座按升序填 ``field_list``。"""
    names = list(field_list)
    others = [seat for seat in range(SEATS) if seat != rotation]
    row = [""] * SEATS
    row[rotation] = seat_name
    for seat, name in zip(others, names):
        row[seat] = name
    return row


# 并行任务只能传可 pickle 的裸数据（`names` 由 worker 自己重建），绝不能传决策器对象。
PlayTask = tuple[list[str], int, int, int, int]
SeatTask = tuple[int, int, str, list[str], int, int, int]


def _play_task(task: PlayTask) -> MatchOutcome:
    """一场完整对局，返回四座产出。用于「另三座＝baseline」的对照侧（一场顶四座）。"""
    names, index, rounds, base_score, seed = task
    return play(names, index, rounds=rounds, base_score=base_score, seed=seed)


def _seat_task(task: SeatTask) -> tuple[int, int, int, int, int, int]:
    """只取 `seat_name` 在 `rotation` 座的产出。

    **一次只算一座不是浪费**：treatment 在每个旋转座坐的都是**不同的一局**
    （``names_for`` 把另三座填成 field，所以座位排布不同 ⇒ 牌局不同），不存在可复用的重复。
    拆成 ``4×matches`` 个同粒度任务是为了让**一个进程池**吃饱——
    原先是每个旋转单独起池、每池只有 ``matches`` 个任务，8 进程时利用率只有 6/8。
    """
    index, rotation, seat_name, field_list, rounds, base_score, seed = task
    outcome = play(
        names_for(rotation, seat_name, field_list),
        index,
        rounds=rounds,
        base_score=base_score,
        seed=seed,
    )
    return outcome[rotation]  # type: ignore[return-value]


def _run_tasks(func, tasks: list, jobs: int) -> list:
    """按 ``jobs`` 个进程跑 ``tasks``，**返回顺序与输入一致**。

    为什么并行与串行结果必须逐位相同：``run_match`` 自己建 ``random.Random(seed)``，
    策略层没有任何模块级随机源（``strategy/search.py``、``strategy/rollout.py`` 的 rng
    都由显式 seed 构造），每场次的决策器也是当场新建的、不跨场携带状态。
    所以 ``play`` 是 ``(names, index, rounds, base_score, seed)`` 的纯函数 —— 并行只影响
    谁在哪个核上算，不影响算出来是什么。
    """
    if jobs <= 1 or len(tasks) <= 1:
        return [func(task) for task in tasks]
    # 每个 worker 拿 4 段：场次耗时相近，分段只是为了摊掉进程启动与调度开销
    chunk = max(1, len(tasks) // (jobs * 4))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(func, tasks, chunksize=chunk))


def default_jobs() -> int:
    """留出余量的默认进程数。

    **余量是给真机采集的**：碰/吃窗口只有 600 ms，采集进程是 ``nice 0`` 而我们 ``nice 15``，
    内核会立刻抢占，但如果把 16 个核全占满，抢占带来的缓存/内存带宽抖动仍可能让
    决策算不完 600 ms。所以默认只用到约 3/4 的核，上限 12。
    """
    return max(1, min(12, (os.cpu_count() or 4) * 3 // 4))


def describe(differences: list[float], label: str) -> None:
    count = len(differences)
    if count < 2:
        print(f"  {label:10s} 样本不足")
        return
    mean = sum(differences) / count
    variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
    standard_error = math.sqrt(variance / count)
    if standard_error == 0:
        print(f"  {label:10s} 差分恒为 {mean:+.3f}（无方差）")
        return
    statistic = mean / standard_error
    margin = 1.96 * standard_error
    verdict = "显著" if abs(statistic) > 1.96 else "**不显著**"
    print(
        f"  {label:10s} 均值 {mean:+8.3f}  标准误 {standard_error:6.3f}  "
        f"t {statistic:+6.2f}  95%CI [{mean - margin:+.3f}, {mean + margin:+.3f}]  {verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="配对 A/B 检验")
    parser.add_argument("--treatment", default="risk")
    parser.add_argument("--baseline", default="heuristic")
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--base-score", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "并行进程数。**默认 1（串行）保证与旧版输出逐位相同**；队列用 "
            "`tools/iterate_loop.py` 显式传值。结果与并行度无关（见 ``_run_tasks`` 的说明）。"
        ),
    )
    parser.add_argument(
        "--field",
        default="",
        help=(
            "另三座坐谁（默认＝baseline，即原来的行为）。可以给 1 个名字（三座相同）"
            "或 **3 个名字（按座位升序分配给非旋转座）**。**这是评估「副露」类假设的必要条件**："
            "真机对手副露 1.093/局，而我们自己的策略只 0.591/局——所以「默认 field=baseline」"
            "等于让我们对着三个几乎不副露的复制品打分，结构上测不出副露的价值。"
            "给 3 个不同名字还有一个独立用处：**异质场地**能打散「三个自己的复制品可被"
            "同一套偏离方式利用」这个偏差——实测 16 个单旋钮档位里多数总得分小幅为正"
            "（合并均值约 +0.40），怀疑来自该偏差，见 `feed-high` 的符号镜像检定。"
        ),
    )
    args = parser.parse_args(argv)

    rounds, base = args.rounds, args.base_score
    field_list = [name.strip() for name in (args.field or args.baseline).split(",") if name.strip()]
    if len(field_list) == 1:
        field_list = field_list * 3
    if len(field_list) != 3:
        raise SystemExit(f"--field 只接受 1 个或 3 个名字，收到 {len(field_list)} 个")
    field = args.field or args.baseline
    jobs = max(1, args.jobs)
    print(
        f"treatment={args.treatment}  baseline={args.baseline}  场上另三座={field_list}  "
        f"场数 {args.matches}  每场 {rounds} 局  种子 {args.seed}（四座位旋转）  并行 {jobs}"
    )

    started = time.perf_counter()
    # 对照侧与试验侧必须在**同一个场**里测，否则差分会混入场强差异。
    # field 全是 baseline 时退化成「一个 [baseline]*4 跑一遍、按旋转取座」，与原实现等价
    # （也省掉 4 倍重复计算）。
    same_field = len(set(field_list)) == 1 and field_list[0] == args.baseline
    if same_field:
        shared = _run_tasks(
            _play_task,
            [
                ([args.baseline] * SEATS, index, rounds, base, args.seed)
                for index in range(args.matches)
            ],
            jobs,
        )
        baseline_runs = [[row[seat] for seat in range(SEATS)] for row in shared]
    else:
        baseline_runs = _run_tasks(
            _seat_task,
            [
                (index, rotation, args.baseline, field_list, rounds, base, args.seed)
                for index in range(args.matches)
                for rotation in range(SEATS)
            ],
            jobs,
        )
        baseline_runs = [  # 与 same_field 分支同形：[[四座], ...] 按场次索引
            [baseline_runs[index * SEATS + seat] for seat in range(SEATS)]
            for index in range(args.matches)
        ]

    score_diff: list[float] = []
    place_diff: list[float] = []
    god_diff: list[float] = []
    win_diff: list[float] = []
    fan_diff: list[float] = []
    game_place_diff: list[float] = []
    collected: list[tuple[float, float, float, float, float]] = []

    # 四个旋转一次性提交给同一个进程池：任务数 4×matches，进程数够多时几乎无空转。
    plan = [(rotation, index) for index in range(args.matches) for rotation in range(SEATS)]
    played = _run_tasks(
        _seat_task,
        [
            (index, rotation, args.treatment, field_list, rounds, base, args.seed)
            for rotation, index in plan
        ],
        jobs,
    )
    for rotation in range(SEATS):
        mine_total = theirs_total = 0
        for (row_rotation, index), mine in zip(plan, played):
            if row_rotation != rotation:
                continue
            theirs = baseline_runs[index][rotation]
            score_diff.append(mine[0] - theirs[0])
            place_diff.append(mine[1] - theirs[1])
            god_diff.append(mine[2] - theirs[2])
            win_diff.append(mine[3] - theirs[3])
            fan_diff.append(mine[4] - theirs[4])
            game_place_diff.append(mine[5] - theirs[5])
            collected.append((mine[0], theirs[0], mine[3], theirs[3]))
            mine_total += mine[0]
            theirs_total += theirs[0]
        print(
            f"  旋转 {rotation}（treatment 在 {rotation} 座）  "
            f"treatment 总得分 {mine_total:>8d}   baseline {theirs_total:>8d}   "
            f"差 {mine_total - theirs_total:+7d}"
        )

    elapsed = time.perf_counter() - started
    pairs = len(score_diff)
    mine_score = sum(row[0] for row in collected)
    theirs_score = sum(row[1] for row in collected)
    print(f"\n配对样本 {pairs} 场（{SEATS} 旋转 × {args.matches} 场），用时 {elapsed:.1f}s")
    print(
        f"总得分合计 treatment {mine_score} vs baseline {theirs_score}"
        f"（{mine_score - theirs_score:+d}）  "
        f"胡率 {sum(r[2] for r in collected) / (pairs * rounds):.1%} vs "
        f"{sum(r[3] for r in collected) / (pairs * rounds):.1%}"
    )

    print("\n逐场配对差分（treatment − baseline，正值表示 treatment 更好）")
    describe(score_diff, "总得分")
    describe(game_place_diff, "每场名次分")
    describe(place_diff, "名次分")
    describe(god_diff, "白板数")
    describe(win_diff, "胡次数")
    describe(fan_diff, "番数总和")
    print(
        "  （口径：`每场名次分` = 按**整场累计总得分**给四家排 +3/+1/−1/−3，**与平台战绩同口径**；"
        "`名次分` = **逐局**名次之和，是历史对照口径，两者不可混用）"
    )
    print(f"  （每场 {rounds} 局，胡次数均差 0.10 ≈ 胡率差 {0.1 / rounds:.2%}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
