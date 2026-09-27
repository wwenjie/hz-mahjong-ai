"""查「拷响 / 承包」是否存在：同一家被同一人吃碰杠 ≥3 次时，官方得分是否超出常规模型。

**为什么查这个**：网上资料提到杭州麻将可能有「承包」类规则（吃/碰/杠同一家达 3 次，
该家和牌则你承包其全部输分）。我们的 `TournamentConfig.you_cai_bi_kao` 在自动房是 False，
但**正式赛事的配置可能不同**。若该规则存在，那么「喂牌」的代价就不是我们假设的
「让对手推进一点」，而是**一笔可能翻数倍的或有负债**——而我们的喂牌代价是
`visible_need(tile) × Σ对手听牌概率`，**完全与「我已经喂了这家几次」无关**。

**判据（纯经验，不依赖文档）**：找出「同一家对同一人的吃碰杠 ≥3 次」的局，
把官方公布的 `scores` 与我们的 `score_module.seat_deltas` 对账。
若两者在这些局上系统性不符，而其他局符合 → 该规则存在。

用法::

    uv run python tools/analyze_chengbao.py --events 'data/auto_sessions/*/events/*.json' --limit 600
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from majiang.rules import score as score_module
from majiang.sim import replay

SEATS = 4
CLAIMS = {"peng", "chi", "gang"}


def scan(payload: dict, counters: Counter, examples: list[str]) -> None:
    # **胜者只在顶层 ``rounds[]`` 里**：``round_ended.data`` 只有 draw/fan/detail/scores
    # （agent B 实测确认）。第一版我在这里读 ``state.result["winner"]``，取到 None
    # 就静默 `continue`，于是 262 个样本全被丢掉、却报出「官方得分与模型完全一致」
    # ——差点交付一个假阴性。这类「读错字段导致样本为空」必须用计数器暴露出来。
    official = {
        int(entry.get("round_no", 0) or 0): entry
        for entry in (payload.get("rounds") or [])
    }
    for state, events in replay.iter_rounds(payload):
        fed: Counter = Counter()
        for event in events:
            kind = str(event.get("type"))
            if kind in CLAIMS:
                seat = event.get("seat")
                discarder = state.last_discarder
                if (
                    isinstance(seat, int)
                    and 0 <= seat < SEATS
                    and 0 <= discarder < SEATS
                    and discarder != seat
                ):
                    fed[(seat, discarder)] += 1
            replay.apply_event(state, event)

        entry = official.get(state.round_no) or {}
        scores = entry.get("scores") or []
        if len(scores) != SEATS or entry.get("is_draw"):
            counters["流局/无结果"] += 1
            continue
        worst = max(fed.values(), default=0)
        counters[f"单对最大喂牌次数={min(worst, 4)}"] += 1
        if worst < 3:
            continue

        winner = entry.get("winner")
        if not isinstance(winner, int) or not 0 <= winner < SEATS:
            counters["喂牌≥3 但取不到赢家"] += 1
            continue
        dealer = int(getattr(state, "dealer", 0) or 0)
        if not 0 <= dealer < SEATS:
            dealer = 0
        fan = int(entry.get("multiplier", 1) or 1)
        try:
            predicted = score_module.seat_deltas(
                fan, 1, winner_seat=winner, dealer_seat=dealer
            )
        except Exception as exc:  # noqa: BLE001
            counters[f"模型异常:{type(exc).__name__}"] += 1
            continue
        official = tuple(int(value) for value in scores)
        counters["喂牌≥3 的局"] += 1
        if official == predicted:
            counters["  官方得分 == 我们的模型"] += 1
        else:
            counters["  官方得分 != 我们的模型"] += 1
            # 找出被喂得最多的那家是不是赢家（承包的典型后果）
            top = max(fed.items(), key=lambda kv: kv[1])
            if top[0][0] == winner:
                counters["    其中赢家正是被喂最多的那家"] += 1
            if len(examples) < 8:
                examples.append(
                    f"喂牌 {dict(fed)} 赢家={winner} 番={fan} "
                    f"官方={official} 模型={tuple(predicted)}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查拷响/承包是否存在")
    parser.add_argument("--events", default="data/auto_sessions/*/events/*.json")
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.events))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有匹配到事件流", file=sys.stderr)
        return 1

    counters: Counter = Counter()
    examples: list[str] = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        try:
            scan(payload, counters, examples)
        except Exception as exc:  # noqa: BLE001 —— 带异常类型，避免被裸 except 吞掉
            counters[f"跳过:{type(exc).__name__}"] += 1

    print(f"文件 {len(paths)} 个")
    for key in sorted(counters):
        print(f"  {key}: {counters[key]}")
    print()
    if examples:
        print("官方得分与模型不符的样例：")
        for line in examples:
            print("  " + line)
    print()
    mismatch = counters.get("  官方得分 != 我们的模型", 0)
    if mismatch:
        print(f"结论：**有 {mismatch} 局不符** —— 需要逐例判读（也可能是我方模型对某些番型的"
              f"折算差一档，而非承包规则）。请人工核对样例后定性。")
    else:
        print("结论：喂牌 ≥3 次的局里**官方得分与我们不含承包的模型完全一致** ——"
              "该平台这批配置下没有承包类计分。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
