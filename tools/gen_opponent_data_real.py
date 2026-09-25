"""从**真实对局**事件流生成对手模型数据（tasks.md 6A.4 → 6B.1）。

与 ``gen_opponent_data.py``（本地自对弈生成）的区别只在数据来源：这里用 test/auto 房
采集到的真实事件流（含四家起手手牌与完整事件序列），经 ``sim.replay`` 重建出逐时刻局面后
取样。特征提取与标签口径**完全复用**同一套代码，因此两个数据集可以直接拼接训练。

样本定义（与自对弈版本一致）：在某个决策时刻，以座位 ``observer`` 为视角、座位 ``target``
为对手，取其公开特征；标签是**该时刻 target 是否 0 向听**——由重建出的完整手牌给出，
是**无噪声的事实**。

合规说明：标签使用了对手手牌，但这只发生在**离线数据生成**阶段；生成出来的每一条样本
（``x``）仅含公开特征，线上推理路径不会读取对手手牌。

用法::

    uv run python tools/gen_opponent_data_real.py \\
        --events 'data/harvest/*/events/*.json' --out data/opponent_real.part000.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from majiang.rules import shanten as shanten_module
from majiang.sim import replay
from majiang.strategy.opponent_features import FEATURE_COUNT, extract

SEATS = 4
# 只在会引发决策的时刻取样，避免在同一局面附近堆大量几乎重复的样本
SAMPLE_ON = ("tile_drawn", "tile_discarded")


def loader(pattern: str) -> list[str]:
    """支持逗号分隔的多个 glob。"""
    paths: list[str] = []
    for piece in pattern.split(","):
        found = sorted(glob.glob(piece.strip()))
        if not found:
            raise SystemExit(f"没有匹配到文件: {piece.strip()}")
        paths.extend(found)
    return paths


def is_tenpai(state: replay.ReplayState, seat: int) -> bool:
    """标签：该座位此刻是否 0 向听（用重建出的完整手牌，离线专用）。"""
    seat_state = state.seats[seat]
    try:
        return shanten_module.shanten(seat_state.hand, len(seat_state.melds)) == 0
    except shanten_module.ShantenError:
        return False


def samples_from(payload: dict) -> tuple[list[list[float]], list[float], Counter, Counter]:
    """一份事件流（**一场 8 局**）→ 样本。返回 ``(特征行, 标签, 计数, 重建异常)``。

    逐局重建：8 局各有自己的起手手牌，铺在同一局面上跑会让第 2 局起的局面全错。
    """
    stats: Counter = Counter()
    rows: list[list[float]] = []
    labels: list[float] = []
    anomalies: Counter = Counter()
    for state, events in replay.iter_rounds(payload):
        for event in events:
            # 注意：此时 state 是**该事件发生之前**的局面
            if str(event.get("type")) in SAMPLE_ON:
                if not state.opened:
                    # 未摸第一张牌时牌墙为 84，TableState（按发牌 53 张定义）不接受
                    stats["skipped-before-first-draw"] += 1
                else:
                    stats["moments"] += 1
                    for observer in range(SEATS):
                        try:
                            situation = state.situation_for(observer)
                        except Exception:  # noqa: BLE001 —— 单个时刻不可重建不应中断整局
                            stats["situation-failed"] += 1
                            continue
                        for target in range(SEATS):
                            if target == observer:
                                continue
                            rows.append(extract(situation, target))
                            labels.append(1.0 if is_tenpai(state, target) else 0.0)
            replay.apply_event(state, event)
        anomalies.update(state.anomalies)
    return rows, labels, stats, anomalies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从真实对局事件流生成对手模型数据")
    parser.add_argument("--events", default="data/harvest/*/events/*.json")
    parser.add_argument("--out", default="data/opponent_real.part000.npz")
    args = parser.parse_args(argv)

    paths = loader(args.events)
    rows: list[list[float]] = []
    labels: list[float] = []
    totals: Counter = Counter()
    games = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            totals["bad-json"] += 1
            continue
        try:
            game_rows, game_labels, stats, anomalies = samples_from(payload)
        except Exception as exc:  # noqa: BLE001 —— 单局失败不应中断整批
            totals[f"replay-failed:{type(exc).__name__}"] += 1
            continue
        games += 1
        rows.extend(game_rows)
        labels.extend(game_labels)
        totals.update(stats)
        for kind, times in anomalies.items():
            totals[f"anomaly:{kind}"] += times

    if not rows:
        print("没有生成任何样本", file=sys.stderr)
        return 1
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if x.shape[1] != FEATURE_COUNT:
        raise SystemExit(f"特征数不符：生成 {x.shape[1]}，代码 {FEATURE_COUNT}")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, x=x, y=y)

    print(f"对局 {games} 局（文件 {len(paths)} 个），样本 {x.shape[0]} 条，正例率 {y.mean():.1%}")
    print(f"统计: {dict(totals)}")
    print(f"已写入 {target.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
