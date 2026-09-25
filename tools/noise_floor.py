"""噪声基底：真机数据的房间聚类到底让「胜率差」有多不精确（用于判断真机 A/B 能不能成）。

**为什么现在能量这个**：本轮查出 `HeuristicDecider.configure()` 会把全部变体开关抹平
（见 `notes/agent-a.md`），所以真机上 `no-chase` / `ukeire` / `meld-equal` **跑的都是默认档**。
后见之明看这是坏事，但作为**零假设实验**它很值钱：那些「臂」在统计上就是同一策略的
随机分组，它们之间的胜率差就是**纯噪声的尺度**，而且天然包含了房间聚类
（每个房四家固定、每房约 80 手）。

做法：按**房**做 bootstrap——把房随机对半分，各半算我们的胜率，取差值，重复多次。
报告差值的标准差、以及「要检出 X 个百分点的真实差异需要多少手」。

单看二项分布会严重低估不确定性：手数不是独立的，同一个房里四家固定、牌风与对手强度
都在房内相关。

用法::

    uv run python tools/noise_floor.py --manifest notes/manifest-20260926.txt
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
import sys
from pathlib import Path

from majiang.sim import replay

SEATS = 4
# 实测速率：111 房 / 29.8 小时，每房约 78 手（M=10 桌 × 8 回合）
HANDS_PER_HOUR = 285.0


def per_room(paths: list[str], ours: str) -> dict[str, tuple[int, int]]:
    """房号 → (我们的手数, 我们的胡次数)。"""
    rooms: dict[str, list[int]] = {}
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
        if len(ids) != SEATS or ours not in ids:
            continue
        room = str(payload.get("room_id") or "")
        if not room:
            continue
        mine = ids.index(ours)
        bucket = rooms.setdefault(room, [0, 0])
        for result in payload.get("rounds") or []:
            scores = result.get("scores") or []
            if len(scores) != SEATS:
                continue
            bucket[0] += 1
            if not result.get("is_draw") and result.get("winner") == mine:
                bucket[1] += 1
    return {room: (counts[0], counts[1]) for room, counts in rooms.items()}


def win_rate(rooms: list[tuple[int, int]]) -> tuple[float, int]:
    hands = sum(item[0] for item in rooms)
    wins = sum(item[1] for item in rooms)
    return (wins / hands if hands else 0.0), hands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真机数据的噪声基底")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--rounds", type=int, default=3000, help="bootstrap 次数")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)

    lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
    paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if not paths:
        print("清单为空", file=sys.stderr)
        return 1

    per = per_room(paths, args.ours)
    if not per:
        print("没有可用的房", file=sys.stderr)
        return 1
    overall, hands = win_rate(list(per.values()))
    print(f"房 {len(per)} 个，我们 {hands} 手，总胜率 {overall:.2%}")
    print()

    rng = random.Random(args.seed)
    keys = list(per)
    deltas: list[float] = []
    for _ in range(args.rounds):
        rng.shuffle(keys)
        cut = len(keys) // 2
        left, hands_left = win_rate([per[key] for key in keys[:cut]])
        right, hands_right = win_rate([per[key] for key in keys[cut:]])
        deltas.append(left - right)

    spread = stats.pstdev(deltas)
    half_hands = hands / 2
    independent = (overall * (1 - overall) / half_hands) ** 0.5
    print(f"把房随机对半分，两侧胜率差的标准差 = {spread:.2%}")
    print(f"（每侧约 {half_hands:.0f} 手；若手数彼此独立，二项分布给出的标准差仅 {independent:.2%}）")
    print(f"→ 聚类带来的方差膨胀 ≈ {(spread / independent) ** 2:.1f} 倍")
    print()

    # 由 bootstrap 反推「每手的有效标准差」。差值 = 两侧之差，故单侧 SE = spread / sqrt(2)。
    # 有了 sigma_eff，任意手数 n 的单侧 SE = sigma_eff / sqrt(n)，可外推所需样本量。
    sigma_eff = (spread / 2**0.5) * half_hands**0.5
    z = 1.96 + 0.84  # 双侧 5% + 80% 功效
    print(f"{'想检出的胜率差':>14} {'每臂手数':>12} {'交错采集所需时长':>18}")
    for target in (0.02, 0.03, 0.05, 0.08):
        needed = (z * 2**0.5 * sigma_eff / target) ** 2
        # 交错：同一时段内两臂各得约一半手数，故总时长 = 2 * 每臂手数 / 速率
        hours = 2 * needed / HANDS_PER_HOUR
        feasible = "" if hours < 200 else "  ← 来不及"
        print(f"{target:>13.0%} {needed:>12.0f} {hours:>15.0f} 小时{feasible}")
    print()
    print(f"（速率按实测 {HANDS_PER_HOUR:.0f} 手/小时；交错采集时两臂共享这个速率）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
