"""庄闲分解：我们的庄家局打得怎么样（tasks.md 5.3 里最贵的一项，但从未单独量过）。

**为什么值得单独量**：本平台庄家自摸要付三家各 ×8，闲家自摸时庄家也付 ×8。
也就是**庄闲的赔付方向是 8 倍不对称**——庄家局的一胜一负相当于闲家局的 8 倍。
我们的目标函数里写了这一条（5.3），但从来没有把「我们的庄家局」与「对手的庄家局」
分开量过。如果我们在庄家局显著弱于对手，那是一个比听口宽度更贵的缺口。

口径：庄家取自事件流的 ``blocks[].dealer``（每局的权威来源），胜负与得分取自
``rounds[]``（流局数另按 ``round_ended.data.draw`` 计，见 measure_strength 的说明）。

输出四个格子：我们-庄 / 我们-闲 / 对手-庄 / 对手-闲，给出**胜率**与**每局净分**。

用法::

    uv run python tools/analyze_dealer.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

from majiang.sim import replay

SEATS = 4
US = "我们"
THEM = "对手"


def scan(payload: dict, ours: str, buckets: dict[tuple[str, str], list[int]]) -> None:
    ids = [str(seat.get("user_id", "")) for seat in (payload.get("seats") or [])]
    if len(ids) != SEATS or ours not in ids:
        return
    mine = ids.index(ours)
    official = {
        int(entry.get("round_no", 0) or 0): entry for entry in (payload.get("rounds") or [])
    }

    for span in replay.round_spans(payload):
        entry = official.get(span.round_no)
        if not entry:
            continue
        scores = entry.get("scores") or []
        if len(scores) != SEATS:
            continue
        dealer = span.dealer if 0 <= span.dealer < SEATS else int(entry.get("dealer", 0) or 0)
        winner = None if entry.get("is_draw") else entry.get("winner")
        for seat, uid in enumerate(ids):
            group = US if uid == ours else THEM
            role = "庄家" if seat == dealer else "闲家"
            bucket = buckets[(group, role)]
            bucket[0] += 1                                   # 手数
            bucket[1] += int(scores[seat])                    # 净分
            if isinstance(winner, int) and winner == seat:
                # 赢的是自摸：把这一手记为胡，同时记下番数
                bucket[2] += 1
                bucket[3] += int(entry.get("multiplier", 1) or 1)


def report(buckets: dict[tuple[str, str], list[int]]) -> None:
    print(f"{'':<8}{'角色':<6}{'手数':>8}{'胜率':>9}{'每局净分':>11}{'均番':>7}")
    for group in (US, THEM):
        for role in ("庄家", "闲家"):
            hands, score, wins, fan = buckets[(group, role)]
            if not hands:
                continue
            win_rate = wins / hands
            print(
                f"{group:<8}{role:<6}{hands:>8}{win_rate:>9.2%}"
                f"{score / hands:>11.3f}{(fan / wins if wins else 0):>7.3f}"
            )
        # 折叠：把庄闲合并，方便看总量
        hands = sum(buckets[(group, r)][0] for r in ("庄家", "闲家"))
        score = sum(buckets[(group, r)][1] for r in ("庄家", "闲家"))
        wins = sum(buckets[(group, r)][2] for r in ("庄家", "闲家"))
        print(f"{group:<8}{'合计':<6}{hands:>8}{wins / hands if hands else 0:>9.2%}"
              f"{score / hands if hands else 0:>11.3f}")
        print()
    print("庄家局对比（我们 vs 对手）：")
    us_h, us_s, us_w, _ = buckets[(US, "庄家")]
    them_h, them_s, them_w, _ = buckets[(THEM, "庄家")]
    if us_h and them_h:
        print(f"  胜率 我们 {us_w / us_h:.2%}  vs 对手 {them_w / them_h:.2%}"
              f"   差 {us_w / us_h - them_w / them_h:+.2%}")
        print(f"  每局净分 我们 {us_s / us_h:+.3f}  vs 对手 {them_s / them_h:+.3f}"
              f"   差 {us_s / us_h - them_s / them_h:+.3f}")
    us_h2, us_s2, us_w2, _ = buckets[(US, "闲家")]
    them_h2, them_s2, them_w2, _ = buckets[(THEM, "闲家")]
    if us_h2 and them_h2:
        print(f"  闲家局对比：胜率 {us_w2 / us_h2:.2%} vs {them_w2 / them_h2:.2%}"
              f"（{us_w2 / us_h2 - them_w2 / them_h2:+.2%}）"
              f"  每局净分 {us_s2 / us_h2:+.3f} vs {them_s2 / them_h2:+.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="庄闲分解")
    parser.add_argument("--manifest", default="notes/manifest-20260926.txt")
    parser.add_argument("--events", default="", help="改用 glob（与 --manifest 二选一）")
    parser.add_argument("--ours", default="u_a7f7c67bb14a")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.events:
        paths = sorted(glob.glob(args.events))
        label = args.events
    else:
        lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
        paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        label = args.manifest
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("没有文件", file=sys.stderr)
        return 1

    buckets: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    used = skipped = 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        try:
            scan(payload, args.ours, buckets)
        except Exception as exc:  # noqa: BLE001 —— 计数器带异常类型，避免被裸 except 吞掉
            print(f"  跳过 {Path(path).name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        used += 1

    print(f"文件 {used} 个（跳过 {skipped}），来源 {label}")
    print()
    report(buckets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
