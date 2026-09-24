"""与官方番型计算端点逐例对拍（tasks.md 2.13 / 2.14）。

官方端点 ``POST /portal/api/tools/fan-calc`` 免认证、每 IP 限速 10/s，纯计算无状态，
可作为番型口径的权威参照。本脚本生成用例、逐个提交、与本地规则引擎逐字段比对。

用法::

    uv run python tools/fan_calc_diff.py --cases 300
    uv run python tools/fan_calc_diff.py --server https://host:port --cases 50

比对字段：``hu`` / ``baotou`` / ``fan`` / ``detail`` / 三家得分。退出码非 0 表示
存在不一致。
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from majiang.rules import fan as fan_module
from majiang.rules import score as score_module
from majiang.rules import tiles
from majiang.rules.tiles import GOD

DEFAULT_SERVER = "https://10.240.169.190:18080"
FAN_CALC_PATH = "/portal/api/tools/fan-calc"
REQUEST_INTERVAL = 0.16
HAND_SIZE = 13
MAX_CHAIN = 6
FOUR_GODS = 4

BUILTIN_CASES: tuple[tuple[str, int, int], ...] = (
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", 0, 0),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", 0, 0),
    ("1w2w3w4w5w6w7w8w9w白白白白", 0, 0),
    ("1w1w2w2w3w3w4w4w5w5w6w6w5b", 0, 0),
    ("1w1w1w1w2w2w2w2w3w3w4w4w5w", 0, 0),
    ("1w1w1w1w2w2w2w2w3w3w3w3w白", 3, 3),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", 6, 0),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", 2, 2),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", 3, 3),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", 4, 4),
    ("1w2w3w4w5w6w7w8w9w1b2b3b5b", 2, 0),
    ("1w2w3w4w5w6w7w8w9w1b2b3b白", 4, 2),
)


@dataclass(frozen=True, slots=True)
class Outcome:
    hu: bool
    baotou: bool
    fan: int
    detail: tuple[str, ...]
    scores: dict[str, dict[str, object]] | None = None
    error: str | None = None


def tokens(spec: str) -> list[str]:
    out: list[str] = []
    pending = ""
    for char in spec:
        if pending:
            out.append(pending + char)
            pending = ""
        elif char.isdigit():
            pending = char
        else:
            out.append(char)
    assert not pending, f"规格结尾有多余数字: {spec!r}"
    return out


def spec_of(counts: list[int]) -> list[str]:
    return tiles.to_codes(tiles.tiles_of(counts))


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def call_endpoint(
    server: str,
    hand: list[str],
    draw: str,
    count: int,
    piao: int,
    base: int = 1,
    attempts: int = 5,
) -> Outcome:
    body = json.dumps(
        {"hand": hand, "draw": draw, "chain": {"count": count, "piao": piao}, "base": base}
    ).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(
            server.rstrip("/") + FAN_CALC_PATH, data=body, method="POST"
        )
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=20, context=_ssl_context()) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            if exc.code == 429 and attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
                continue
            return Outcome(False, False, 0, (), error=raw)
        return Outcome(
            hu=bool(payload.get("hu")),
            baotou=bool(payload.get("baotou")),
            fan=int(payload.get("fan") or 0),
            detail=tuple(payload.get("detail") or ()),
            scores=payload.get("scores"),
        )
    return Outcome(False, False, 0, (), error="RATE_LIMITED 重试耗尽")


def local_outcome(counts: list[int], draw: int, count: int, piao: int, base: int = 1) -> Outcome:
    try:
        result = fan_module.compute_fan(
            counts, draw, 0, chain_count=count, piao_count=piao
        )
    except fan_module.FanError as exc:
        return Outcome(False, False, 0, (), error=str(exc))
    if not result.hu:
        return Outcome(False, result.baotou, 0, ())
    return Outcome(
        hu=True,
        baotou=result.baotou,
        fan=result.fan,
        detail=result.detail,
        scores=score_module.settle(result.fan, base).to_api_shape(),
    )


def random_winning_hand(rng: random.Random) -> list[int]:
    """生成一个合法的 14 张手牌（每牌种不超过 4 张），偏向成胡形态。"""
    kinds = [t for t in range(tiles.TILE_KINDS) if t != tiles.GOD]
    usage = [0] * tiles.TILE_KINDS

    def take(tile: int, amount: int) -> bool:
        if usage[tile] + amount > tiles.COPIES_PER_KIND:
            return False
        usage[tile] += amount
        return True

    if rng.random() < 0.25:
        picked = rng.sample(kinds, 7)
        for tile in picked:
            take(tile, 2)
        if rng.random() < 0.4:
            # 把一对提升为四张，同时撤掉另一对，以保持 14 张
            take(picked[0], 2)
            usage[picked[1]] = 0
        return usage

    for _ in range(50):
        pair_tile = rng.choice(kinds)
        if take(pair_tile, 2):
            break
    starts = [s for s in range(tiles.TILE_KINDS) if tiles.run_is_valid(s)]
    built = 0
    for _ in range(200):
        if built == 4:
            break
        if rng.random() < 0.45:
            if take(rng.choice(kinds), 3):
                built += 1
        else:
            start = rng.choice(starts)
            if all(usage[start + offset] < tiles.COPIES_PER_KIND for offset in range(3)):
                for offset in range(3):
                    take(start + offset, 1)
                built += 1
    for _ in range(rng.randint(0, 2)):
        present = [t for t in kinds if usage[t] > 0]
        if not present or usage[GOD] >= tiles.COPIES_PER_KIND:
            break
        pick = rng.choice(present)
        usage[pick] -= 1
        usage[GOD] += 1
    return usage


def random_case(rng: random.Random) -> tuple[list[int], int, int, int]:
    """返回 ``(13 张手牌, 摸牌, 动作链次数, 飘次数)``，保证每牌种不超过 4 张。"""
    if rng.random() < 0.72:
        drawn = random_winning_hand(rng)
        present = [t for t, amount in enumerate(drawn) if amount > 0]
        draw = rng.choice(present)
        hand = list(drawn)
        hand[draw] -= 1
    else:
        wall = [t for t in range(tiles.TILE_KINDS) for _ in range(tiles.COPIES_PER_KIND)]
        rng.shuffle(wall)
        hand = tiles.counts_from(wall[:HAND_SIZE])
        draw = rng.choice(
            [t for t in range(tiles.TILE_KINDS) if hand[t] < tiles.COPIES_PER_KIND]
        )
    count = rng.choice([0, 0, 1, 1, 2, 3, 4, 5, 6])
    gods = hand[GOD] + (1 if draw == tiles.GOD else 0)
    piao = min(rng.randint(0, count), max(0, FOUR_GODS - gods))
    return hand, draw, count, piao


def compare(name: str, expected: Outcome, actual: Outcome) -> str | None:
    if expected.error or actual.error:
        if (expected.error is None) == (actual.error is None):
            return None
        return f"{name}: 错误不一致\n  端点: {expected.error}\n  本地: {actual.error}"
    problems = []
    if expected.hu != actual.hu:
        problems.append(f"hu 端点={expected.hu} 本地={actual.hu}")
    if expected.hu:
        if expected.baotou != actual.baotou:
            problems.append(f"baotou 端点={expected.baotou} 本地={actual.baotou}")
        if expected.fan != actual.fan:
            problems.append(f"fan 端点={expected.fan} 本地={actual.fan}")
        if expected.detail != actual.detail:
            problems.append(f"detail 端点={list(expected.detail)} 本地={list(actual.detail)}")
        if expected.scores != actual.scores:
            problems.append(f"scores 端点={expected.scores} 本地={actual.scores}")
    return f"{name}: " + "; ".join(problems) if problems else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="与官方番型端点对拍")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--cases", type=int, default=200, help="随机用例数")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-report", type=int, default=12)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    failures: list[str] = []
    checked = 0
    wins = 0

    plan: list[tuple[str, list[int], int, int, int]] = []
    for spec, count, piao in BUILTIN_CASES:
        hand = tiles.counts_from(tiles.parse_all(tokens(spec)))
        drawable = [t for t in range(tiles.TILE_KINDS) if hand[t] < tiles.COPIES_PER_KIND]
        plan.append((f"内置 {spec}", hand, rng.choice(drawable), count, piao))
    for index in range(args.cases):
        hand, draw, count, piao = random_case(rng)
        plan.append((f"随机 #{index}", hand, draw, count, piao))

    for name, hand, draw, count, piao in plan:
        assert tiles.total_tiles(hand) == HAND_SIZE, f"{name}: 手牌不是 13 张"
        assert hand[draw] + 1 <= tiles.COPIES_PER_KIND, f"{name}: 摸牌超过 4 张"
        expected = call_endpoint(args.server, spec_of(hand), tiles.to_code(draw), count, piao)
        actual = local_outcome(hand, draw, count, piao)
        time.sleep(REQUEST_INTERVAL)
        checked += 1
        if expected.hu:
            wins += 1
        problem = compare(name, expected, actual)
        if problem:
            failures.append(problem)

    print(f"比对 {checked} 例（其中胡牌 {wins} 例），不一致 {len(failures)} 例")
    for line in failures[: args.max_report]:
        print("  ✗", line)
    if len(failures) > args.max_report:
        print(f"  ... 另有 {len(failures) - args.max_report} 例未展示")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
