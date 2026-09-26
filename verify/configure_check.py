#!/usr/bin/env python3
"""独立验证 configure 修复（曾让真机所有变体档位静默退化成默认档的 bug）。

行为级断言（不只是读代码）：
1. 非默认 PolicyConfig（tiebreak / meld_tolerance / chase_baotou 全开）经 configure()
   注入赛事配置后，所有变体字段**保持原值**（修复前会被静默重置为默认）
2. base_score / you_cai_bi_kao 被赛事配置正确覆盖（这是 configure 的本职）
3. decider.name 带变体后缀（日志可区分手臂的前提）

用法：uv run python verify/configure_check.py
"""
from dataclasses import fields

from majiang.client.models import TournamentConfig
from majiang.strategy.policy import (
    Commitment,
    HeuristicDecider,
    MeldTolerance,
    Mode,
    PolicyConfig,
)

FAKE_TOURNAMENT = TournamentConfig.parse({
    "M": 4, "Rounds": 8, "BaseScore": 2, "YouCaiBiKao": True,
    "DiscardTimeoutSec": 3, "PengTimeoutSec": 1, "ChiTimeoutSec": 1, "Kind": "qualifier",
})


def main():
    variant = PolicyConfig(  # 全部变体字段取非默认值
        mode=Mode.FINAL,
        tiebreak="blocks",
        meld_tolerance=MeldTolerance.EQUAL,
        chase_baotou=False,
        commitment=Commitment.MELD,
        preserve_god=True,
        natural_route=True,
        route_aware=True,
        pair_route_pairs=4,
        shanten_weight=99.0,
    )
    decider = HeuristicDecider(variant)
    before = {f.name: getattr(decider.config, f.name) for f in fields(PolicyConfig)}

    decider.configure(FAKE_TOURNAMENT)
    after = {f.name: getattr(decider.config, f.name) for f in fields(PolicyConfig)}

    failures = []
    for name in before:
        if name in ("base_score", "you_cai_bi_kao"):
            continue
        if before[name] != after[name]:
            failures.append(f"{name}: {before[name]!r} -> {after[name]!r}（被重置）")
    if after["base_score"] != 2:
        failures.append(f"base_score 未覆盖: {after['base_score']}")
    if after["you_cai_bi_kao"] is not True:
        failures.append(f"you_cai_bi_kao 未覆盖: {after['you_cai_bi_kao']}")

    name = decider.name
    print(f"decider.name = {name!r}")
    if name == "heuristic":
        failures.append("name 未带变体后缀（日志里无法区分手臂）")

    if failures:
        print("FAIL:")
        for f_ in failures:
            print(" -", f_)
        raise SystemExit(1)
    print("OK: configure 保留全部变体字段，仅覆盖赛事配置字段，name 带后缀")


if __name__ == "__main__":
    main()
