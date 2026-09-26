"""决策器版本库（champion / challenger）。

**用途**：保留历史版本，使**任意两版可以直接对打**。做法是把每个「冠军」的**变体开关**
冻结成一个具名版本，`--decider v1` / `--decider v2` 即可指名：

    uv run python tools/ab_test.py --treatment v3 --baseline v2

只冻结**变体开关**（`tiebreak` / `meld_tolerance` / `chase_baotou` / `feed_weight` …），
**不冻结** `base_score` / `you_cai_bi_kao` —— 那两个由运行时从服务端注入
（`HeuristicDecider.configure()`）。冻结它们会让版本在真机上与服务端实际规则不符。

**纪律（重要）**：每次改默认档**必须**同时新增一个版本快照。否则「旧默认档」的行为就此丢失，
以后再也无法与新默认直接对打，只能拿一个近似的档位当替身——而 `blocks` 那次就是这么补救的。
新增时写清日期与改动理由。

**它不替代 `DECIDERS`**：`DECIDERS` 是「尚未胜出的实验档位」，本文件是「已胜出并冻结的版本」。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .policy import HeuristicDecider, Mode, PolicyConfig


@dataclass(frozen=True, slots=True)
class Version:
    id: str
    date: str
    note: str
    knobs: Mapping[str, Any] = field(default_factory=dict)


VERSIONS: tuple[Version, ...] = (
    Version(
        id="v1",
        date="2026-09-26",
        note=(
            "本轮之前的线上基线：同向听用骨架厚度（blocks）做次排序、"
            "chase_baotou 开启、喂牌权重 3.0、吃碰闸门 strict。"
        ),
        knobs={"tiebreak": "blocks"},
    ),
    Version(
        id="v2",
        date="2026-09-27",
        note=(
            "exact-ukeire：同向听改用**精确**进张次排序（只在打完仍 1 向听的前 2 候选上算，"
            "带 0.6 秒墙钟上限）。两个独立种子共 2000 配对场复现显著："
            "名次分 +0.285（t 4.03）· 胡次数 +0.088（t 4.46）· 总得分 +1.525（t 2.55）。"
            "真机侧机制指标同向：出牌后到听率 22.0% → 23.8%（对手 +0.7，差分 +1.1）。"
            "**当前默认档。**"
        ),
        knobs={},
    ),
)

BY_ID: dict[str, Version] = {version.id: version for version in VERSIONS}


def build(version_id: str, mode: Mode) -> HeuristicDecider:
    """按版本号构造决策器。未知版本号抛 ``KeyError``（由调用方转成启动错误）。"""
    return HeuristicDecider(PolicyConfig.for_mode(mode, **dict(BY_ID[version_id].knobs)))


def is_version(name: str) -> bool:
    return name in BY_ID


def describe() -> str:
    return "\n".join(
        f"  {version.id}  {version.date}  {version.note.splitlines()[0]}"
        for version in VERSIONS
    )
