"""候选级 5 字段的公共构造器——训练导出与 `botlike` 推理共用（A 2026-10-06 03:41 裁决）。

**为什么必须共用**：train-serve skew 的根源是「训练导出口径」与「推理构造口径」各写一遍。
本函数是**唯一**的候选字段构造入口，两侧都调它 ⇒ skew 结构性消失。

口径（与 `stage_a_dataset_export.py:candidate_rows` 逐行一致）：
- `main_total` / `shanten`：`decider._score_discard().total` / `.shanten`
- `wait_copies` / `wait_kinds`：**只在并列层且听牌态**（`do_tie and s in tied and wait_aware`）才计算
- `ukeire_exact`：**只在并列层非听牌**（`do_tie and s in tied and not wait_aware`）才计算
- 不适用字段 = `None`（不是 0.0——0.0 会丢「不适用」语义）
"""
from __future__ import annotations

from typing import Sequence

from majiang.rules.action import Action, DISCARD
from majiang.rules.shanten import visible_counts, ukeire
from majiang.rules.win import winning_draws
from majiang.strategy import features

# 与 v5 的 `ukeire_max_shanten` 一致（policy.py:234）
UKEIRE_MAX_SHANTEN = 3


def candidate_dicts(
    decider,
    situation,
    candidates: Sequence[Action],
) -> tuple[list[dict], list[Action]]:
    """同 `candidate_features`，但返回 dict 列表（保留 None 语义）。

    供训练导出器用（需要 `is_bot`/`is_v5` 等额外字段时，在调用侧加）。
    """
    visible = visible_counts(
        situation.hand.counts,
        [meld.tiles for meld in situation.all_melds],
        situation.discards,
    )
    meld_count = situation.hand.meld_count

    scores = [decider._score_discard(situation, action) for action in candidates]
    if not scores:
        return [], []

    top_score = max(scores, key=lambda s: s.total)
    top_sh = top_score.shanten
    tied = [score for score in scores if score.shanten == top_sh]
    do_tie = len(tied) >= 2 and 0 <= top_sh <= UKEIRE_MAX_SHANTEN
    wait_aware = top_sh == 0
    memo: dict = {}

    fields: list[dict] = []
    kept: list[Action] = []
    for score, action in zip(scores, candidates):
        tile = action.tile
        if tile is None:
            continue
        counts = list(situation.hand.counts)
        counts[tile] -= 1

        wait_copies = ukeire_exact = wait_kinds = None
        if do_tie and any(score is item for item in tied):
            if wait_aware:
                from majiang.strategy.policy import _wait_copies
                wc = _wait_copies(counts, meld_count, visible, tile)
                if wc is not None:
                    wait_copies = wc
                try:
                    wait_kinds = len(winning_draws(counts, meld_count))
                except ValueError:
                    wait_kinds = 0
            else:
                try:
                    entries = ukeire(counts, meld_count, visible=visible, memo=memo)
                    ukeire_exact = sum(copy for _, copy in entries)
                except Exception:
                    ukeire_exact = 0

        fields.append({
            "tile": tile,
            "main_total": float(score.total),
            "shanten": float(score.shanten),
            "wait_copies": wait_copies,
            "ukeire_exact": ukeire_exact,
            "wait_kinds": wait_kinds,
        })
        kept.append(action)

    return fields, kept


def candidate_features(
    decider,
    situation,
    candidates: Sequence[Action],
) -> tuple[list[list[float]], list[Action]]:
    """对每个合法弃牌候选构造 34 维向量（29 维局面特征 + 5 个候选字段）。

    返回 (rows, kept_actions)。rows 的每个元素是 34 维 list[float]；
    kept_actions 与 rows 一一对应（tile 为 None 的候选被跳过）。
    """
    visible = visible_counts(
        situation.hand.counts,
        [meld.tiles for meld in situation.all_melds],
        situation.discards,
    )
    situation_features = list(features.extract(situation))
    meld_count = situation.hand.meld_count

    scores = [decider._score_discard(situation, action) for action in candidates]
    if not scores:
        return [], []
    # top_sh 必须取「total 最大的候选」的 shanten（导出器 `candidate_rows` 里 scores 按 total 降序排）
    top_score = max(scores, key=lambda s: s.total)
    top_sh = top_score.shanten
    tied = [score for score in scores if score.shanten == top_sh]
    do_tie = len(tied) >= 2 and 0 <= top_sh <= UKEIRE_MAX_SHANTEN
    wait_aware = top_sh == 0
    memo: dict = {}

    rows: list[list[float]] = []
    kept: list[Action] = []
    for score, action in zip(scores, candidates):
        tile = action.tile
        if tile is None:
            continue
        counts = list(situation.hand.counts)
        counts[tile] -= 1

        wait_copies = ukeire_exact = wait_kinds = None
        if do_tie and any(score is item for item in tied):
            if wait_aware:
                from majiang.strategy.policy import _wait_copies
                wc = _wait_copies(counts, meld_count, visible, tile)
                if wc is not None:
                    wait_copies = wc
                try:
                    wait_kinds = len(winning_draws(counts, meld_count))
                except ValueError:
                    wait_kinds = 0
            else:
                try:
                    entries = ukeire(counts, meld_count, visible=visible, memo=memo)
                    ukeire_exact = sum(copy for _, copy in entries)
                except Exception:
                    ukeire_exact = 0

        rows.append(
            situation_features
            + [
                float(score.total),
                float(score.shanten),
                float(wait_copies) if wait_copies is not None else 0.0,
                float(ukeire_exact) if ukeire_exact is not None else 0.0,
                float(wait_kinds) if wait_kinds is not None else 0.0,
            ]
        )
        kept.append(action)

    return rows, kept
