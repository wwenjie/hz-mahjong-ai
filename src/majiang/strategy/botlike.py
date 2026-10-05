"""`botlike` 决策器：把 Stage B 的 **bot 出牌预测器**（GBDT）当对手模型用。

**为什么它存在**（A 2026-10-06 01:57 提出的方向）：我们**所有离线 A/B 的对手都是我们自己**（`field` 只能是我们的档位），
而对手是**对称**的 ⇒ 任何**利用对手行为**的改动（喂牌/防守/副露互动/竞速）在那种场里**天然被压掉**。
今天最硬的一条 venue 证据正指向这里：**我们的财神/爆头行为在自对弈里已与强 bot 同量级，差距只在真机出现**。

**做法**：`agent/out/stage-b-gbdt.joblib` 是一个 **77.4% top-1 的 bot 选择预测器**（Stage B 的负结果模型，
但作为"对手模型"它正是我们要的东西）。把它包成决策器，放进 `ab_test --field botlike`，
自对弈的另三座就变成**bot 行为分布**而不是我们自己 ⇒ **A/B 场地从「对自己」变成「对 bot-like」**。

**特征口径必须与训练时逐位一致**（防 train-serve skew）：29 维 `features.extract(situation)` +
5 个候选字段 `main_total / shanten / wait_copies / ukeire_exact / wait_kinds`，
定义与 `agent/verify/stage_a_dataset_export.py`（C 的数据底座）逐项相同。

**纪律**：① 本类**只用于测量**（作为 `field`），**不作为待采纳的候选臂**；
② 一律用 `GuardedDecider` 包（异常/超预算 ⇒ 回退 `FirstLegalDecider`），**绝不把 bot 的非法/超时行为模仿进来**；
③ 它的可信度由**保真度门**保证：在真机 replay 上的 top-1 一致率必须与训练报告（77.4%）相符。
"""

from __future__ import annotations

from collections.abc import Sequence

from majiang.rules import shanten as shanten_module
from majiang.rules import tiles, win as win_module
from majiang.rules.action import DISCARD, Action, legal_actions
from majiang.rules.situation import Situation
from majiang.strategy import features
from majiang.strategy import versions
from majiang.strategy.features import UKEIRE_MAX_SHANTEN  # noqa: F401
from majiang.strategy.policy import (
    Mode,
    PolicyConfig,
    HeuristicDecider,
    _wait_copies,
)

UKEIRE_MAX_SHANTEN = 3  # 与训练侧导出器同值（`stage_a_dataset_export.py`）
DEFAULT_MODEL = "agent/out/stage-b-gbdt.joblib"


class BotLikeDecider:
    """GBDT bot 出牌预测器 → argmax 决策器。"""

    def __init__(self, model_path: str = DEFAULT_MODEL) -> None:
        import joblib  # 局部导入：只有真用这个决策器时才需要 sklearn/joblib

        self.model = joblib.load(model_path)
        # 只借它的 `_score_discard`（`main_total`/`shanten`）与决策器接口，不借它的选择逻辑。
        #
        # **必须是 v6**（A 2026-10-06 03:45 的保真度 diff 抓到的第一处 train-serve skew）：
        # 数据集的 `main_total` 由 `agent/verify/stage_a_dataset_export.py:352` 的 `versions.build("v6", ...)` 产出，
        # 而我最初用了 `PolicyConfig.for_mode(...)`（= **默认档**）⇒ 34 维里 `main_total`（idx 29）在**全部 30 个抽样点**都不一致（最大偏差 95）。
        # 这类「训练/推理用不同配置」的偏差正是保真度门要抓的东西；本行是它的修复。
        self.scorer = versions.build("v6", Mode.QUALIFIER)
        # `GuardedDecider` 会读 `inner.name` 写日志（它兜底/留痕时要标档位名）。
        self.name = "botlike"

    def feature_rows(self, situation: Situation, candidates: Sequence[Action]):
        """构造候选级 34 维向量——**与训练侧（`stage_a_dataset_export.py:103-155`）逐行同口径**。

        **为什么这一段必须与导出器一字不差**（A 2026-10-06 03:41 的反例）：
        我第一版按「每个候选都算」写，而导出器的次级字段**只在 `do_tie and s in tied` 时才算**
        （`tied` = 与最优候选同向听的那些；`do_tie` = 并列≥2 且向听 ≤ `UKEIRE_MAX_SHANTEN`），
        且 `wait_aware` 由 **`top_sh == 0`** 判定（不是「该候选的 `wait_copies` 是否为 0」）。
        ⇒ 结果是：反例 2/3 里**我算了、数据集是 0**（反例 3 差到 95）——**train-serve skew**。
        现在逐行对齐：同样的 `top_sh/tied/do_tie/wait_aware` 门，同样的 `None` 语义，
        同样的「`wait_aware` ⇒ `wait_copies`+`wait_kinds`；否则 ⇒ `ukeire_exact`」二选一。
        """
        visible = shanten_module.visible_counts(
            situation.hand.counts,
            [meld.tiles for meld in situation.all_melds],
            situation.discards,
        )
        situation_features = list(features.extract(situation))
        meld_count = situation.hand.meld_count
        scores = [self.scorer._score_discard(situation, action) for action in candidates]
        if not scores:
            return [], []
        top_sh = scores[0].shanten
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
                    wc = _wait_copies(counts, meld_count, visible, tile)
                    if wc is not None:
                        wait_copies = wc
                    try:
                        wait_kinds = len(win_module.winning_draws(counts, meld_count))
                    except ValueError:
                        wait_kinds = 0
                else:
                    try:
                        entries = shanten_module.ukeire(
                            counts, meld_count, visible=visible, memo=memo
                        )
                        ukeire_exact = sum(copy for _, copy in entries)
                    except Exception:  # noqa: BLE001
                        ukeire_exact = 0
            rows.append(
                situation_features
                + [
                    float(score.total),
                    float(score.shanten),
                    float(wait_copies or 0),
                    float(ukeire_exact or 0),
                    float(wait_kinds or 0),
                ]
            )
            kept.append(action)
        return rows, kept

    def choose(
        self, situation: Situation, actions: Sequence[Action], *, budget_ms: int = 1800
    ) -> Action | None:
        import numpy as np

        candidates = [action for action in actions if action.kind == DISCARD]
        if not candidates:
            return None
        rows, kept = self.feature_rows(situation, candidates)
        if not rows:
            return None
        probabilities = self.model.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
        return kept[int(np.argmax(probabilities))]


def build(mode: Mode = Mode.QUALIFIER, model_path: str = DEFAULT_MODEL) -> object:
    """工厂：返回受 `GuardedDecider` 保护的 `botlike` 决策器（与线上同一道兜底）。"""
    from majiang.runtime.decider import FirstLegalDecider, GuardedDecider

    return GuardedDecider(BotLikeDecider(model_path), fallback=FirstLegalDecider())


__all__ = ["BotLikeDecider", "build", "DEFAULT_MODEL", "legal_actions"]
