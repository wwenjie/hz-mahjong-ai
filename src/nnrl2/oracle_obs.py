"""Oracle（先知）观测编码器：公开 obs + 三家真实手牌。

合规红线：本模块只进训练管线（先知训练），交付的比赛模型绝不调用。
先知观测 = situation_to_obs 的完整输出 + oracle_hands 字段（3×34 其他家手牌计数）。
牌墙不单独喂——给了三家手牌后，牌墙分布可由公开信息（自家手牌+弃牌+副露）推导。

token 化：3 家手牌各展开为 14 张 tile token（-1 填充），共 42 个附加 token，
接在原 369 token 之后 → pos_emb 需 411 维。
"""

from __future__ import annotations

import numpy as np

from nnrl2.obs import TILE_KINDS, situation_to_obs

# 先知附加 token 数：3 家 × 14 张
ORACLE_EXTRA_TOKENS = 3 * 14  # 42
BASE_TOKENS = 369
ORACLE_TOKENS = BASE_TOKENS + ORACLE_EXTRA_TOKENS  # 411


def situation_to_oracle_obs(situation, seat: int, state) -> dict[str, np.ndarray]:
    """把 Situation + RoundState 转成先知观测（公开信息 + 三家真实手牌）。

    Args:
        situation: 主仓 Situation 对象（公开信息）
        seat: 当前玩家座位（ego-centric）
        state: RoundState（全知状态，state.seats[i].hand 为 34 维计数）

    Returns:
        dict of numpy arrays：与 situation_to_obs 相同键 + "oracle_hands" (3, 34)
    """
    obs = situation_to_obs(situation, seat)

    # 三家真实手牌（相对座位序：下家、对家、上家）
    oracle_hands = np.zeros((3, TILE_KINDS), dtype=np.int8)
    for i in range(1, 4):
        abs_seat = (seat + i) % 4
        for tile in range(TILE_KINDS):
            oracle_hands[i - 1, tile] = state.seats[abs_seat].hand[tile]
    obs["oracle_hands"] = oracle_hands
    return obs


def collate_oracle_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """把一批先知 obs 堆叠成 batch（numpy）。所有键 stack，oracle_hands (B,3,34)。"""
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list]) for k in keys}
