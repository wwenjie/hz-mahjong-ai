"""将主仓 Situation 编码为 Mahjax 风格的 dict observation。

设计原则：
- **token 化**：牌用类型 id（0-33），手牌用计数向量，弃牌用序列
- **ego-centric**：当前玩家视角，其他玩家相对座位
- **无人工特征**：不计算向听/进张等启发式量（让 Transformer 自己学）
- **合规**：只用公开信息 + 自身手牌；不读对手暗牌

字段（对齐 Mahjax 的 dict observation）：
- hand: (34,) int8, 暗手各牌型计数 [0,4]
- discards: (4, 24) int8, 各家弃牌序列（相对座位），-1 填充
- melds: (4, 4, 3) int8, 各家副露 [kind_id, tile, concealed]，-1 填充
- action_history: (200,) int8, 本局动作历史（简化版），-1 填充
- scores: (4,) int32, 各家累计分（相对座位）
- god: () int8, 财神牌型
- wall_remaining: () int32, 牌墙剩余
- turn: () int8, 当前回合数
- phase: () int8, 0=摸打, 1=响应
- target: () int8, 被响应的牌（响应阶段），-1 否则
- seat: () int8, 当前玩家座位
- dealer: () int8, 庄家座位
"""

from __future__ import annotations

import numpy as np

# 常量（与主仓 tiles 模块对齐）
TILE_KINDS = 34
GOD = 33  # 白板
MAX_DISCARDS = 24  # 每局最大弃牌数
MAX_MELDS = 4  # 每家最多副露数
MAX_HISTORY = 200  # 动作历史最大长度

# Meld kind 映射到整数 id
_MELD_KIND_TO_ID = {"chi": 0, "peng": 1, "gang": 2}


def situation_to_obs(situation, seat: int) -> dict[str, np.ndarray]:
    """把主仓 Situation 转成 Mahjax 风格 dict observation。

    Args:
        situation: 主仓 Situation 对象
        seat: 当前玩家座位（ego-centric）

    Returns:
        dict of numpy arrays
    """
    obs = {}

    # --- hand: (34,) 暗手计数 ---
    hand = np.zeros(TILE_KINDS, dtype=np.int8)
    for tile in range(TILE_KINDS):
        hand[tile] = situation.hand.counts[tile]
    obs["hand"] = hand

    # --- discards: (4, MAX_DISCARDS) 各家弃牌序列 ---
    discards = np.full((4, MAX_DISCARDS), -1, dtype=np.int8)
    for i, seat_discards in enumerate(situation.discards):
        rel_seat = (i - seat) % 4  # 相对座位
        for j, tile in enumerate(seat_discards[:MAX_DISCARDS]):
            discards[rel_seat, j] = tile
    obs["discards"] = discards

    # --- melds: (4, MAX_MELDS, 3) 各家副露 ---
    # [kind_id, tile, concealed]
    melds = np.full((4, MAX_MELDS, 3), -1, dtype=np.int8)
    for i, seat_melds in enumerate(situation.melds):
        rel_seat = (i - seat) % 4
        for j, meld in enumerate(seat_melds[:MAX_MELDS]):
            kind_id = _MELD_KIND_TO_ID.get(meld.kind, -1)
            # 取 meld 的第一张牌作为代表（chi 有多张，取最小）
            tile = min(meld.tiles) if meld.tiles else -1
            melds[rel_seat, j, 0] = kind_id
            melds[rel_seat, j, 1] = tile
            melds[rel_seat, j, 2] = 1 if meld.concealed else 0
    obs["melds"] = melds

    # --- action_history: (MAX_HISTORY,) 简化动作历史 ---
    # 主仓没有现成 action_history，用 discards 拼接近似
    history = []
    for seat_idx in range(4):
        for tile in situation.discards[seat_idx][:MAX_DISCARDS]:
            if tile >= 0:
                history.append(tile)
    action_history = np.full(MAX_HISTORY, -1, dtype=np.int8)
    if history:
        action_history[:len(history)] = history[:MAX_HISTORY]
    obs["action_history"] = action_history

    # --- scores: (4,) 各家累计分 ---
    scores = np.zeros(4, dtype=np.int32)
    if situation.table.scores:
        for i in range(4):
            rel_seat = (i - seat) % 4
            scores[rel_seat] = situation.table.scores[i]
    obs["scores"] = scores

    # --- god: () 财神 ---
    # GodState 的字段名需要确认
    god_tile = getattr(situation.god, 'god', GOD)
    obs["god"] = np.int8(god_tile)

    # --- wall_remaining: () 牌墙剩余 ---
    obs["wall_remaining"] = np.int32(situation.table.wall_remaining)

    # --- turn: () 当前回合 ---
    obs["turn"] = np.int8(situation.turn)

    # --- phase: () 阶段 ---
    # 主仓 phase: "draw" 或响应阶段
    obs["phase"] = np.int8(0 if situation.phase == "draw" else 1)

    # --- target: () 被响应的牌 ---
    obs["target"] = np.int8(situation.offered_tile if situation.offered_tile is not None else -1)

    # --- seat: () 当前玩家座位 ---
    obs["seat"] = np.int8(seat)

    # --- dealer: () 庄家座位 ---
    obs["dealer"] = np.int8(situation.table.dealer_seat)

    return obs


def obs_to_flat(obs: dict[str, np.ndarray]) -> np.ndarray:
    """把 dict observation 展平成固定长度向量（用于简单基线模型）。"""
    features = []
    features.extend(obs["hand"])  # 34
    features.extend(obs["discards"].flatten())  # 4*24=96
    features.extend(obs["melds"].flatten())  # 4*4*3=48
    features.extend(obs["action_history"])  # 200
    features.extend(obs["scores"])  # 4
    features.append(obs["god"])  # 1
    features.append(obs["wall_remaining"])  # 1
    features.append(obs["turn"])  # 1
    features.append(obs["phase"])  # 1
    features.append(obs["target"])  # 1
    features.append(obs["seat"])  # 1
    features.append(obs["dealer"])  # 1
    return np.asarray(features, dtype=np.float32)


# 特征维度常量
FLAT_FEATURE_DIM = 34 + 96 + 48 + 200 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # = 389
