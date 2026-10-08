"""Transformer BC 模型（Mahjax 风格）。

架构：
- 输入 = dict observation（hand, discards, melds, action_history, scores, scalars）
- 每种观测先 embedding → 拼接 → Transformer encoder → policy head (34) + value head (1)
- 策略头输出 34 维 logits（出牌决策）
- 价值头输出标量（用于 PPO 的 critic）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TileEmbedding(nn.Module):
    """牌型 embedding（0-33 + padding=34）。"""

    def __init__(self, d_model: int, num_tiles: int = 35):
        super().__init__()
        self.emb = nn.Embedding(num_tiles, d_model, padding_idx=34)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,) int64; -1 (padding) → padding_idx=34
        x = torch.where(x < 0, torch.full_like(x, 34), x)
        return self.emb(x)


class ObservationEncoder(nn.Module):
    """把 dict observation 编码成固定维度向量序列。

    手牌编码：从 (34,) 计数展开为 (14,) 牌 id 序列（-1 填充），保留逐张牌身份。
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # Tile embeddings（hand_tiles/discards/melds/action_history，值域 0-33 + padding=34）
        self.tile_emb = TileEmbedding(d_model)

        # Scores: (4,) → 每值一个 token
        self.score_emb = nn.Linear(1, d_model)

        # Scalars: god/wall_remaining/turn/phase/target/seat/dealer → 每值一个 token
        self.scalar_emb = nn.Linear(1, d_model)

        self.output_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _counts_to_tiles(counts: torch.Tensor, max_len: int = 14) -> torch.Tensor:
        """把 (batch, 34) 计数展开成 (batch, max_len) 牌 id 序列，-1 填充。

        全向量化实现（零 .item()、零 Python 循环），避免 GPU→CPU 同步瓶颈。
        与原版双重循环输出完全一致：按 tile_id 顺序、每张牌重复 min(count, 4) 次、截断到 max_len。
        """
        batch_size = counts.shape[0]
        device = counts.device
        counts_c = counts.clamp(0, 4).long()  # (batch, 34)

        # 展平：每个 (batch, tile) 位置的重复次数
        flat_counts = counts_c.reshape(-1)  # (batch*34,)
        tile_ids = torch.arange(34, device=device).repeat(batch_size)  # (batch*34,)
        owner = torch.arange(batch_size, device=device).repeat_interleave(34)  # (batch*34,)

        # 展开所有牌（变长 1D），及每张牌所属的 batch 行
        expanded = tile_ids.repeat_interleave(flat_counts)
        exp_owner = owner.repeat_interleave(flat_counts)

        # 每张牌在其 batch 行内的序号 = 全局序号 - 该行起始全局序号
        counts_per_row = counts_c.sum(dim=1)  # (batch,)
        row_start = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            row_start[1:] = counts_per_row.cumsum(0)[:-1]
        global_idx = torch.arange(expanded.shape[0], device=device)
        pos_in_row = global_idx - row_start[exp_owner]

        # 截断到 max_len 并 scatter 写入
        mask = pos_in_row < max_len
        out = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
        out[exp_owner[mask], pos_in_row[mask]] = expanded[mask]
        return out

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = obs["hand"].shape[0]
        tokens = []

        # Hand: (batch, 34) counts → (batch, 14) tile ids → 14 tokens
        hand_tiles = self._counts_to_tiles(obs["hand"].long())  # (batch, 14)
        tokens.append(self.tile_emb(hand_tiles))  # (batch, 14, d_model)

        # Discards: (batch, 4, 24) → 96 tokens
        discards = obs["discards"].long().view(batch_size, -1)
        tokens.append(self.tile_emb(discards))

        # Melds: (batch, 4, 4, 3) → 48 tokens
        melds = obs["melds"].long().view(batch_size, -1)
        tokens.append(self.tile_emb(melds))

        # Action history: (batch, 200) → 200 tokens
        tokens.append(self.tile_emb(obs["action_history"].long()))

        # Scores: (batch, 4) → 4 tokens
        scores = obs["scores"].float()
        tokens.append(self.score_emb(scores.unsqueeze(-1)))  # (batch, 4, d_model)

        # Scalars: 7 个各一个 token
        scalars = torch.stack([
            obs["god"].float(),
            obs["wall_remaining"].float(),
            obs["turn"].float(),
            obs["phase"].float(),
            obs["target"].float(),
            obs["seat"].float(),
            obs["dealer"].float(),
        ], dim=-1)  # (batch, 7)
        tokens.append(self.scalar_emb(scalars.unsqueeze(-1)))  # (batch, 7, d_model)

        # Total: 14 + 96 + 48 + 200 + 4 + 7 = 369 tokens
        x = torch.cat(tokens, dim=1)
        return self.output_proj(x)


class MahjongTransformer(nn.Module):
    """Transformer encoder + policy/value heads。"""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.encoder = ObservationEncoder(d_model)

        # Positional encoding (learnable)
        # seq_len = 14(hand) + 96(discards) + 48(melds) + 200(action_history) + 4(scores) + 7(scalars) = 369
        self.pos_emb = nn.Parameter(torch.randn(1, 369, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Policy head: 34 classes (discard)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 34),
        )

        # Value head: scalar
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dict of tensors

        Returns:
            policy_logits: (batch, 34)
            value: (batch, 1)
        """
        # Encode observation
        x = self.encoder(obs)  # (batch, 389, d_model)

        # Add positional encoding
        x = x + self.pos_emb

        # Transformer
        x = self.transformer(x)  # (batch, 389, d_model)

        # Pool: mean over sequence
        x = x.mean(dim=1)  # (batch, d_model)

        # Heads
        policy_logits = self.policy_head(x)  # (batch, 34)
        value = self.value_head(x)  # (batch, 1)

        return policy_logits, value

    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """推理接口：返回出牌概率分布。"""
        self.eval()
        with torch.no_grad():
            policy_logits, _ = self.forward(obs)
            return F.softmax(policy_logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
