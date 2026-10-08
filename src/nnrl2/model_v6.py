"""双头 BC 模型 v6：共享主干 + 动作类型头 + tile 头。

设计依据 Suphx（arXiv:2003.13590）的分离模型范式：
- 动作类型头：6 类（discard/chi/peng/gang/hu/pass），解决单头网络被 discard 样本淹没的问题
- tile 头：34 类（出哪张牌），仅在动作类型为 discard 时计算损失
- 类别加权：稀有动作（chi/peng/gang/hu/pass）按频率倒数加权，防止梯度被 discard 主导

架构：
- 输入 = dict observation（与 v5 相同）
- 共享主干 = ObservationEncoder + Transformer（与 v5 相同）
- 动作头 = Linear(d_model → 6)
- tile 头 = Linear(d_model → 34)
- value 头 = Linear(d_model → 1)（PPO 用）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnrl2.model import ObservationEncoder, count_parameters


class MahjongTransformerV6(nn.Module):
    """双头 Transformer：动作类型（6 类）+ tile（34 类）。"""

    # 动作类型常量
    ACTION_DISCARD = 0
    ACTION_CHI = 1
    ACTION_PENG = 2
    ACTION_GANG = 3
    ACTION_HU = 4
    ACTION_PASS = 5
    NUM_ACTIONS = 6
    TILE_PAD = 34  # 非 tile 动作的占位

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
        self.pos_emb = nn.Parameter(torch.randn(1, 369, d_model))

        # Transformer encoder（共享主干）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 动作类型头：6 类
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.NUM_ACTIONS),
        )

        # tile 头：34 类（出哪张牌）
        self.tile_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 34),
        )

        # Value head: scalar（PPO 用）
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: (batch, 6) 动作类型
            tile_logits:   (batch, 34) 出牌选择
            value:         (batch, 1)
        """
        x = self.encoder(obs)
        x = x + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)  # (batch, d_model)

        action_logits = self.action_head(x)
        tile_logits = self.tile_head(x)
        value = self.value_head(x)
        return action_logits, tile_logits, value

    @torch.no_grad()
    def predict_full(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理接口：返回 (动作类型概率, tile 概率)。"""
        self.eval()
        action_logits, tile_logits, _ = self.forward(obs)
        return F.softmax(action_logits, dim=-1), F.softmax(tile_logits, dim=-1)

    @torch.no_grad()
    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """兼容 v5 接口：返回出牌概率分布。"""
        self.eval()
        _, tile_logits, _ = self.forward(obs)
        return F.softmax(tile_logits, dim=-1)


__all__ = ["MahjongTransformerV6", "count_parameters"]
