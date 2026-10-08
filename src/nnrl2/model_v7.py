"""Suphx 式多头 BC 模型 v7：共享主干 + 5 个独立二分类头 + tile 头。

设计依据 Suphx（arXiv:2003.13590）的分离模型范式：
- 每个动作（discard/chi/peng/gang/hu）一个独立二分类头（sigmoid），
  回答「这个状态下要不要做这个动作？」
- tile 头：34 类（出哪张牌），仅在 discard 样本上计算损失
- hu 头保留学习（与 Suphx 纯规则不同：我们的规则含"漂"决策，值得学）
- pass 不设头：所有头都拒绝即 pass

与 v6（6 类 softmax）的本质区别：
  softmax 强制六选一，pass 占 80% 会把稀有动作的梯度淹没；
  独立 sigmoid 头各自训练，类别不均衡用 pos_weight 处理，
  且训练时只对「该决策点可行的动作」（avail mask）计算损失——
  不可行的动作既不当正样本也不当负样本。

推理决策顺序（与 Suphx 决策流一致）：
  hu > gang > peng > chi > discard；全否则 pass。

架构：
- 输入 = dict observation（与 v5/v6 相同）
- 共享主干 = ObservationEncoder + Transformer（与 v5/v6 相同）
- 5 个二分类头 = Linear(d_model → 1) 各带隐藏层
- tile 头 = Linear(d_model → 34)
- value 头 = Linear(d_model → 1)（PPO 用）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnrl2.model import ObservationEncoder, count_parameters


class MahjongTransformerV7(nn.Module):
    """Suphx 式多头 Transformer：5 个二分类头 + tile 34 类。"""

    # 动作类型常量（与 v6 数据集一致）
    ACTION_DISCARD = 0
    ACTION_CHI = 1
    ACTION_PENG = 2
    ACTION_GANG = 3
    ACTION_HU = 4
    ACTION_PASS = 5
    NUM_ACTIONS = 6
    TILE_PAD = 34

    # 二分类头对应的动作（pass 不设头）
    HEAD_ACTIONS = (ACTION_HU, ACTION_GANG, ACTION_PENG, ACTION_CHI, ACTION_DISCARD)
    HEAD_NAMES = ("hu", "gang", "peng", "chi", "discard")

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
        self.pos_emb = nn.Parameter(torch.randn(1, 369, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def _binary_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        # 5 个独立二分类头
        self.hu_head = _binary_head()
        self.gang_head = _binary_head()
        self.peng_head = _binary_head()
        self.chi_head = _binary_head()
        self.discard_head = _binary_head()

        self.tile_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 34),
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _backbone(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.encoder(obs)
        x = x + self.pos_emb
        x = self.transformer(x)
        return x.mean(dim=1)  # (batch, d_model)

    def heads(self) -> tuple[nn.Module, ...]:
        """按 HEAD_ACTIONS 顺序返回 5 个头。"""
        return (self.hu_head, self.gang_head, self.peng_head, self.chi_head, self.discard_head)

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            binary_logits: (batch, 5) — 顺序为 (hu, gang, peng, chi, discard)
            tile_logits:   (batch, 34)
            value:         (batch, 1)
        """
        x = self._backbone(obs)
        binary_logits = torch.cat([h(x) for h in self.heads()], dim=1)  # (batch, 5)
        tile_logits = self.tile_head(x)
        value = self.value_head(x)
        return binary_logits, tile_logits, value

    @torch.no_grad()
    def predict_full(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理接口：返回 (动作类型概率[batch,6], tile 概率[batch,34])。

        6 类概率由二分类头按优先级合成：
          p(hu)=σ(hu_logit)
          p(gang)=(1-p_hu)·σ(gang_logit)
          ...依优先级链式展开，p(pass)=剩余概率。
        """
        self.eval()
        binary_logits, tile_logits, _ = self.forward(obs)
        sig = torch.sigmoid(binary_logits)  # (batch, 5) hu/gang/peng/chi/discard
        p_hu, p_gang, p_peng, p_chi, p_dis = sig.unbind(dim=1)
        rem = 1.0 - p_hu
        out = torch.zeros(sig.shape[0], self.NUM_ACTIONS, device=sig.device)
        out[:, self.ACTION_HU] = p_hu
        out[:, self.ACTION_GANG] = rem * p_gang; rem *= (1 - p_gang)
        out[:, self.ACTION_PENG] = rem * p_peng; rem *= (1 - p_peng)
        out[:, self.ACTION_CHI] = rem * p_chi; rem *= (1 - p_chi)
        out[:, self.ACTION_DISCARD] = rem * p_dis; rem *= (1 - p_dis)
        out[:, self.ACTION_PASS] = rem
        return out, F.softmax(tile_logits, dim=-1)

    @torch.no_grad()
    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """兼容 v5 接口：返回出牌概率分布。"""
        self.eval()
        _, tile_logits, _ = self.forward(obs)
        return F.softmax(tile_logits, dim=-1)


__all__ = ["MahjongTransformerV7", "count_parameters"]
