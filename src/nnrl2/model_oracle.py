"""Oracle（先知）模型：v7 架构 + 三家真实手牌 token。

继承 MahjongTransformerV7，改动：
1. pos_emb 369 → 411（前 369 维可从 BC checkpoint 热启动，后 42 随机初始化）
2. forward 额外消费 obs["oracle_hands"] (batch, 3, 34)：计数→3×14 tile tokens

合规红线：先知模型只进训练管线。蒸馏出的正常 v7 模型（无 oracle_hands 输入）
才是交付候选——本类不得出现在比赛 decider 路径中。
"""

from __future__ import annotations

import torch

from nnrl2.model_v7 import MahjongTransformerV7
from nnrl2.oracle_obs import BASE_TOKENS, ORACLE_EXTRA_TOKENS, ORACLE_TOKENS


class MahjongOracleV7(MahjongTransformerV7):
    """先知版 v7：输入追加 oracle_hands，位置编码扩到 411。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = torch.nn.Parameter(torch.randn(1, ORACLE_TOKENS, self.d_model))

    def _backbone(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # 公开 obs → 369 tokens（复用 v7 编码器）
        base_tokens = {k: v for k, v in obs.items() if k != "oracle_hands"}
        x = self.encoder(base_tokens)  # (batch, 369, d_model)

        # 先知手牌 → 42 tokens
        oh = obs["oracle_hands"].long()  # (batch, 3, 34)
        batch = oh.shape[0]
        oracle_tokens = [
            self.encoder._counts_to_tiles(oh[:, i, :]) for i in range(3)
        ]  # 3 × (batch, 14)
        oracle_tiles = torch.stack(oracle_tokens, dim=1).view(batch, -1)  # (batch, 42)
        xo = self.encoder.tile_emb(oracle_tiles)  # (batch, 42, d_model)

        x = torch.cat([x, xo], dim=1)  # (batch, 411, d_model)
        x = x + self.pos_emb
        x = self.transformer(x)
        return x.mean(dim=1)

    def load_base_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> None:
        """从 v7 BC checkpoint 热启动：全部权重直接迁移，pos_emb 前 369 维拷贝、后 42 保留随机。"""
        base_pos = state_dict["pos_emb"]  # (1, 369, d)，非破坏性读取
        filtered = {k: v for k, v in state_dict.items() if k != "pos_emb"}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert list(missing) == ["pos_emb"], f"missing keys: {missing}"
        with torch.no_grad():
            self.pos_emb[:, :BASE_TOKENS, :] = base_pos
        # 后 42 维保持 __init__ 的随机初始化


__all__ = ["MahjongOracleV7", "BASE_TOKENS", "ORACLE_EXTRA_TOKENS", "ORACLE_TOKENS"]
