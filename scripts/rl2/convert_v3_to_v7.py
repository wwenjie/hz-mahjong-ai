#!/usr/bin/env python
"""把 MahjongV3MultiTask checkpoint 转成 MahjongTransformerV7 兼容格式。

v3 的 backbone 就是原生 MahjongTransformerV7（同构），转换规则：
  - `backbone.<key>` → `<key>`
  - 丢弃 `final_score_head.*` 和 `other_hands_head.*`（推理不需要辅助头）
  - 保留元信息（epoch/args），标记 aux_heads_trained=True
"""

from __future__ import annotations

import sys
import torch

def convert(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    new_sd: dict[str, torch.Tensor] = {}
    dropped = 0
    for k, v in sd.items():
        if k.startswith("backbone."):
            new_sd[k[len("backbone."):]] = v
        else:
            dropped += 1  # final_score_head / other_hands_head
    print(f"保留 backbone 权重 {len(new_sd)} 个，丢弃辅助头 {dropped} 个")

    # 验证：与 MahjongTransformerV7 的 key 完全对齐（strict 加载测试）
    sys.path.insert(0, "src")
    from nnrl2.model_v7 import MahjongTransformerV7
    probe = MahjongTransformerV7()
    probe.load_state_dict(new_sd, strict=True)
    n = sum(p.numel() for p in probe.parameters())
    print(f"strict 加载验证通过，主模型参数量 {n/1e6:.2f}M")

    out = {
        "model_state_dict": new_sd,
        "epoch": ckpt.get("epoch"),
        "args": ckpt.get("args"),
        "aux_heads_trained": True,
        "converted_from": src,
    }
    torch.save(out, dst)
    print(f"已保存 -> {dst}")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "runs/bc_v3_oracle.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "runs/bc_v3_oracle_v7compat.pt"
    convert(src, dst)
