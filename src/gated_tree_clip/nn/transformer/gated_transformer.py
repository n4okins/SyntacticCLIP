from typing import Optional

import torch
import torch.nn as nn

from ..attention.res_gated_attn_block import ResidualGatedAttentionBlock
from ..misc import SyntacticDistanceGate

__all__ = [
    "SyntacticTransformer",
]


class SyntacticTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.induction_head = SyntacticDistanceGate(embed_dim, num_heads)
        self.res_attn_blocks = nn.ModuleList(
            [
                ResidualGatedAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=batch_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        atention_gate, distance = self.induction_head(x)

        for i, res_attn_block in enumerate(self.res_attn_blocks):
            x, attn_weight, distance = res_attn_block(
                x,
                attn_mask=attention_mask,
                attn_gate=attention_gate if i == 0 else None,
            )

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        return x, attn_weight, distance
