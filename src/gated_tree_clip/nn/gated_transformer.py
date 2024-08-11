from typing import Optional, override

import torch
import torch.nn as nn
from utils.clogging import getColoredLogger

from .gated_attention import ResidualAttentionWithSyntacticDistanceBlock
from .syntactic_distance_gate import SyntacticDistanceGate
from .transformer import TextTransformer, VisionTransformer

logger = getColoredLogger(__name__)

__all__ = ["TransformerWithSyntacticDistance", "VisionTransformerWithSyntacticDistance", "TextTransformerWithSyntacticDistance"]


class TransformerWithSyntacticDistance(nn.Module):
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
        self.induction_head = SyntacticDistanceGate(embed_dim, num_heads, batch_first=batch_first)
        self.res_attn_blocks = nn.ModuleList(
            [
                ResidualAttentionWithSyntacticDistanceBlock(
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
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
        is_checkpoint: bool = False,
        return_weight: bool = False,
        return_distance: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        atention_gate, distance = self.induction_head(x)

        for i, res_attn_block in enumerate(self.res_attn_blocks):
            if is_checkpoint and not torch.jit.is_scripting():
                x, attn_weight, distance = torch.utils.checkpoint.checkpoint(res_attn_block, x, None, None, attention_mask)
            else:
                x, attn_weight, distance = res_attn_block(
                    x, attn_mask=attention_mask, attn_gate=attention_gate if i == 0 else None
                )

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        ret = [x]
        if return_weight:
            ret.append(attn_weight)
        if return_distance:
            ret.append(distance)
        return tuple(ret) if len(ret) > 1 else ret[0]


class VisionTransformerWithSyntacticDistance(VisionTransformer, TransformerWithSyntacticDistance): ...


class TextTransformerWithSyntacticDistance(TextTransformer, TransformerWithSyntacticDistance): ...
