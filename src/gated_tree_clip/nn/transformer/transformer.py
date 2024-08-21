from typing import Optional

import torch
import torch.nn as nn

from ..attention.res_attn_block import ResidualAttentionBlock


class Transformer(nn.Module):
    """Basic Transformer
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of heads
        num_layers (int): Number of layers
        batch_first (bool): Batch first
    """

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
        self.res_attn_blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, sequence_length, embed_dim] if batch_first else [sequence_length, batch, embed_dim]
            attention_mask: [sequence_length, sequence_length]

        Returns:
            x: [batch, sequence_length, embed_dim] if batch_first else [sequence_length, batch, embed_dim]
            attn_weights: [batch or 1, num_layers, sequence_length, sequence_length]
        """

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        if return_weights:
            ret_weights = []

        for res_attn_block in self.res_attn_blocks:
            x, attn_weight = res_attn_block(x, attn_mask=attention_mask)
            if return_weights:
                ret_weights.append(attn_weight)

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        if return_weights:
            return x, torch.stack(ret_weights, dim=1)
        return x, attn_weight.unsqueeze(0)
