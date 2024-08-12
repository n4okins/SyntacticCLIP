from typing import Optional, override

import torch
import torch.nn as nn
from utils.clogging import getColoredLogger

from .gated_attention import ResidualAttentionWithSyntacticDistanceBlock
from .syntactic_distance_gate import SyntacticDistanceGate
from .transformer import TextTransformer, VisionTransformer

logger = getColoredLogger(__name__)

__all__ = [
    "TransformerWithSyntacticDistance",
    "VisionTransformerWithSyntacticDistance",
    "TextTransformerWithSyntacticDistance",
]


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
        self.induction_head = SyntacticDistanceGate(
            embed_dim, num_heads, batch_first=batch_first
        )
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
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
        return_weight: bool = False,
        return_distance: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
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

        ret = [x]
        if return_weight:
            ret.append(attn_weight)
        if return_distance:
            ret.append(distance)
        return tuple(ret) if len(ret) > 1 else ret[0]


class VisionTransformerWithSyntacticDistance(VisionTransformer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_layers, batch_first)
        self.transformer = TransformerWithSyntacticDistance(
            self.patch_embed_dim, num_heads, num_layers, batch_first
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
        return_weight: bool = False,
        return_distance: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        """
        batch_size, channels, height, width = x.shape

        self.positional_embedding.to(x.dtype)

        # [batch, channels, height, width] -> [batch, self.patch_embed_dim, *self.positional_grid_size]
        x = self.conv(x)

        # num_patches := self.positional_grid_size[0] * self.positional_grid_size[1]
        # [batch, self.patch_embed_dim, *self.positional_grid_size] -> [batch, num_patches, self.patch_embed_dim]
        x = x.reshape(batch_size, self.patch_embed_dim, -1).permute(0, 2, 1)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = torch.cat(
            [self.class_embedding.view(1, 1, -1).expand(batch_size, -1, -1), x], dim=1
        )
        x = x + self.positional_embedding

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = self.patchdropout_pre(x)
        x = self.layernorm_pre(x)
        x, *w = self.transformer(
            x,
            attention_mask=attention_mask,
            attention_gate=attention_gate,
            return_weight=return_weight,
            return_distance=return_distance,
        )
        print(x)
        print(w)
        x = self.layernorm_post(x)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, self.patch_embed_dim], [batch, num_patches, self.patch_embed_dim]
        # _tokens: unused
        pooled, _tokens = x[:, 0], x[:, 1:]

        # [batch, self.patch_embed_dim] -> [batch, self.embed_dim]
        pooled = pooled @ self.head_weight

        if return_weight:
            return pooled, w
        return pooled


class TextTransformerWithSyntacticDistance(TextTransformer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_layers, batch_first)
        self.transformer = TransformerWithSyntacticDistance(
            embed_dim, num_heads, num_layers, batch_first
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
        return_weight: bool = False,
        return_distance: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, sequence_length]
        """
        x_ = x
        batch_size, sequence_length = x.shape

        x = self.embedding(x)
        x = x + self.positional_embedding[:sequence_length]
        x, *w = self.transformer(
            x,
            attention_mask=attention_mask or self.attention_mask,
            attention_gate=attention_gate,
            return_weight=return_weight,
            return_distance=return_distance,
        )
        x = self.layernorm_post(x)

        # _tokens: unused
        pooled, _tokens = x[torch.arange(batch_size), x_.argmax(dim=-1)], x
        pooled = pooled @ self.head_weight

        if return_weight:
            return pooled, w
        return pooled
