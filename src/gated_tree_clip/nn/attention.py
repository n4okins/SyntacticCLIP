from typing import Optional

import torch
import torch.nn as nn

from ..utils.clogging import getColoredLogger
from .layernorm import CastLayerNorm
from .layerscale import LayerScale

logger = getColoredLogger(__name__)

__all__ = ["ResidualAttentionBlock"]


class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block
    Args:
        embed_dim (int): Embedding dimension
        res_mlp_dim (Optional[int]): Residual MLP dimension
        res_mlp (Optional[nn.Module]): Residual MLP
        num_heads (int): Number of heads
        batch_first (bool): Batch first
        is_cross_attention (bool): Cross attention
        init_layer_scale_ratio (Optional[float]): Initial layer scale ratio
    """

    def __init__(
        self,
        embed_dim: int = 512,
        *,
        res_mlp_dim: Optional[int] = None,
        res_mlp: Optional[nn.Module] = None,
        num_heads: int = 8,
        batch_first: bool = True,
        is_cross_attention: bool = False,
        init_layer_scale_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        res_mlp_dim = res_mlp_dim or embed_dim * 4
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.is_cross_attention = is_cross_attention
        self.init_layer_scale_ratio = init_layer_scale_ratio

        self.layer_norm_1 = CastLayerNorm(normalized_shape=embed_dim)
        if is_cross_attention:
            self.layer_norm_1_kv = CastLayerNorm(normalized_shape=embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)

        self.layer_scale_1 = (
            LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layer_scale_ratio)
            if init_layer_scale_ratio
            else nn.Identity()
        )
        self.layer_norm_2 = CastLayerNorm(normalized_shape=embed_dim)

        self.res_mlp = res_mlp or nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=res_mlp_dim),
            nn.GELU(),
            nn.Linear(in_features=res_mlp_dim, out_features=embed_dim),
        )

        self.layer_scale_2 = (
            LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layer_scale_ratio)
            if init_layer_scale_ratio
            else nn.Identity()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key = self.layer_norm_1_kv(key) if hasattr(self, "layer_norm_1_kv") and key is not None else None
        value = self.layer_norm_1_kv(value) if hasattr(self, "layer_norm_1_kv") and value is not None else None
        attention_mask = attention_mask.to(query.dtype) if attention_mask is not None else None
        _normed_query = self.layer_norm_1(query)
        attention_out = self.attention(
            _normed_query,
            key if key is not None else _normed_query,
            value if value is not None else _normed_query,
            need_weights=False,
            attn_mask=attention_mask,
        )[0]
        x = query + self.layer_scale_1(attention_out)
        x = x + self.layer_scale_2(self.res_mlp(self.layer_norm_2(x)))
        return x
