from typing import Optional

import torch
import torch.nn as nn

from ..activation import QuickGELU
from ..misc import LayerScale, SyntacticDistanceGate
from ..normalization import CastLayerNorm
from .gated_multihead_attention import GatedMultiheadAttention


class ResidualGatedAttentionBlock(nn.Module):
    """residual attention with syntactic distance block
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
        num_heads: int = 8,
        batch_first: bool = True,
        *,
        num_lookback_range: int = 3,
        tau: float = 1.0,
        gate_dropout_p: float = 0.0,
        res_mlp: Optional[nn.Module] = None,
        res_mlp_dim: Optional[int] = None,
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

        self.layernorm_1 = CastLayerNorm(normalized_shape=embed_dim)
        self.layernorm_1_kv = (
            CastLayerNorm(normalized_shape=embed_dim)
            if is_cross_attention
            else nn.Identity()
        )

        self.attention = GatedMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first
        )
        self.gate = SyntacticDistanceGate(
            embed_dim=embed_dim,
            num_lookback_range=num_lookback_range,
            num_gate_heads=num_heads,
            tau=tau,
            dropout_p=gate_dropout_p,
        )

        self.layerscale_1 = (
            LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layer_scale_ratio)
            if init_layer_scale_ratio
            else nn.Identity()
        )
        self.layernorm_2 = CastLayerNorm(normalized_shape=embed_dim)

        self.res_mlp = (
            res_mlp
            if res_mlp
            else nn.Sequential(
                nn.Linear(in_features=embed_dim, out_features=res_mlp_dim),
                QuickGELU(),
                nn.Linear(in_features=res_mlp_dim, out_features=embed_dim),
            )
        )

        self.layerscale_2 = (
            LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layer_scale_ratio)
            if init_layer_scale_ratio
            else nn.Identity()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        attn_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query: (batch_size, seq_len, embed_dim)
        # key: (batch_size, seq_len, embed_dim)
        # value: (batch_size, seq_len, embed_dim)
        # attn_mask: (batch_size, seq_len, seq_len)
        # attn_gate: (batch_size, seq_len, seq_len)

        attn_mask = attn_mask.to(query.dtype) if attn_mask is not None else None
        _normed_query = self.layernorm_1(query)
        key = (
            self.layernorm_1_kv(key)
            if self.is_cross_attention and key is not None
            else _normed_query
        )
        value = (
            self.layernorm_1_kv(value)
            if self.is_cross_attention and value is not None
            else _normed_query
        )

        if attn_gate is None:
            attn_gate, distance = self.gate(key)
        else:
            distance = None

        attn_out, attn_weight = self.attention(
            _normed_query,
            key,
            value,
            need_weights=True,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
        )
        x = query + self.layerscale_1(attn_out)
        x = x + self.layerscale_2(self.res_mlp(self.layernorm_2(x)))
        return x, attn_weight, distance
