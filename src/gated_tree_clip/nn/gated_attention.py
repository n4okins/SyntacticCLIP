from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clogging import getColoredLogger

from .attention import MultiheadAttention
from .layernorm import CastLayerNorm
from .layerscale import LayerScale
from .syntactic_distance_gate import SyntacticDistanceGate

logger = getColoredLogger(__name__)

__all__ = ["ResidualAttentionWithSyntacticDistanceBlock", "MultiheadAttentionWithGate"]


class MultiheadAttentionWithGate(MultiheadAttention):
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        attn_gate: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        attn_weight_div_delta: float = 1e-12,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input shape: (seq_len, batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
        """
        if attn_gate is None:
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        key = key if key is not None else query
        value = value if value is not None else query

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

        assert not (
            query.is_nested or key.is_nested or value.is_nested
        ), f"{self.__class__.__name__} does not support NestedTensor."

        if not self.batch_first:
            # query: (seq_len, batch_size, embed_dim)
            # key: (seq_len, batch_size, embed_dim)
            # value: (seq_len, batch_size, embed_dim)
            assert (
                key.dim() == 3
            ), f"key must have 3 dimensions (seq_len, batch_size, embed_dim), got {key.dim()=}"
            assert (
                value.dim() == 3
            ), f"value must have 3 dimensions (seq_len, batch_size, embed_dim), got {value.dim()=}"
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

        # query: (batch_size, seq_len, embed_dim)
        # key: (batch_size, seq_len, embed_dim)
        # value: (batch_size, seq_len, embed_dim)
        if self._qkv_same_embed_dim:
            W_q, W_k, W_v = self.in_proj_weight.chunk(3, dim=0)
            if self.in_proj_bias is not None:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None
        else:
            W_q, W_k, W_v = (
                self.q_proj_weight,
                self.k_proj_weight,
                self.v_proj_weight,
            )

        query = F.linear(query, W_q, b_q)
        key = F.linear(key, W_k, b_k)
        value = F.linear(value, W_v, b_v)

        if self.add_bias_kv:
            key += self.bias_k
            value += self.bias_v

        batch_size, seq_len, _ = query.size()
        attn_head_biases = torch.zeros(
            (seq_len, seq_len), dtype=query.dtype, device=query.device
        )

        if is_causal:
            assert attn_mask is None, "attn_mask is not None when is_causal is True"
            causal_mask = torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=query.device
            ).triu(diagonal=0)
            attn_head_biases.masked_fill_(causal_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_head_biases.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_head_biases += attn_mask

        query = query.contiguous().view(
            batch_size * self.num_heads, seq_len, self.head_dim
        )
        key = key.contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        value = value.contiguous().view(
            batch_size * self.num_heads, seq_len, self.head_dim
        )
        attn_head_weights = torch.bmm(query, key.transpose(1, 2))
        # attn_head_weights: (batch_size * num_heads, seq_len, seq_len)

        assert (
            tuple(attn_head_weights.size())
            == (
                batch_size * self.num_heads,
                seq_len,
                seq_len,
            )
        ), f"{attn_head_weights.size()=}, expected {(batch_size * self.num_heads, seq_len, seq_len)}"

        if key_padding_mask is not None:
            attn_head_weights = attn_head_weights.view(
                batch_size, self.num_heads, seq_len, seq_len
            )
            attn_head_weights = attn_head_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_head_weights = attn_head_weights.view(
                batch_size * self.num_heads, seq_len, seq_len
            )

            # attn_head_weights: (batch_size * num_heads, seq_len, seq_len)
            # attn_gate: (batch_size * num_heads, seq_len, seq_len)

        num_gate_heads = attn_gate.size(0) // batch_size
        assert (
            tuple(attn_gate.size())
            == (
                batch_size * num_gate_heads,
                seq_len,
                seq_len,
            )
        ), f"{attn_gate.size()=}, expected {(batch_size * num_gate_heads, seq_len, seq_len)}"
        # attn_gate: (batch_size * num_gate_heads, seq_len, seq_len)

        if self.num_heads - num_gate_heads > 0:
            attn_gate = (
                torch.cat(
                    [
                        attn_gate.view(batch_size, num_gate_heads, seq_len, seq_len),
                        attn_gate.new_ones(
                            (
                                batch_size,
                                self.num_heads - num_gate_heads,
                                seq_len,
                                seq_len,
                            )
                        ),
                    ],
                    dim=1,
                )
                .view(batch_size * self.num_heads, seq_len, seq_len)
                .contiguous()
            )

        assert tuple(attn_gate.size()) == (
            batch_size * self.num_heads,
            seq_len,
            seq_len,
        ), f"{attn_gate.size()=}, expected {(batch_size, seq_len, seq_len)}"

        # attn_gate: (batch_size * self.num_heads, seq_len, seq_len)
        # attn_head_weights: (batch_size * self.num_heads, seq_len, seq_len)
        attn_head_weights = attn_head_weights * attn_gate
        attn_head_weights /= (
            attn_head_weights.sum(dim=-1, keepdim=True) + attn_weight_div_delta
        )
        attn_head_weights = F.softmax(attn_head_weights, dim=-1)

        attn_output = torch.bmm(attn_head_weights, value)
        # attn_output: (batch_size * num_heads, seq_len, head_dim)

        assert (
            tuple(attn_output.size())
            == (
                batch_size * self.num_heads,
                seq_len,
                self.head_dim,
            )
        ), f"{attn_output.size()=}, expected {(batch_size * self.num_heads, seq_len, self.head_dim)}"

        attn_output = (
            attn_output.transpose(1, 0)
            .contiguous()
            .view(seq_len, batch_size, self.embed_dim)
        ).contiguous()
        attn_output = self.out_proj(attn_output)
        # attn_output: (seq_len, batch_size, embed_dim)
        # attn_head_weights: (batch_size * num_heads, seq_len, seq_len)

        attn_weights = None
        if need_weights:
            attn_weights = attn_head_weights.view(
                batch_size, self.num_heads, seq_len, seq_len
            )
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1).contiguous()
            else:
                attn_weights = attn_weights.contiguous()

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)
        attn_output = attn_output.contiguous()

        # attn_output: (batch_size, seq_len, embed_dim) or (seq_len, batch_size, embed_dim)
        # attn_weights: (batch_size, seq_len, seq_len)
        return attn_output, attn_weights


class ResidualAttentionWithSyntacticDistanceBlock(nn.Module):
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

        self.layer_norm_1 = CastLayerNorm(normalized_shape=embed_dim)
        self.layer_norm_1_kv = (
            CastLayerNorm(normalized_shape=embed_dim)
            if is_cross_attention
            else nn.Identity()
        )

        self.attention = MultiheadAttentionWithGate(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first
        )
        self.gate = SyntacticDistanceGate(
            in_embed_dim=embed_dim,
            num_lookback_range=num_lookback_range,
            num_gate_heads=num_heads,
            batch_first=batch_first,
            tau=tau,
            dropout_p=gate_dropout_p,
        )

        self.layer_scale_1 = (
            LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layer_scale_ratio)
            if init_layer_scale_ratio
            else nn.Identity()
        )
        self.layer_norm_2 = CastLayerNorm(normalized_shape=embed_dim)

        self.res_mlp = (
            res_mlp
            if res_mlp
            else nn.Sequential(
                nn.Linear(in_features=embed_dim, out_features=res_mlp_dim),
                nn.GELU(),
                nn.Linear(in_features=res_mlp_dim, out_features=embed_dim),
            )
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

        _normed_query = self.layer_norm_1(query)
        key = (
            self.layer_norm_1_kv(key)
            if self.is_cross_attention and key is not None
            else _normed_query
        )
        value = (
            self.layer_norm_1_kv(value)
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
        x = query + self.layer_scale_1(attn_out)
        x = x + self.layer_scale_2(self.res_mlp(self.layer_norm_2(x)))
        return x, attn_weight, distance
