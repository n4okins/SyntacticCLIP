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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_proj = nn.Linear(self.embed_dim * self.num_heads, self.embed_dim, bias=True)

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

        key = key if key is not None else query
        value = value if value is not None else query

        for v in (query, key, value):
            if v.dim() == 2:
                if self.batch_first:
                    v.unsqueeze_(0)
                else:
                    v.unsqueeze_(1)

        batch_size, seq_len, _ = query.size()

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
        use_fast_path = any(
            (
                not torch.backends.mha.get_fastpath_enabled(),
                query is not key or key is not value,
                self.in_proj_weight is None,
                self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype,
                query.dtype != self.in_proj_weight.dtype,
                self.training,
                self.num_heads % 2 != 0,
                not self.batch_first,
                self.bias_k is not None or self.bias_v is not None,
                self.add_zero_attn,
                not self._qkv_same_embed_dim,
                query.is_nested and (key_padding_mask is not None or attn_mask is not None),
                torch.is_autocast_enabled(),
            )
        )

        if not use_fast_path and self._qkv_same_embed_dim and self.in_proj_bias is not None:
            # if self.in_proj_bias is not None and self.in_proj_weight is not None:
            return torch._native_multi_head_attention(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
                merged_mask,
                need_weights,
                average_attn_weights,
                mask_type,
            )

        assert not (query.is_nested or key.is_nested or value.is_nested), "MultiheadAttention does not support NestedTensor."

        if self.batch_first:
            # query: (batch_size, seq_len, embed_dim)
            # key: (batch_size, seq_len, embed_dim)
            # value: (batch_size, seq_len, embed_dim)
            assert key.dim() == 3, f"key must have 3 dimensions (batch_size, seq_len, embed_dim), got {key.dim()}"
            assert value.dim() == 3, f"value must have 3 dimensions (batch_size, seq_len, embed_dim), got {value.dim()}"
            query = query.transpose(1, 0).contiguous()
            key = key.transpose(1, 0).contiguous()
            value = value.transpose(1, 0).contiguous()
            # query: (seq_len, batch_size, embed_dim)
            # key: (seq_len, batch_size, embed_dim)
            # value: (seq_len, batch_size, embed_dim)

        if attn_gate is None:
            # if attn_gate is None, then use the original multi-head attention
            multi_head_attention_forward_kwargs = dict(
                query=query,
                key=key,
                value=value,
                embed_dim_to_check=self.embed_dim,
                num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight,
                in_proj_bias=self.in_proj_bias,
                bias_k=self.bias_k,
                bias_v=self.bias_v,
                add_zero_attn=self.add_zero_attn,
                dropout_p=self.dropout_p,
                out_proj_weight=self.out_proj.weight,
                out_proj_bias=self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
            if not self._qkv_same_embed_dim:
                multi_head_attention_forward_kwargs.update(
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight,
                    k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight,
                )
            attn_output, attn_weights = F.multi_head_attention_forward(**multi_head_attention_forward_kwargs)
            # attn_output: (seq_len, batch_size, embed_dim)
            # attn_weights: (batch_size, seq_len, seq_len)

        else:
            # attn_gate: (batch_size, seq_len, seq_len)
            # query: (seq_len, batch_size, embed_dim)
            # key: (seq_len, batch_size, embed_dim)
            # value: (seq_len, batch_size, embed_dim)

            # to batch_first
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

            if self._qkv_same_embed_dim:
                W_q, W_k, W_v = self.in_proj_weight.chunk(3, dim=0)
                if self.in_proj_bias is not None:
                    b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
                else:
                    b_q = b_k = b_v = None
            else:
                W_q, W_k, W_v = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight

            query = F.linear(query, W_q, b_q)
            key = F.linear(key, W_k, b_k)
            value = F.linear(value, W_v, b_v)
            if self.add_bias_kv:
                key += self.bias_k
                value += self.bias_v

            query = query.repeat(self.num_heads, 1, 1, 1).view(self.num_heads * batch_size, seq_len, self.embed_dim)
            key = key.repeat(self.num_heads, 1, 1, 1).view(self.num_heads * batch_size, seq_len, self.embed_dim)
            value = value.repeat(self.num_heads, 1, 1, 1).view(self.num_heads * batch_size, seq_len, self.embed_dim)

            attn_head_weights = torch.bmm(query, key.transpose(1, 2)) / (self.embed_dim**0.5)
            attn_head_biases = torch.zeros((seq_len, seq_len), dtype=query.dtype, device=query.device)

            if is_causal:
                assert attn_mask is None, "attn_mask is not None when is_causal is True"
                causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=query.device).triu(diagonal=0)
                attn_head_biases.masked_fill_(causal_mask.logical_not(), float("-inf"))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_head_biases.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_head_biases += attn_mask

            if key_padding_mask is not None:
                attn_head_weights = attn_head_weights.view(batch_size, self.num_heads, seq_len, seq_len)
                attn_head_weights = attn_head_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
                attn_head_weights = attn_head_weights.view(batch_size * self.num_heads, seq_len, seq_len)
                if average_attn_weights:
                    attn_head_weights = F.softmax(attn_head_weights, dim=-1)

            # attn_head_weights: (batch_size * num_heads, seq_len, seq_len)
            # attn_gate: (batch_size * num_heads, seq_len, seq_len)
            if attn_gate is not None:
                attn_head_weights = attn_head_weights * attn_gate
                attn_head_weights /= attn_head_weights.sum(dim=-1, keepdim=True) + attn_weight_div_delta

            attn_output = torch.bmm(attn_head_weights, value)
            attn_output = attn_output.chunk(self.num_heads, dim=0)
            attn_output = torch.cat(attn_output, dim=2)
            # attn_output: (batch_size, seq_len, embed_dim * num_heads)
            assert list(attn_output.size()) == [batch_size, seq_len, self.embed_dim * self.num_heads]
            attn_output = self.out_proj(attn_output)
            attn_weights = None
            if need_weights:
                attn_weights = attn_head_weights.view(batch_size, self.num_heads, seq_len, seq_len)
                if average_attn_weights:
                    attn_weights = attn_weights.mean(dim=1)

        if not self.batch_first:
            attn_output = attn_output.transpose(1, 0)
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
        self.layer_norm_1_kv = CastLayerNorm(normalized_shape=embed_dim) if is_cross_attention else nn.Identity()

        self.attention = MultiheadAttentionWithGate(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)
        self.gate = SyntacticDistanceGate(
            in_channels=embed_dim, lookback_range=3, num_gate_heads=num_heads, batch_first=batch_first
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
        key = self.layer_norm_1_kv(key) if self.is_cross_attention and key is not None else _normed_query
        value = self.layer_norm_1_kv(value) if self.is_cross_attention and value is not None else _normed_query

        attn_gate, distance = self.gate(key)
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

