import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clogging import getColoredLogger

from .layernorm import CastLayerNorm
from .layerscale import LayerScale

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
__all__ = ["ResidualAttentionBlock", "MultiheadAttention"]


class MultiheadAttention(nn.Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout_p: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = False,
        device: torch.device | str = None,
        dtype: torch.dtype = None,
    ):
        assert embed_dim > 0, f"embed_dim must be greater than 0, got {embed_dim}"
        assert num_heads > 0, f"num_heads must be greater than 0, got {num_heads}"
        assert embed_dim % num_heads == 0, f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"

        device_and_dtypes = {"device": device, "dtype": dtype}

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.head_dim = embed_dim // num_heads

        self.in_proj_weight: Optional[nn.Parameter]

        self.q_proj_weight: Optional[nn.Parameter]
        self.k_proj_weight: Optional[nn.Parameter]
        self.v_proj_weight: Optional[nn.Parameter]

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **device_and_dtypes))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **device_and_dtypes))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **device_and_dtypes))
            self.register_buffer("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **device_and_dtypes))
            self.register_buffer("p_proj_weight", None)
            self.register_buffer("k_proj_weight", None)
            self.register_buffer("v_proj_weight", None)

        self.in_proj_bias: Optional[nn.Parameter]

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **device_and_dtypes))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **device_and_dtypes)

        self.add_bias_kv = add_bias_kv
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim), **device_and_dtypes)
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim), **device_and_dtypes)
        else:
            self.register_buffer("bias_k", None)
            self.register_buffer("bias_v", None)

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.add_bias_kv:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input shape: (seq_len, batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
        """

        is_batched_query = query.dim() == 3
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
        use_fast_path = any(
            (
                not torch.backends.mha.get_fastpath_enabled(),
                not is_batched_query,
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

        if self.batch_first and is_batched_query:
            assert key.dim() == 3, f"key must have 3 dimensions (batch_size, seq_len, embed_dim), got {key.dim()}"
            assert value.dim() == 3, f"value must have 3 dimensions (batch_size, seq_len, embed_dim), got {value.dim()}"
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

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
            need_weights=need_weights,
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

        attn_output, attn_output_weights = F.multi_head_attention_forward(**multi_head_attention_forward_kwargs)

        if self.batch_first and is_batched_query:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_output_weights

    def incomplete_scaled_dot_product_self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        # multi-head scaled dot product self attention
        # ただし、query, keyのみ入力として、最後のattn_weight @ valueは計算せずにattn_weightのみ返す
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / (query.size(-1) ** 0.5) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight

    def merge_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        query: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[int]]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        merged_mask: Optional[torch.Tensor] = None
        # mask_type = 1: key_padding_mask, 2: attn_mask, 3: key_padding_mask + attn_mask
        mask_type: Optional[int] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attention_mask is not None:
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            if attention_mask.dim() == 3:
                attention_mask_expanded = attention_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attention_mask_expanded = attention_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attention_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attention_mask_expanded + key_padding_mask_expanded

        return merged_mask, mask_type


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

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)

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
    ) -> torch.Tensor:
        attn_mask = attn_mask.to(query.dtype) if attn_mask is not None else None

        _normed_query = self.layer_norm_1(query)
        key = self.layer_norm_1_kv(key) if self.is_cross_attention and key is not None else _normed_query
        value = self.layer_norm_1_kv(value) if self.is_cross_attention and value is not None else _normed_query

        attn_out, attn_weight = self.attention(
            _normed_query,
            key,
            value,
            need_weights=True,
            attn_mask=attn_mask,
        )

        x = query + self.layer_scale_1(attn_out)
        x = x + self.layer_scale_2(self.res_mlp(self.layer_norm_2(x)))
        return x, attn_weight
