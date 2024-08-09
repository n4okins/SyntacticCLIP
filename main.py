# %%
import math
from typing import Any, Callable, Optional, override

import gated_tree_clip.nn as gtcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)

# %%


class MultiheadAttentionWithGate(nn.Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
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
        self.dropout = nn.Dropout(dropout)

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
        self.scaling = self.head_dim ** -0.5
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
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
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
        attn_gate: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
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
            dropout_p=self.dropout.p,
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
        attn_output, attn_output_weights = F.multi_head_attention_forward(**multi_head_attention_forward_kwargs)
        # attn_output: (seq_len, batch_size, embed_dim)
        # attn_output_weights: (batch_size, seq_len, seq_len)

        if attn_gate is not None:
            # attn_gate: (batch_size, seq_len, seq_len)
            assert attn_gate.size() == attn_output_weights.size(), f"attn_gate and attn_output_weights must have the same size, got {attn_gate.size()=} and {attn_output_weights.size()=}"
            attn_output_weights *= attn_gate
            attn_output_weights /= attn_output_weights.sum(dim=-1, keepdim=True) + 1e-12

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        attn_output = self.dropout(attn_output)

        return attn_output, attn_output_weights

    def merge_masks(
        self, attention_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], query: torch.Tensor
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


class SyntacticDistanceGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        *,
        tau: float = 1.0,
        dropout: float = 0.0,
        batch_first: bool = True,
        distance_activation_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.lookback_range = kernel_size
        self.batch_first = batch_first
        self.tau = tau
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, 1, kernel_size, padding=kernel_size),
        )
        self.distance_activation_fn = distance_activation_fn or nn.Tanh()

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, embed_dim) or (seq_len, batch_size, embed_dim)
        # gate: (batch_size, seq_len, seq_len)
        # distance: (batch_size, seq_len, 1)
        if x.dim() == 2:
            if self.batch_first:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)

        if self.batch_first:
            x = x.transpose(1, 2)
        else:
            x = x.permute(1, 2, 0)

        batch_size, embed_dim, seq_len = x.size()

        # distance: Syntactic Distance [d_i, ...]: i番目の単語の構文距離 (構文高？)
        # distance := distance  (batch_size, seq_len, 1)
        # distance[i] = \tanh(W_D [k_{i-M}, k_{i-M+1}, ..., K_{i}]^{\top} + b_D)
        # conv_input: (batch_size, embed_dim, seq_len)
        distance = self.conv(x)
        # conv_output: (batch_size, distance_dim, seq_len + lookback_range)
        distance = distance[:, :, 1 : -self.lookback_range]
        distance = self.distance_activation_fn(distance)
        distance = distance.transpose(2, 1).contiguous()

        alpha = (F.hardtanh((distance - distance.transpose(2, 1)) * self.tau) + 1) / 2
        lower_tri = alpha.tril(diagonal=-1) + torch.ones_like(alpha).triu(diagonal=0)
        upper_tri = torch.ones_like(alpha).tril(diagonal=0) + alpha.triu(diagonal=1)
        gate = lower_tri * upper_tri
        # gate := gate  (batch_size, seq_len, seq_len), 0 <= gate <= 1
        # distance := distance  (batch_size, seq_len, 1), -1 <= distance <= 1
        return gate, distance


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

        self.layer_norm_1 = gtcnn.CastLayerNorm(normalized_shape=embed_dim)
        self.layer_norm_1_kv = gtcnn.CastLayerNorm(normalized_shape=embed_dim) if is_cross_attention else nn.Identity()

        self.attention = MultiheadAttentionWithGate(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)
        self.gate = SyntacticDistanceGate(in_channels=embed_dim, kernel_size=3, batch_first=batch_first)

        self.layer_scale_1 = (
            gtcnn.LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layer_scale_ratio)
            if init_layer_scale_ratio
            else nn.Identity()
        )
        self.layer_norm_2 = gtcnn.CastLayerNorm(normalized_shape=embed_dim)

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
            gtcnn.LayerScale(embed_dim=embed_dim, init_scale_ratio=init_layer_scale_ratio)
            if init_layer_scale_ratio
            else nn.Identity()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
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

        attention_gate, distance = self.gate(key)
        attention_out, _ = self.attention(
            _normed_query,
            key,
            value,
            need_weights=False,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
        )

        x = query + self.layer_scale_1(attention_out)
        x = x + self.layer_scale_2(self.res_mlp(self.layer_norm_2(x)))
        # x: (batch_size, seq_len, embed_dim)
        # distance: (batch_size, seq_len, 1)
        return x, distance


x = torch.randn(2, 9, 4)
block = ResidualAttentionWithSyntacticDistanceBlock(embed_dim=4, num_heads=2, batch_first=True)

y, d = block(x, attn_gate=torch.rand(2, 9, 9))
print(y.size(), d.size())
# batch_size = 6
# seq_len = 9
# embed_dim = 4
# kernel_size = 3
# syntactic_distance_gate = SyntacticDistanceGate(embed_dim, kernel_size, batch_first=True)
# x = torch.stack([torch.ones(batch_size, seq_len, embed_dim, requires_grad=True).triu(k) for k in range(embed_dim)]).sum(dim=0)
# gate, distance = syntactic_distance_gate(x)
# print(distance.size(), gate.size())
# %%


# SeqL = 9
# KernelSize = 3

# attn:
# 0: [0.1, 0.2, 0.3, 0.4, 0.5]
# 1: [0.0, 0.1, 0.2, 0.3, 0.4]
# 2: [0.0, 0.0, 0.1, 0.2, 0.3]
# 3: [0.0, 0.0, 0.0, 0.1, 0.2]
# 4: [0.0, 0.0, 0.0, 0.0, 0.1]

# gate:
# gate_{0,0} = 0
# gate_{0,1} = cumprod_{j=1 to 0} attn_{0,j} = 0
# gate_{0,2} = cumprod_{j=2 to 1} attn_{0,j} = 0

# *** Conv1D ***
# pl0 pl1 pl2 0 1 2 3 4 5 6 7 8 pr0 pr1 pr2


# Output:
# 0: W[pl0 pl1 pl2] + B
# 1: W[pl1 pl2 0] + B = d0
# 2: W[pl2 0 1] + B = d1
# 3: W[0 1 2] + B = d2
# 4: W[1 2 3] + B = d3
# 5: W[2 3 4] + B = d4
# 6: W[3 4 5] + B = d5
# 7: W[4 5 6] + B = d6
# 8: W[5 6 7] + B = d7
# 9: W[6 7 8] + B = d8
# 10: W[7 8 pr0] + B
# 11: W[8 pr0 pr1] + B
# 12: W[pr0 pr1 pr2] + B
