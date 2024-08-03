import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clogging import getColoredLogger

from .from_fairseq import FairseqDropout
from .layernorm import CastLayerNorm
from .layerscale import LayerScale

logger = getColoredLogger(__name__)

__all__ = ["ResidualAttentionBlock", "MultiheadAttention"]


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        use_fairseq_dropout: bool = True,
    ):
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        device_and_dtypes = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = (
            FairseqDropout(dropout, module_name=self.__class__.__name__) if use_fairseq_dropout else nn.Dropout(dropout)
        )

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

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

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim), **device_and_dtypes)
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim), **device_and_dtypes)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight, gain=1 / math.sqrt(2))
        else:
            # gain=1 / math.sqrt(2)
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
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        why_not_fast_pathes = []
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_pathes.append("floating-point masks are not supported for fast path.")

        is_batched = query.dim() == 3

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

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_pathes.append("torch.backends.mha.get_fastpath_enabled() was not True")
        elif not is_batched:
            why_not_fast_pathes.append(f"input not batched; expected query.dim() of 3 but got {query.dim()}")
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_pathes.append("non-self attention was used (query, key, and value are not the same Tensor)")
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_pathes.append(
                f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
            )
        elif self.in_proj_weight is None:
            why_not_fast_pathes.append("in_proj_weight was None")
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_pathes.append(
                f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
            )
        elif self.training:
            why_not_fast_pathes.append("training is enabled")
        elif (self.num_heads % 2) != 0:
            why_not_fast_pathes.append("self.num_heads is not even")
        elif not self.batch_first:
            why_not_fast_pathes.append("batch_first was not True")
        elif self.bias_k is not None:
            why_not_fast_pathes.append("self.bias_k was not None")
        elif self.bias_v is not None:
            why_not_fast_pathes.append("self.bias_v was not None")
        elif self.add_zero_attn:
            why_not_fast_pathes.append("add_zero_attn was enabled")
        elif not self._qkv_same_embed_dim:
            why_not_fast_pathes.append("_qkv_same_embed_dim was not True")
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_pathes.append(
                "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
            )
        elif torch.is_autocast_enabled():
            why_not_fast_pathes.append("autocast is enabled")
        if not why_not_fast_pathes:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_pathes.append("some Tensor argument has_torch_function")
            elif not torch.jit.is_scripting() and any(
                type(x) is torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
                for x in torch.utils._python_dispatch._get_current_dispatch_mode_stack()
            ):
                # _is_make_fx_tracing()
                why_not_fast_pathes.append("we are running make_fx tracing")
            elif not all(
                x.device.type in ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
                if x is not None
                else True
                for x in tensor_args
            ):
                # _check_arg_device()
                why_not_fast_pathes.append(
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad if x is not None else False for x in tensor_args):
                why_not_fast_pathes.append(
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_fast_pathes:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
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

            else:
                logger.warning(f"The fast path was not hit because {'\n - '.join(why_not_fast_pathes)}")

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {'\n - '.join(why_not_fast_pathes)}"
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout.p,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout.p,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0).contiguous()

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

        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)

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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_mask = attention_mask.to(query.dtype) if attention_mask is not None else None

        _normed_query = self.layer_norm_1(query)
        key = self.layer_norm_1_kv(key) if self.is_cross_attention and key is not None else _normed_query
        value = self.layer_norm_1_kv(value) if self.is_cross_attention and value is not None else _normed_query

        attention_out, _ = self.attention(
            _normed_query,
            key,
            value,
            need_weights=False,
            attn_mask=attention_mask,
        )

        x = query + self.layer_scale_1(attention_out)
        x = x + self.layer_scale_2(self.res_mlp(self.layer_norm_2(x)))
        return x
