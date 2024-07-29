import math
from typing import Callable, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gated_tree_clip.utils.clogging import getColoredLogger

from .normalization import LayerNorm
from .scaler import LayerScale

logger = getColoredLogger(__name__)

__all__ = [
    "Attention",
    "ResidualAttentionBlock",
    "AttentionalPooler",
    "CustomResidualAttentionBlock",
]


class Attention(nn.Module):
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L89
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        scaled_cosine: bool = False,
        scale_heads: bool = False,
        logit_scale_max: float = math.log(1.0 / 0.01),
        batch_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert input_dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first
        self.use_fsdpa = hasattr(nn.functional, "scaled_dot_product_attention")

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.input_qkv_weight = nn.Parameter(torch.randn((input_dim * 3, input_dim)) * self.scale)

        if qkv_bias:
            self.input_qkv_bias = nn.Parameter(torch.zeros(input_dim * 3))
        else:
            self.input_qkv_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)

        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None

        self.out_proj = nn.Linear(input_dim, input_dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)  # (B, S, C) -> (S, B, C)

        sequence_length, batch_size, channel_size = x.shape

        query, key, value = F.linear(x, self.input_qkv_weight, self.input_qkv_bias).chunk(3, dim=-1)

        query = query.reshape(sequence_length, batch_size * self.num_heads, -1).transpose(0, 1)
        key = key.reshape(sequence_length, batch_size * self.num_heads, -1).transpose(0, 1)
        value = value.reshape(sequence_length, batch_size * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
            del new_attn_mask

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(query, dim=-1), F.normalize(key, dim=-1).transpose(-1, -2))

            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(batch_size, self.num_heads, sequence_length, sequence_length) * logit_scale
            attn = attn.view(-1, sequence_length, sequence_length)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, value)
        else:
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                query = query * self.scale
                attn = torch.bmm(query, key.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, value)

        if self.head_scale is not None:
            x = x.view(batch_size, self.num_heads, sequence_length, channel_size) * self.head_scale
            x = x.view(-1, sequence_length, channel_size)

        x = x.transpose(0, 1).reshape(sequence_length, batch_size, channel_size)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [("c_fc", nn.Linear(d_model, mlp_width)), ("gelu", act_layer()), ("c_proj", nn.Linear(mlp_width, d_model))]
            )
        )
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("ln", norm_layer(mlp_width) if scale_fc else nn.Identity()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
