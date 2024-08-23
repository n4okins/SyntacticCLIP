from typing import Optional

import torch
import torch.nn as nn
from utils.clogging import getColoredLogger

from ..dropout.patch_dropout import PatchDropout
from ..normalization.cast_layernorm import CastLayerNorm
from ..transformer.transformer import Transformer

logger = getColoredLogger(__name__)


class VisionTransformer(nn.Module):
    """Vision Transformer
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of heads
        num_layers (int): Number of layers
        input_image_size (int | tuple[int, int] | tuple[int, int, int]): Input image size
        patch_embed_dim (int): Patch embedding dimension
        patch_size (tuple[int, int]): Patch size
        patch_stride (Optional[tuple[int, int]]): Patch stride
        patch_dropout_prob (float): Patch dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 12,
        num_layers: int = 12,
        batch_first: bool = True,
        *,
        input_image_size: int | tuple[int, int] | tuple[int, int, int] = 224,
        patch_embed_dim: int = 768,
        patch_size: tuple[int, int] = (32, 32),
        patch_stride: Optional[tuple[int, int]] = None,
        patch_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.transformer = Transformer(
            patch_embed_dim, num_heads, num_layers, batch_first
        )
        self.embed_dim = embed_dim
        self.patch_embed_dim = patch_embed_dim

        # image size adjustment
        if isinstance(input_image_size, int):
            input_image_size = (3, input_image_size, input_image_size)
        elif len(input_image_size) == 2:
            input_image_size = (3, *input_image_size)
        elif len(input_image_size) > 3:
            logger.warnning(
                f"{input_image_size=} is not a valid image size. Using the first 3 elements."
            )
            input_image_size = input_image_size[:3]

        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size

        self.scale = patch_embed_dim ** (-0.5)
        self.input_image_size = input_image_size

        # check if the input image size is divisible by the patch size
        assert (
            input_image_size[1] % patch_size[0] == 0
        ), f"{input_image_size=} {patch_size=} {patch_stride=}"
        assert (
            input_image_size[2] % patch_size[1] == 0
        ), f"{input_image_size=} {patch_size=} {patch_stride=}"

        self.class_embedding = nn.Parameter(self.scale * torch.randn(patch_embed_dim))
        self.positional_grid_size = (
            input_image_size[1] // patch_size[0],
            input_image_size[2] // patch_size[1],
        )
        self.positional_embedding = nn.Parameter(
            self.scale
            * torch.randn(
                self.positional_grid_size[0] * self.positional_grid_size[1] + 1,
                patch_embed_dim,
            )
        )

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=patch_embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            bias=False,
        )

        self.patchdropout_pre = (
            PatchDropout(p=patch_dropout_prob)
            if patch_dropout_prob > 0
            else nn.Identity()
        )
        self.layernorm_pre = CastLayerNorm(normalized_shape=patch_embed_dim)
        self.layernorm_post = CastLayerNorm(normalized_shape=patch_embed_dim)

        self.head_weight = nn.Parameter(
            self.scale * torch.randn(patch_embed_dim, embed_dim)
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
        x, w = self.transformer(
            x, attention_mask=attention_mask, return_weights=return_weights
        )
        x = self.layernorm_post(x)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, self.patch_embed_dim], [batch, num_patches, self.patch_embed_dim]
        # _tokens: unused
        pooled, _tokens = x[:, 0], x[:, 1:]

        # [batch, self.patch_embed_dim] -> [batch, self.embed_dim]
        pooled = pooled @ self.head_weight
        return pooled, w
