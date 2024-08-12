from typing import Any, Optional

import torch
import torch.nn as nn
from utils.clogging import getColoredLogger

from .gated_transformer import (
    TextTransformerWithSyntacticDistance,
    VisionTransformerWithSyntacticDistance,
)
from .transformer import TextTransformer, VisionTransformer

logger = getColoredLogger(__name__)

__all__ = ["CLIPBase", "CLIPEncoder", "SyntacticCLIP"]


class CLIPEncoder(nn.Module):
    """CLIP encoder wrapper
    Args:
        embed_dim (int): Embedding dimension (output dim of the backbone)
        backbone (nn.Module): Backbone
    """

    def __init__(
        self, embed_dim: int = 512, backbone: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = backbone or nn.Identity()

    def forward(self, x: Any, **kwargs: Any) -> Any:
        return self.backbone(x, **kwargs)


class CLIPBase(nn.Module):
    """CLIP base model
    Args:
        embed_dim (int): Embedding dimension
        init_logit_scale (float | torch.Tensor): Initial logit scale
        init_logit_bias (Optional[float]): Initial logit bias
    """

    def __init__(
        self,
        embed_dim: int = 512,
        *,
        visual_backbone: Optional[nn.Module] = None,
        textual_backbone: Optional[nn.Module] = None,
        init_logit_scale: float | torch.Tensor = torch.log(torch.tensor(1.0 / 0.07)),
        init_logit_bias: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias: torch.Tensor
        if init_logit_bias is None:
            self.register_buffer("logit_bias", torch.zeros([]), persistent=False)
        else:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)

        self.visual = CLIPEncoder(
            embed_dim, visual_backbone or VisionTransformer(embed_dim)
        )
        self.textual = CLIPEncoder(
            embed_dim, textual_backbone or TextTransformer(embed_dim)
        )

    def encode_image(
        self,
        images: torch.Tensor,
        normalize: bool = False,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Encode image
        Args:
            images (torch.Tensor): [B, C, H, W]
            normalize (bool): Normalize the output features
            return_weights (bool): Return attention weights
        Returns:
            torch.Tensor: [B, D], D=visual.embed_dim
        """
        feats, weights = self.visual(images, return_weights=return_weights)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats, weights

    def encode_text(
        self,
        tokens: torch.Tensor,
        normalize: bool = False,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Encode text
        Args:
            tokens (torch.Tensor): [B, N]
            normalize (bool): Normalize the output features
            return_weights (bool): Return attention weights
        Returns:
            torch.Tensor: [B, D], D=textual.embed_dim
        """
        feats, weights = self.textual(tokens, return_weights=return_weights)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats, weights

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        *,
        normalize: bool = False,
        softmax: bool = True,
        return_weights: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        image_output, image_weights = self.encode_image(
            images, normalize=normalize, return_weights=return_weights
        )
        (
            text_output,
            text_weights,
        ) = self.encode_text(tokens, normalize=normalize, return_weights=return_weights)
        probs = (
            self.logit_scale.exp() * (image_output @ text_output.T) + self.logit_bias
        )
        if softmax:
            probs = probs.softmax(dim=1)
        return probs, (image_output, text_output), (image_weights, text_weights)


class SyntacticCLIP(CLIPBase):
    def __init__(
        self,
        embed_dim: int = 512,
        **kwargs: Any,
    ) -> None:
        visual_backbone = VisionTransformerWithSyntacticDistance(embed_dim)
        textual_backbone = TextTransformerWithSyntacticDistance(embed_dim)
        super().__init__(
            embed_dim,
            visual_backbone=visual_backbone,
            textual_backbone=textual_backbone,
            **kwargs,
        )
