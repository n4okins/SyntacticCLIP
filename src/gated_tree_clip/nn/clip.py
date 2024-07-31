from typing import Any, Optional

import torch
import torch.nn as nn

from ..utils.clogging import getColoredLogger
from .transformer import TextTransformer, VisionTransformer

logger = getColoredLogger(__name__)

__all__ = ["CLIPBase", "CLIPEncoder", "GatedTreeCLIP"]


class CLIPEncoder(nn.Module):
    """CLIP encoder wrapper
    Args:
        embed_dim (int): Embedding dimension (output dim of the backbone)
        backbone (nn.Module): Backbone
    """

    def __init__(self, embed_dim: int = 512, backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = backbone or nn.Identity()

    def forward(self, x: Any) -> torch.Tensor:
        return self.backbone(x)


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

        self.visual = CLIPEncoder(embed_dim, visual_backbone or VisionTransformer(embed_dim))
        self.textual = CLIPEncoder(embed_dim, textual_backbone or TextTransformer(embed_dim))

    def get_logits(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        normalize: bool = True,
        softmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get logits for the image-text pair
        Args:
            image (torch.Tensor): [B, C, H, W]
            tokens (torch.Tensor): [B, N]
            normalize (bool): Normalize the output logits
            softmax (bool): Apply softmax to the output
        Returns:
            logits, logits.T (tuple[torch.Tensor, torch.Tensor]): ([1, B], [B, 1])
        """
        image_features = self.encode_image(image, normalize=normalize)
        text_features = self.encode_text(tokens, normalize=normalize)

        logits = self.logit_scale.exp() * (image_features @ text_features.T) + self.logit_bias
        if softmax:
            logits = logits.softmax(dim=-1)
        return logits, logits.T

    def encode_image(self, images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Encode image
        Args:
            images (torch.Tensor): [B, C, H, W]
            normalize (bool): Normalize the output features
        Returns:
            torch.Tensor: [B, D], D=visual.embed_dim
        """
        feats = self.visual(images)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, tokens: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Encode text
        Args:
            tokens (torch.Tensor): [B, N]
            normalize (bool): Normalize the output features
        Returns:
            torch.Tensor: [B, D], D=textual.embed_dim
        """
        feats = self.textual(tokens)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def forward(
        self, images: torch.Tensor, tokens: torch.Tensor, normalize: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.encode_image(images, normalize),
            self.encode_text(tokens, normalize),
            self.logit_scale.exp(),
        )


class GatedTreeCLIP(CLIPBase): ...
