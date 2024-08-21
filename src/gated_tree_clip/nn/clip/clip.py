from typing import Optional

import torch
import torch.nn as nn
from utils.clogging import getColoredLogger

from ..transformer.text_transformer import TextTransformer
from ..transformer.vision_transformer import VisionTransformer

logger = getColoredLogger(__name__)

__all__ = ["CLIP"]


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        visual_backbone: Optional[VisionTransformer] = None,
        textual_backbone: Optional[TextTransformer] = None,
    ):
        super().__init__()
        if visual_backbone is None:
            visual_backbone = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=12,
                num_layers=12,
            )

        if textual_backbone is None:
            textual_backbone = TextTransformer(
                embed_dim=embed_dim,
                num_heads=8,
                num_layers=12,
            )
        self.embed_dim = embed_dim
        self.visual = visual_backbone
        self.textual = textual_backbone
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    @property
    def dtype(self):
        return self.visual.conv.weight.dtype

    def encode_image(self, image: torch.Tensor, normalize: bool = True):
        feats, *_ = self.visual(image)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, text: torch.Tensor, normalize: bool = True):
        feats, *_ = self.textual(text)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def get_features(
        self, image: torch.Tensor, text: torch.Tensor, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image, normalize=normalize)
        text_features = self.encode_text(text, normalize=normalize)
        return image_features, text_features

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        *,
        normalize: bool = True,
        softmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features, text_features = self.get_features(images, tokens, normalize=normalize)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t() + self.logit_bias
        if softmax:
            logits_per_image = logits_per_image.softmax(dim=1)
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
