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
        visual_num_heads: int = 12,
        visual_num_layers: int = 12,
        textual_num_heads: int = 8,
        textual_num_layers: int = 12,
        *,
        input_image_size: int | tuple[int, int] | tuple[int, int, int] = 224,
        patch_embed_dim: int = 768,
        patch_size: tuple[int, int] = (32, 32),
        patch_stride: Optional[tuple[int, int]] = None,
        patch_dropout_prob: float = 0.0,
        vocab_size: int = 49408,
        vocab_embed_dim: int = 512,
        max_context_length: int = 77,
        pad_token_id: int = 0,
        visual_backbone: Optional[nn.Module] = None,
        textual_backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        if visual_backbone is None:
            visual_backbone = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=visual_num_heads,
                num_layers=visual_num_layers,
                batch_first=True,
                input_image_size=input_image_size,
                patch_embed_dim=patch_embed_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_dropout_prob=patch_dropout_prob,
            )

        if textual_backbone is None:
            textual_backbone = TextTransformer(
                embed_dim=embed_dim,
                num_heads=textual_num_heads,
                num_layers=textual_num_layers,
                vocab_size=vocab_size,
                vocab_embed_dim=vocab_embed_dim,
                max_context_length=max_context_length,
                pad_token_id=pad_token_id,
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
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, text: torch.Tensor, normalize: bool = True):
        feats, *_ = self.textual(text)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
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
