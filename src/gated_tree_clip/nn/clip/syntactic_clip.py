from typing import Any, Optional

import torch.nn as nn
from utils.clogging import getColoredLogger

from ..transformer.gated_text_transformer import SyntacticTextTransformer
from ..transformer.gated_vision_transformer import SyntacticVisionTransformer
from .clip import CLIP

logger = getColoredLogger(__name__)


class SyntacticCLIP(CLIP):
    def __init__(
        self,
        embed_dim: int = 512,
        visual_num_heads: int = 12,
        visual_num_layers: int = 12,
        textual_num_heads: int = 8,
        textual_num_layers: int = 12,
        *,
        visual_backbone: Optional[nn.Module] = None,
        textual_backbone: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if visual_backbone is None:
            visual_backbone = SyntacticVisionTransformer(
                embed_dim=embed_dim,
                num_heads=visual_num_heads,
                num_layers=visual_num_layers,
            )
        if textual_backbone is None:
            textual_backbone = SyntacticTextTransformer(
                embed_dim=embed_dim,
                num_heads=textual_num_heads,
                num_layers=textual_num_layers,
            )
        super().__init__(
            embed_dim,
            visual_backbone=visual_backbone,
            textual_backbone=textual_backbone,
            **kwargs,
        )
