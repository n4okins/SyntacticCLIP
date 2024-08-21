from typing import Any

from utils.clogging import getColoredLogger

from ..transformer.gated_text_transformer import SyntacticTextTransformer
from ..transformer.gated_vision_transformer import SyntacticVisionTransformer
from .clip import CLIP

logger = getColoredLogger(__name__)


class SyntacticCLIP(CLIP):
    def __init__(
        self,
        embed_dim: int = 512,
        **kwargs: Any,
    ) -> None:
        visual_backbone = SyntacticVisionTransformer(embed_dim)
        textual_backbone = SyntacticTextTransformer(embed_dim)
        super().__init__(
            embed_dim,
            visual_backbone=visual_backbone,
            textual_backbone=textual_backbone,
            **kwargs,
        )
