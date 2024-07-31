from typing import Optional

from ..utils.clogging import getColoredLogger
from .transformer import TextTransformer, Transformer, VisionTransformer

logger = getColoredLogger(__name__)

__all__ = ["GatedTreeTransformer", "GatedTreeVisionTransformer", "GatedTreeTextTransformer"]

# TODO: Implement GatedTreeTransformer, GatedTreeVisionTransformer, and GatedTreeTextTransformer


class GatedTreeTransformer(Transformer): ...


class GatedTreeVisionTransformer(VisionTransformer, GatedTreeTransformer): ...


class GatedTreeTextTransformer(TextTransformer, GatedTreeTransformer): ...
