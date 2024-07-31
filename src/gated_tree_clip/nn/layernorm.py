import torch
import torch.nn.functional as F
from torch import nn

from ..utils.clogging import getColoredLogger

logger = getColoredLogger(__name__)

__all__ = ["CastLayerNorm"]


class CastLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype).
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L24
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
