import os
from dataclasses import dataclass

from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("DEBUG")
PROJECT_ROOT = initializer(globals(), logger=logger)
logger.info(f"{PROJECT_ROOT=}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


__all__ = ["MagicNumbers"]


@dataclass(frozen=True)
class _MagicNumbers:
    RGB_CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    RGB_CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


MagicNumbers = _MagicNumbers()
