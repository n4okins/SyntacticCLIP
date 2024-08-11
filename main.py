# %%
from typing import Optional, override

import gated_tree_clip.nn as gtcnn
import open_clip
import torch
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)

tokenizer = open_clip.get_tokenizer("ViT-B-32")
sentences = ["a photo of a cat", "a photo of a dog"]
tokens = tokenizer(sentences)

images = torch.rand(2, 3, 224, 224)
embed_dim = 512
model = gtcnn.CLIPBase(
    embed_dim,
    visual_backbone=gtcnn.VisionTransformerWithSyntacticDistance(embed_dim),
    textual_backbone=gtcnn.TextTransformerWithSyntacticDistance(embed_dim),
)
image_feats = model.encode_image(images)
text_feats = model.encode_text(tokens)
text_probs = (100 * image_feats @ text_feats.T).softmax(dim=-1)
print(image_feats)
print(text_feats)
print(image_feats.shape, text_feats.shape)
print(text_probs)
# %%
