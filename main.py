# %%
import os
from typing import Any, Optional

import gated_tree_clip.nn as gtcnn
import open_clip
import requests
import torch
import torchinfo
from PIL import Image
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)


class VisionTransformerWithSyntacticDistance(gtcnn.VisionTransformer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_layers, batch_first)
        self.transformer = gtcnn.TransformerWithSyntacticDistance(self.patch_embed_dim, num_heads, num_layers, batch_first)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
        return_weight: bool = False,
        return_distance: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        """
        batch_size, channels, height, width = x.shape

        self.positional_embedding.to(x.dtype)

        # [batch, channels, height, width] -> [batch, self.patch_embed_dim, *self.positional_grid_size]
        x = self.conv(x)

        # num_patches := self.positional_grid_size[0] * self.positional_grid_size[1]
        # [batch, self.patch_embed_dim, *self.positional_grid_size] -> [batch, num_patches, self.patch_embed_dim]
        x = x.reshape(batch_size, self.patch_embed_dim, -1).permute(0, 2, 1)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = torch.cat([self.class_embedding.view(1, 1, -1).expand(batch_size, -1, -1), x], dim=1)
        x = x + self.positional_embedding

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, num_patches + 1, self.patch_embed_dim]
        x = self.patchdropout_pre(x)
        x = self.layernorm_pre(x)
        print(return_weight, return_distance)
        x, *w = self.transformer(
            x,
            attention_mask=attention_mask,
            attention_gate=attention_gate,
            return_weight=return_weight,
            return_distance=return_distance,
        )
        print("Vision\n", x, w)
        x = self.layernorm_post(x)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, self.patch_embed_dim], [batch, num_patches, self.patch_embed_dim]
        # _tokens: unused
        pooled, _tokens = x[:, 0], x[:, 1:]

        # [batch, self.patch_embed_dim] -> [batch, self.embed_dim]
        pooled = pooled @ self.head_weight

        if return_weight:
            return pooled, w
        return pooled


class TextTransformerWithSyntacticDistance(gtcnn.TextTransformer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_layers, batch_first)
        self.transformer = gtcnn.TransformerWithSyntacticDistance(embed_dim, num_heads, num_layers, batch_first)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
        return_weight: bool = False,
        return_distance: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, sequence_length]
        """
        x_ = x
        batch_size, sequence_length = x.shape

        x = self.embedding(x)
        x = x + self.positional_embedding[:sequence_length]
        x, *w = self.transformer(
            x,
            attention_mask=attention_mask or self.attention_mask,
            attention_gate=attention_gate,
            return_weight=return_weight,
            return_distance=return_distance,
        )
        print("Text:\n", x, w)
        x = self.layernorm_post(x)

        # _tokens: unused
        pooled, _tokens = x[torch.arange(batch_size), x_.argmax(dim=-1)], x
        pooled = pooled @ self.head_weight

        if return_weight:
            return pooled, w
        return pooled


class SyntacticCLIP(gtcnn.CLIPBase):
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


def convert_model_params_laion2b_s34b_b79k_to_SyntacticCLIP512(
    model_laion2b_s34b_b79k, verbose: bool = False
) -> gtcnn.CLIPBase:
    model_laion2b_s34b_b79k_state_dict = model_laion2b_s34b_b79k.state_dict()
    weight_mapping = {
        "positional_embedding": "textual.backbone.positional_embedding",
        "text_projection": "textual.backbone.head_weight",
        "visual.class_embedding": "visual.backbone.class_embedding",
        "visual.positional_embedding": "visual.backbone.positional_embedding",
        "visual.proj": "visual.backbone.head_weight",
        "visual.conv1.weight": "visual.backbone.conv.weight",
        "visual.ln_pre.weight": "visual.backbone.layernorm_pre.weight",
        "visual.ln_pre.bias": "visual.backbone.layernorm_pre.bias",
        "visual.ln_post.weight": "visual.backbone.layernorm_post.weight",
        "visual.ln_post.bias": "visual.backbone.layernorm_post.bias",
        "token_embedding.weight": "textual.backbone.embedding.weight",
        "logit_scale": "logit_scale",
        "ln_final.weight": "textual.backbone.layernorm_post.weight",
        "ln_final.bias": "textual.backbone.layernorm_post.bias",
    }
    for i in range(12):
        weight_mapping.update(
            {
                f"visual.transformer.resblocks.{i}.ln_1.weight": f"visual.backbone.transformer.res_attn_blocks.{i}.layer_norm_1.weight",
                f"visual.transformer.resblocks.{i}.ln_1.bias": f"visual.backbone.transformer.res_attn_blocks.{i}.layer_norm_1.bias",
                f"visual.transformer.resblocks.{i}.attn.in_proj_weight": f"visual.backbone.transformer.res_attn_blocks.{i}.attention.in_proj_weight",
                f"visual.transformer.resblocks.{i}.attn.in_proj_bias": f"visual.backbone.transformer.res_attn_blocks.{i}.attention.in_proj_bias",
                f"visual.transformer.resblocks.{i}.attn.out_proj.weight": f"visual.backbone.transformer.res_attn_blocks.{i}.attention.out_proj.weight",
                f"visual.transformer.resblocks.{i}.attn.out_proj.bias": f"visual.backbone.transformer.res_attn_blocks.{i}.attention.out_proj.bias",
                f"visual.transformer.resblocks.{i}.ln_2.weight": f"visual.backbone.transformer.res_attn_blocks.{i}.layer_norm_2.weight",
                f"visual.transformer.resblocks.{i}.ln_2.bias": f"visual.backbone.transformer.res_attn_blocks.{i}.layer_norm_2.bias",
                f"visual.transformer.resblocks.{i}.mlp.c_fc.weight": f"visual.backbone.transformer.res_attn_blocks.{i}.res_mlp.0.weight",
                f"visual.transformer.resblocks.{i}.mlp.c_fc.bias": f"visual.backbone.transformer.res_attn_blocks.{i}.res_mlp.0.bias",
                f"visual.transformer.resblocks.{i}.mlp.c_proj.weight": f"visual.backbone.transformer.res_attn_blocks.{i}.res_mlp.2.weight",
                f"visual.transformer.resblocks.{i}.mlp.c_proj.bias": f"visual.backbone.transformer.res_attn_blocks.{i}.res_mlp.2.bias",
                f"transformer.resblocks.{i}.ln_1.weight": f"textual.backbone.transformer.res_attn_blocks.{i}.layer_norm_1.weight",
                f"transformer.resblocks.{i}.ln_1.bias": f"textual.backbone.transformer.res_attn_blocks.{i}.layer_norm_1.bias",
                f"transformer.resblocks.{i}.attn.in_proj_weight": f"textual.backbone.transformer.res_attn_blocks.{i}.attention.in_proj_weight",
                f"transformer.resblocks.{i}.attn.in_proj_bias": f"textual.backbone.transformer.res_attn_blocks.{i}.attention.in_proj_bias",
                f"transformer.resblocks.{i}.attn.out_proj.weight": f"textual.backbone.transformer.res_attn_blocks.{i}.attention.out_proj.weight",
                f"transformer.resblocks.{i}.attn.out_proj.bias": f"textual.backbone.transformer.res_attn_blocks.{i}.attention.out_proj.bias",
                f"transformer.resblocks.{i}.ln_2.weight": f"textual.backbone.transformer.res_attn_blocks.{i}.layer_norm_2.weight",
                f"transformer.resblocks.{i}.ln_2.bias": f"textual.backbone.transformer.res_attn_blocks.{i}.layer_norm_2.bias",
                f"transformer.resblocks.{i}.mlp.c_fc.weight": f"textual.backbone.transformer.res_attn_blocks.{i}.res_mlp.0.weight",
                f"transformer.resblocks.{i}.mlp.c_fc.bias": f"textual.backbone.transformer.res_attn_blocks.{i}.res_mlp.0.bias",
                f"transformer.resblocks.{i}.mlp.c_proj.weight": f"textual.backbone.transformer.res_attn_blocks.{i}.res_mlp.2.weight",
                f"transformer.resblocks.{i}.mlp.c_proj.bias": f"textual.backbone.transformer.res_attn_blocks.{i}.res_mlp.2.bias",
            }
        )

    target_model = SyntacticCLIP(512)
    target_model_state_dict = target_model.state_dict()
    for k, v in weight_mapping.items():
        assert (
            model_laion2b_s34b_b79k_state_dict[k].shape == target_model_state_dict[v].shape
        ), f"{k=}, {v=}, {model_laion2b_s34b_b79k_state_dict[k].shape=}, {target_model_state_dict[v].shape=}"
        if verbose:
            print(k, "->", v)
        target_model_state_dict[v] = model_laion2b_s34b_b79k_state_dict[k]
    target_model.load_state_dict(target_model_state_dict)
    return target_model


openclip_model, _, transform = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", None)
)
openclip_model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

urls = ("http://images.cocodataset.org/val2017/000000039769.jpg", "http://images.cocodataset.org/val2017/000000294350.jpg")
images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in urls]
images = torch.stack(list(map(transform, images)))

sentences = ["a photo of cat", "a photo of dog", "a photo of bird", "a photo of person"]
tokens = tokenizer(sentences)

embed_dim = 512
model = convert_model_params_laion2b_s34b_b79k_to_SyntacticCLIP512(openclip_model)
model.eval()
print(
    torchinfo.summary(model, input_size=[(1, 3, 224, 224), (1, 77)], device="cpu", dtypes=[torch.float32, torch.long], depth=6)
)

with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    probs, (image_outputs, text_outputs), (image_weights, text_weights) = model(
        images, tokens, softmax=True, normalize=True, return_weight=True
    )

print(sentences)
print(probs)
print(image_outputs)
print(text_outputs)
print(image_weights)
print(text_weights)
# %%
