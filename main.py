# %%
from typing import Optional, override

import gated_tree_clip.nn as gtcnn
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from gated_tree_clip.nn import CastLayerNorm, PatchDropout
from tqdm.auto import trange
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)


class TransformerWithSyntacticDistance(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.induction_head = gtcnn.SyntacticDistanceGate(embed_dim, num_heads, batch_first=batch_first)
        self.res_attn_blocks = nn.ModuleList(
            [
                gtcnn.ResidualAttentionWithSyntacticDistanceBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=batch_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
        is_checkpoint: bool = False,
        return_weight: bool = False,
        return_distance: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        atention_gate, distance = self.induction_head(x)

        for i, res_attn_block in enumerate(self.res_attn_blocks):
            if is_checkpoint and not torch.jit.is_scripting():
                x, attn_weight, distance = torch.utils.checkpoint.checkpoint(res_attn_block, x, None, None, attention_mask)
            else:
                x, attn_weight, distance = res_attn_block(
                    x, attn_mask=attention_mask, attn_gate=attention_gate if i == 0 else None
                )

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        ret = [x]
        if return_weight:
            ret.append(attn_weight)
        if return_distance:
            ret.append(distance)
        return tuple(ret) if len(ret) > 1 else ret[0]


class VisionTransformerWithSyntacticDistance(gtcnn.VisionTransformer, TransformerWithSyntacticDistance): ...


class TextTransformerWithSyntacticDistance(gtcnn.TextTransformer, TransformerWithSyntacticDistance): ...


tokenizer = open_clip.get_tokenizer("ViT-B-32")
sentences = ["a photo of a cat", "a photo of a dog"]
tokens = tokenizer(sentences)

# def convert_model_params_laion2b_s34b_b79k_to_CLIPBase512(model_laion2b_s34b_b79k, verbose: bool = False) -> gtcnn.CLIPBase:
#     model_laion2b_s34b_b79k_state_dict = model_laion2b_s34b_b79k.state_dict()
#     weight_mapping = {
#         "positional_embedding": "textual.backbone.positional_embedding",
#         "text_projection": "textual.backbone.head_weight",
#         "visual.class_embedding": "visual.backbone.class_embedding",
#         "visual.positional_embedding": "visual.backbone.positional_embedding",
#         "visual.proj": "visual.backbone.head_weight",
#         "visual.conv1.weight": "visual.backbone.conv.weight",
#         "visual.ln_pre.weight": "visual.backbone.layernorm_pre.weight",
#         "visual.ln_pre.bias": "visual.backbone.layernorm_pre.bias",
#         "visual.ln_post.weight": "visual.backbone.layernorm_post.weight",
#         "visual.ln_post.bias": "visual.backbone.layernorm_post.bias",
#         "token_embedding.weight": "textual.backbone.embedding.weight",
#         "logit_scale": "logit_scale",
#         "ln_final.weight": "textual.backbone.layernorm_post.weight",
#         "ln_final.bias": "textual.backbone.layernorm_post.bias",
#     }
#     for i in range(12):
#         weight_mapping.update(
#             {
#                 f"visual.transformer.resblocks.{i}.ln_1.weight": f"visual.backbone.res_attn_blocks.{i}.layer_norm_1.weight",
#                 f"visual.transformer.resblocks.{i}.ln_1.bias": f"visual.backbone.res_attn_blocks.{i}.layer_norm_1.bias",
#                 f"visual.transformer.resblocks.{i}.attn.in_proj_weight": f"visual.backbone.res_attn_blocks.{i}.attention.in_proj_weight",
#                 f"visual.transformer.resblocks.{i}.attn.in_proj_bias": f"visual.backbone.res_attn_blocks.{i}.attention.in_proj_bias",
#                 f"visual.transformer.resblocks.{i}.attn.out_proj.weight": f"visual.backbone.res_attn_blocks.{i}.attention.out_proj.weight",
#                 f"visual.transformer.resblocks.{i}.attn.out_proj.bias": f"visual.backbone.res_attn_blocks.{i}.attention.out_proj.bias",
#                 f"visual.transformer.resblocks.{i}.ln_2.weight": f"visual.backbone.res_attn_blocks.{i}.layer_norm_2.weight",
#                 f"visual.transformer.resblocks.{i}.ln_2.bias": f"visual.backbone.res_attn_blocks.{i}.layer_norm_2.bias",
#                 f"visual.transformer.resblocks.{i}.mlp.c_fc.weight": f"visual.backbone.res_attn_blocks.{i}.res_mlp.0.weight",
#                 f"visual.transformer.resblocks.{i}.mlp.c_fc.bias": f"visual.backbone.res_attn_blocks.{i}.res_mlp.0.bias",
#                 f"visual.transformer.resblocks.{i}.mlp.c_proj.weight": f"visual.backbone.res_attn_blocks.{i}.res_mlp.2.weight",
#                 f"visual.transformer.resblocks.{i}.mlp.c_proj.bias": f"visual.backbone.res_attn_blocks.{i}.res_mlp.2.bias",
#                 f"transformer.resblocks.{i}.ln_1.weight": f"textual.backbone.res_attn_blocks.{i}.layer_norm_1.weight",
#                 f"transformer.resblocks.{i}.ln_1.bias": f"textual.backbone.res_attn_blocks.{i}.layer_norm_1.bias",
#                 f"transformer.resblocks.{i}.attn.in_proj_weight": f"textual.backbone.res_attn_blocks.{i}.attention.in_proj_weight",
#                 f"transformer.resblocks.{i}.attn.in_proj_bias": f"textual.backbone.res_attn_blocks.{i}.attention.in_proj_bias",
#                 f"transformer.resblocks.{i}.attn.out_proj.weight": f"textual.backbone.res_attn_blocks.{i}.attention.out_proj.weight",
#                 f"transformer.resblocks.{i}.attn.out_proj.bias": f"textual.backbone.res_attn_blocks.{i}.attention.out_proj.bias",
#                 f"transformer.resblocks.{i}.ln_2.weight": f"textual.backbone.res_attn_blocks.{i}.layer_norm_2.weight",
#                 f"transformer.resblocks.{i}.ln_2.bias": f"textual.backbone.res_attn_blocks.{i}.layer_norm_2.bias",
#                 f"transformer.resblocks.{i}.mlp.c_fc.weight": f"textual.backbone.res_attn_blocks.{i}.res_mlp.0.weight",
#                 f"transformer.resblocks.{i}.mlp.c_fc.bias": f"textual.backbone.res_attn_blocks.{i}.res_mlp.0.bias",
#                 f"transformer.resblocks.{i}.mlp.c_proj.weight": f"textual.backbone.res_attn_blocks.{i}.res_mlp.2.weight",
#                 f"transformer.resblocks.{i}.mlp.c_proj.bias": f"textual.backbone.res_attn_blocks.{i}.res_mlp.2.bias",
#             }
#         )

#     target_model = gtcnn.CLIPBase(512)
#     target_model_state_dict = target_model.state_dict()
#     for k, v in weight_mapping.items():
#         assert (
#             model_laion2b_s34b_b79k_state_dict[k].shape == target_model_state_dict[v].shape
#         ), f"{k=}, {v=}, {model_laion2b_s34b_b79k_state_dict[k].shape=}, {target_model_state_dict[v].shape=}"
#         if verbose:
#             print(k, "->", v)
#         target_model_state_dict[v] = model_laion2b_s34b_b79k_state_dict[k]
#     target_model.load_state_dict(target_model_state_dict)
#     return target_model


# tokenizer = open_clip.get_tokenizer("ViT-B-32")
# openclip_model, _, transform_openclip = open_clip.create_model_and_transforms(
#     "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", None)
# )
# openclip_model.eval()

# model = convert_model_params_laion2b_s34b_b79k_to_CLIPBase512(openclip_model)

images = torch.rand(2, 3, 224, 224)
embed_dim = 512
model = gtcnn.CLIPBase(
    embed_dim,
    visual_backbone=VisionTransformerWithSyntacticDistance(embed_dim),
    textual_backbone=TextTransformerWithSyntacticDistance(embed_dim),
)
image_feats = model.encode_image(images)
text_feats = model.encode_text(tokens)
text_probs = (100 * image_feats @ text_feats.T).softmax(dim=-1)
print(image_feats)
print(text_feats)
print(image_feats.shape, text_feats.shape)
print(text_probs)
# %%
