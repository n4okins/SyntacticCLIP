
import os
from pathlib import Path

import gated_tree_clip.nn as gtnn
import torch
from dotenv import load_dotenv
from gated_tree_clip.utils.clogging import getColoredLogger

load_dotenv()

logger = getColoredLogger(__name__)
logger.setLevel("INFO")

datasets_dir = Path.home() / "datasets"

pwd = Path(__file__).parent
logger.info(f"{pwd=}")
models_dir = pwd / "models"
images_dir = pwd / "ignores"

HOSTNAME = os.uname()[1]
CONTROL_SERVER_PREFIX = os.environ.get("CONTROL_SERVER_PREFIX", None)
assert CONTROL_SERVER_PREFIX is not None, "CONTROL_SERVER_PREFIX is not set"
assert not HOSTNAME.startswith(CONTROL_SERVER_PREFIX), f"{HOSTNAME=}, {CONTROL_SERVER_PREFIX=}"
logger.info(f"{HOSTNAME=}, {CONTROL_SERVER_PREFIX=}")

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
            f"visual.transformer.resblocks.{i}.ln_1.weight": f"visual.backbone.res_attn_blocks.{i}.layer_norm_1.weight",
            f"visual.transformer.resblocks.{i}.ln_1.bias": f"visual.backbone.res_attn_blocks.{i}.layer_norm_1.bias",
            f"visual.transformer.resblocks.{i}.attn.in_proj_weight": f"visual.backbone.res_attn_blocks.{i}.attention.in_proj_weight",
            f"visual.transformer.resblocks.{i}.attn.in_proj_bias": f"visual.backbone.res_attn_blocks.{i}.attention.in_proj_bias",
            f"visual.transformer.resblocks.{i}.attn.out_proj.weight": f"visual.backbone.res_attn_blocks.{i}.attention.out_proj.weight",
            f"visual.transformer.resblocks.{i}.attn.out_proj.bias": f"visual.backbone.res_attn_blocks.{i}.attention.out_proj.bias",
            f"visual.transformer.resblocks.{i}.ln_2.weight": f"visual.backbone.res_attn_blocks.{i}.layer_norm_2.weight",
            f"visual.transformer.resblocks.{i}.ln_2.bias": f"visual.backbone.res_attn_blocks.{i}.layer_norm_2.bias",
            f"visual.transformer.resblocks.{i}.mlp.c_fc.weight": f"visual.backbone.res_attn_blocks.{i}.res_mlp.0.weight",
            f"visual.transformer.resblocks.{i}.mlp.c_fc.bias": f"visual.backbone.res_attn_blocks.{i}.res_mlp.0.bias",
            f"visual.transformer.resblocks.{i}.mlp.c_proj.weight": f"visual.backbone.res_attn_blocks.{i}.res_mlp.2.weight",
            f"visual.transformer.resblocks.{i}.mlp.c_proj.bias": f"visual.backbone.res_attn_blocks.{i}.res_mlp.2.bias",
            f"transformer.resblocks.{i}.ln_1.weight": f"textual.backbone.res_attn_blocks.{i}.layer_norm_1.weight",
            f"transformer.resblocks.{i}.ln_1.bias": f"textual.backbone.res_attn_blocks.{i}.layer_norm_1.bias",
            f"transformer.resblocks.{i}.attn.in_proj_weight": f"textual.backbone.res_attn_blocks.{i}.attention.in_proj_weight",
            f"transformer.resblocks.{i}.attn.in_proj_bias": f"textual.backbone.res_attn_blocks.{i}.attention.in_proj_bias",
            f"transformer.resblocks.{i}.attn.out_proj.weight": f"textual.backbone.res_attn_blocks.{i}.attention.out_proj.weight",
            f"transformer.resblocks.{i}.attn.out_proj.bias": f"textual.backbone.res_attn_blocks.{i}.attention.out_proj.bias",
            f"transformer.resblocks.{i}.ln_2.weight": f"textual.backbone.res_attn_blocks.{i}.layer_norm_2.weight",
            f"transformer.resblocks.{i}.ln_2.bias": f"textual.backbone.res_attn_blocks.{i}.layer_norm_2.bias",
            f"transformer.resblocks.{i}.mlp.c_fc.weight": f"textual.backbone.res_attn_blocks.{i}.res_mlp.0.weight",
            f"transformer.resblocks.{i}.mlp.c_fc.bias": f"textual.backbone.res_attn_blocks.{i}.res_mlp.0.bias",
            f"transformer.resblocks.{i}.mlp.c_proj.weight": f"textual.backbone.res_attn_blocks.{i}.res_mlp.2.weight",
            f"transformer.resblocks.{i}.mlp.c_proj.bias": f"textual.backbone.res_attn_blocks.{i}.res_mlp.2.bias",
        }
    )

clipbase = gtnn.CLIPBase(512)
pretrained_state = torch.load(models_dir / "laion_CLIP-ViT-B-32-laion2B-s34B-b79K.pth", weights_only=False)
clipbase_state = clipbase.state_dict()
for k, v in weight_mapping.items():
    assert  pretrained_state[k].shape == clipbase_state[v].shape, f"{k=}, {v=}, {pretrained_state[k].shape=}, {clipbase_state[v].shape=}"
    print(k, "->", v)
    clipbase_state[v] = pretrained_state[k]
torch.save(clipbase_state, models_dir / "CLIPBase.pth")