# %%
import os

import gated_tree_clip.nn as gtnn
import open_clip
import requests
import torch
import torchinfo
from PIL import Image
from torchvision import transforms
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)


# %%
def convert_model_params_laion2b_s34b_b79k_to_CLIPBase512(model_laion2b_s34b_b79k, verbose: bool = False) -> gtnn.CLIPBase:
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

    target_model = gtnn.CLIPBase(512)
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


tokenizer = open_clip.get_tokenizer("ViT-B-32")
openclip_model, _, transform_openclip = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", None)
)
openclip_model.eval()

model = convert_model_params_laion2b_s34b_b79k_to_CLIPBase512(openclip_model)
model.eval()
print("\nOpenCLIP-ViT-B-32 laion2b_s34b_b79k")
torchinfo.summary(
    openclip_model, input_size=[(1, 3, 224, 224), (1, 77)], device="cpu", dtypes=[torch.float32, torch.long], depth=4
)
print("\nCLIPBase512")
torchinfo.summary(model, input_size=[(1, 3, 224, 224), (1, 77)], device="cpu", dtypes=[torch.float32, torch.long], depth=4)
# %%
# CLIPの推論
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
print("\nImage:", url)
# plt.imshow(image)
# plt.axis("off")
# plt.show()

image_tensor = transform(image).unsqueeze(0)
image_tensor_openclip = transform_openclip(image).unsqueeze(0)

sentences = ["a photo of a cat", "a photo of a dog"]
print(f"{sentences=}")
tokens = tokenizer(sentences)

with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    image_features_openclip = openclip_model.encode_image(image_tensor_openclip)
    text_features_openclip = openclip_model.encode_text(tokens)
    image_features_openclip /= image_features_openclip.norm(dim=-1, keepdim=True)
    text_features_openclip /= text_features_openclip.norm(dim=-1, keepdim=True)
    text_probs_openclip = (100 * image_features_openclip @ text_features_openclip.T).softmax(dim=-1)

    logits_openclip, _ = openclip_model.get_logits(image_tensor_openclip, tokens)
    logits_openclip = logits_openclip.softmax(dim=-1)
print(f"openclip_model result: {text_probs_openclip=}, {logits_openclip=}")

with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    image_features, text_features, _ = model(image_tensor_openclip, tokens, normalize=True)  # (transformが違うので)
    # image_features, text_features, _ = model(image_tensor, tokens, normalize=True)
    text_probs = (100 * image_features @ text_features.T).softmax(dim=-1)
    logits, logits_t = model.get_logits(image_tensor_openclip, tokens, normalize=True, softmax=True)

print(f"model result: {text_probs=}, {logits=}")
# %%
