# %%
import os

import gated_tree_clip.nn as gtcnn
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
def convert_model_params_laion2b_s34b_b79k_to_CLIPBase512(model_laion2b_s34b_b79k, verbose: bool = False) -> gtcnn.CLIPBase:
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

    target_model = gtcnn.CLIPBase(512)
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

urls = ("http://images.cocodataset.org/val2017/000000039769.jpg", "http://images.cocodataset.org/val2017/000000294350.jpg")
images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in urls]
print("\nImage (there are two cats):", urls[0])
print("Image (there is a person)", urls[1])
# plt.imshow(image)
# plt.axis("off")
# plt.show()

# images= torch.stack(list(map(transform, images)))
images = torch.stack(list(map(transform_openclip, images)))

sentences = ["a photo of a cat", "a photo of a dog", "a photo of bird", "a photo of person"]
print(f"{sentences=}")
tokens = tokenizer(sentences)

with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    image_features_openclip = openclip_model.encode_image(images)
    text_features_openclip = openclip_model.encode_text(tokens)
    image_features_openclip /= image_features_openclip.norm(dim=-1, keepdim=True)
    text_features_openclip /= text_features_openclip.norm(dim=-1, keepdim=True)
    probs_openclip = (100 * image_features_openclip @ text_features_openclip.T).softmax(dim=-1)

    logits_openclip, _ = openclip_model.get_logits(images, tokens)
    logits_openclip = logits_openclip.softmax(dim=-1)
print(f"openclip_model result: {probs_openclip=}, {logits_openclip=}")

with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    probs, (image_outputs, text_outputs), (image_weights, text_weights) = model(
        images, tokens, softmax=True, normalize=True, return_weights=True
    )

print(f"model result: {probs=}")
# %%
import matplotlib.pyplot as plt

sentences = [
    "the agency described in a statement that the information was a pack of lies.",
    'it said in a news bulletin that reports about the assassination attempt "are cheap lies and rumors."',
]

tokens = tokenizer(sentences)
with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    feats, weights = model.encode_text(tokens, return_weights=True)


print(tokens)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(weights[0, 0, :24, :24].detach().cpu().numpy())
ax[0].axis("off")
ax[1].imshow(weights[1, 0, :24, :24].detach().cpu().numpy())
ax[1].axis("off")
plt.show()
# %%