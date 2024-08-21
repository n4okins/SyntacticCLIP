# %%
import os
from typing import Any, Callable, Iterable, Optional, Type

import open_clip
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from PIL import Image
from torchvision import transforms
from utils.clogging import getColoredLogger
from utils.initialize import initializer

import wandb

logger = getColoredLogger(__name__)
logger.setLevel("DEBUG")
PROJECT_ROOT = initializer(globals(), logger=logger)
logger.info(f"{PROJECT_ROOT=}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class DammyWandb:
    class DammyTable:
        def __call__(self, *args, **kwargs):
            return None

        def add_data(self, *args, **kwargs):
            pass

    class DammyConfig:
        def update(self, *args, **kwargs):
            pass

    def init(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        return self.DammyConfig()

    def log(self, *args, **kwargs):
        pass

    def Table(self, *args, **kwargs):
        return self.DammyTable()

    def finish(self):
        pass


wandb = DammyWandb()


def assert_size_of_tensor(name: str, tensor: Optional[torch.Tensor], *, expected_size: tuple[int, ...], optional=True) -> None:
    if optional and tensor is None:
        return
    else:
        assert tensor.size() == torch.Size(expected_size), f"tensor {name} size is {tensor.size()}, expected {expected_size}"






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
        self.induction_head = SyntacticDistanceGate(embed_dim, num_heads)
        self.res_attn_blocks = nn.ModuleList(
            [
                ResidualAttentionWithSyntacticDistanceBlock(
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
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        atention_gate, distance = self.induction_head(x)

        for i, res_attn_block in enumerate(self.res_attn_blocks):
            x, attn_weight, distance = res_attn_block(
                x,
                attn_mask=attention_mask,
                attn_gate=attention_gate if i == 0 else None,
            )

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        return x, attn_weight, distance


class VisionTransformerWithSyntacticDistance(VisionTransformer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_layers, batch_first)
        self.transformer = TransformerWithSyntacticDistance(self.patch_embed_dim, num_heads, num_layers, batch_first)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
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
        x, w, d = self.transformer(
            x,
            attention_mask=attention_mask,
            attention_gate=attention_gate,
        )
        x = self.layernorm_post(x)

        # [batch, num_patches + 1, self.patch_embed_dim] -> [batch, self.patch_embed_dim], [batch, num_patches, self.patch_embed_dim]
        # _tokens: unused
        pooled, _tokens = x[:, 0], x[:, 1:]

        # [batch, self.patch_embed_dim] -> [batch, self.embed_dim]
        pooled = pooled @ self.head_weight
        return pooled, w, d


class TextTransformerWithSyntacticDistance(TextTransformer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_layers, batch_first)
        self.transformer = TransformerWithSyntacticDistance(embed_dim, num_heads, num_layers, batch_first)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, sequence_length]
        """
        x_ = x
        batch_size, sequence_length = x.shape

        x = self.embedding(x)
        x = x + self.positional_embedding[:sequence_length]
        x, w, d = self.transformer(
            x,
            attention_mask=attention_mask,
            attention_gate=attention_gate,
        )
        x = self.layernorm_post(x)

        # _tokens: unused
        pooled, _tokens = x[torch.arange(batch_size), x_.argmax(dim=-1)], x
        pooled = pooled @ self.head_weight
        return pooled, w, d


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        visual_backbone: Optional[VisionTransformer] = None,
        textual_backbone: Optional[TextTransformer] = None,
    ):
        super().__init__()
        if visual_backbone is None:
            visual_backbone = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=12,
                num_layers=12,
            )

        if textual_backbone is None:
            textual_backbone = TextTransformer(
                embed_dim=embed_dim,
                num_heads=8,
                num_layers=12,
            )
        self.embed_dim = embed_dim
        self.visual = visual_backbone
        self.textual = textual_backbone
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    @property
    def dtype(self):
        return self.visual.conv.weight.dtype

    def encode_image(self, image: torch.Tensor, normalize: bool = True):
        feats, *_ = self.visual(image)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, text: torch.Tensor, normalize: bool = True):
        feats, *_ = self.textual(text)
        if normalize:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def get_features(
        self, image: torch.Tensor, text: torch.Tensor, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image, normalize=normalize)
        text_features = self.encode_text(text, normalize=normalize)
        return image_features, text_features

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        *,
        normalize: bool = True,
        softmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features, text_features = self.get_features(images, tokens, normalize=normalize)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t() + self.logit_bias
        if softmax:
            logits_per_image = logits_per_image.softmax(dim=1)
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


class SyntacticCLIP(CLIP):
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


def convert_model_params_laion2b_s34b_b79k_to_CLIP512(
    model_laion2b_s34b_b79k,
    CLIPModel: Type[CLIP],
    convert_num_layers: int = 12,
    verbose: bool = False,
    freeze: bool = False,
) -> CLIP:
    model_laion2b_s34b_b79k_state_dict = model_laion2b_s34b_b79k.state_dict()
    weight_mapping = {
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.head_weight",
        "visual.class_embedding": "visual.class_embedding",
        "visual.positional_embedding": "visual.positional_embedding",
        "visual.proj": "visual.head_weight",
        "visual.conv1.weight": "visual.conv.weight",
        "visual.ln_pre.weight": "visual.layernorm_pre.weight",
        "visual.ln_pre.bias": "visual.layernorm_pre.bias",
        "visual.ln_post.weight": "visual.layernorm_post.weight",
        "visual.ln_post.bias": "visual.layernorm_post.bias",
        "token_embedding.weight": "textual.embedding.weight",
        "logit_scale": "logit_scale",
        "ln_final.weight": "textual.layernorm_post.weight",
        "ln_final.bias": "textual.layernorm_post.bias",
    }
    for i in range(convert_num_layers):
        weight_mapping.update(
            {
                f"visual.transformer.resblocks.{i}.ln_1.weight": f"visual.transformer.res_attn_blocks.{i}.layernorm_1.weight",
                f"visual.transformer.resblocks.{i}.ln_1.bias": f"visual.transformer.res_attn_blocks.{i}.layernorm_1.bias",
                f"visual.transformer.resblocks.{i}.attn.in_proj_weight": f"visual.transformer.res_attn_blocks.{i}.attention.in_proj_weight",
                f"visual.transformer.resblocks.{i}.attn.in_proj_bias": f"visual.transformer.res_attn_blocks.{i}.attention.in_proj_bias",
                f"visual.transformer.resblocks.{i}.attn.out_proj.weight": f"visual.transformer.res_attn_blocks.{i}.attention.out_proj.weight",
                f"visual.transformer.resblocks.{i}.attn.out_proj.bias": f"visual.transformer.res_attn_blocks.{i}.attention.out_proj.bias",
                f"visual.transformer.resblocks.{i}.ln_2.weight": f"visual.transformer.res_attn_blocks.{i}.layernorm_2.weight",
                f"visual.transformer.resblocks.{i}.ln_2.bias": f"visual.transformer.res_attn_blocks.{i}.layernorm_2.bias",
                f"visual.transformer.resblocks.{i}.mlp.c_fc.weight": f"visual.transformer.res_attn_blocks.{i}.res_mlp.0.weight",
                f"visual.transformer.resblocks.{i}.mlp.c_fc.bias": f"visual.transformer.res_attn_blocks.{i}.res_mlp.0.bias",
                f"visual.transformer.resblocks.{i}.mlp.c_proj.weight": f"visual.transformer.res_attn_blocks.{i}.res_mlp.2.weight",
                f"visual.transformer.resblocks.{i}.mlp.c_proj.bias": f"visual.transformer.res_attn_blocks.{i}.res_mlp.2.bias",
                f"transformer.resblocks.{i}.ln_1.weight": f"textual.transformer.res_attn_blocks.{i}.layernorm_1.weight",
                f"transformer.resblocks.{i}.ln_1.bias": f"textual.transformer.res_attn_blocks.{i}.layernorm_1.bias",
                f"transformer.resblocks.{i}.attn.in_proj_weight": f"textual.transformer.res_attn_blocks.{i}.attention.in_proj_weight",
                f"transformer.resblocks.{i}.attn.in_proj_bias": f"textual.transformer.res_attn_blocks.{i}.attention.in_proj_bias",
                f"transformer.resblocks.{i}.attn.out_proj.weight": f"textual.transformer.res_attn_blocks.{i}.attention.out_proj.weight",
                f"transformer.resblocks.{i}.attn.out_proj.bias": f"textual.transformer.res_attn_blocks.{i}.attention.out_proj.bias",
                f"transformer.resblocks.{i}.ln_2.weight": f"textual.transformer.res_attn_blocks.{i}.layernorm_2.weight",
                f"transformer.resblocks.{i}.ln_2.bias": f"textual.transformer.res_attn_blocks.{i}.layernorm_2.bias",
                f"transformer.resblocks.{i}.mlp.c_fc.weight": f"textual.transformer.res_attn_blocks.{i}.res_mlp.0.weight",
                f"transformer.resblocks.{i}.mlp.c_fc.bias": f"textual.transformer.res_attn_blocks.{i}.res_mlp.0.bias",
                f"transformer.resblocks.{i}.mlp.c_proj.weight": f"textual.transformer.res_attn_blocks.{i}.res_mlp.2.weight",
                f"transformer.resblocks.{i}.mlp.c_proj.bias": f"textual.transformer.res_attn_blocks.{i}.res_mlp.2.bias",
            }
        )

    target_model = CLIPModel(512)
    target_model_state_dict = target_model.state_dict()
    for k, v in weight_mapping.items():
        assert (
            model_laion2b_s34b_b79k_state_dict[k].shape == target_model_state_dict[v].shape
        ), f"{k=}, {v=}, {model_laion2b_s34b_b79k_state_dict[k].shape=}, {target_model_state_dict[v].shape=}"
        if verbose:
            print(k, "->", v)
        target_model_state_dict[v] = model_laion2b_s34b_b79k_state_dict[k]

    target_model.load_state_dict(target_model_state_dict)
    if freeze:
        for name, p in target_model.named_parameters():
            if name in weight_mapping.values():
                p.requires_grad = False
    return target_model


def train_epoch(
    epoch: int,
    model: nn.Module,
    criterion: ContrastiveLoss,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cuda"),
    scaler: Optional[torch.amp.GradScaler] = None,
    max_norm: float = 0.0,
    use_amp: bool = True,
    amp_device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
):
    if scaler is None and use_amp:
        scaler = torch.amp.GradScaler()
    model.train()
    model.to(device)
    total_loss = 0.0
    per10 = len(dataloader) // 10
    logger.info(f"epoch {epoch=} start")
    for i, (images, texts) in enumerate(dataloader):
        images = images.to(device)
        texts = texts.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(enabled=use_amp, device=amp_device, dtype=amp_dtype):
            logits_per_image, logits_per_text = model(images, texts)
            loss = criterion(logits_per_image, logits_per_text)

        if use_amp:
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if i % per10 == 0:
            logger.info(f"{epoch=}, {i=}, {loss=:0.4f}")

    logger.info(f"epoch {epoch=} end")
    return total_loss / len(dataloader)


def inference():
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    openclip_model, _, transform_openclip = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE", None)
    )
    openclip_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    urls = ("http://images.cocodataset.org/val2017/000000039769.jpg", "http://images.cocodataset.org/val2017/000000294350.jpg")
    images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in urls]
    images = torch.stack(list(map(transform, images)))

    sentences = ["a photo of a cat", "a photo of a dog", "a photo of bird", "a photo of person"]
    print(f"{sentences=}")
    tokens = tokenizer(sentences)

    model = CLIP(512)
    n = 12
    model_path = PROJECT_ROOT / "models" / f"{model.__class__.__name__}" / f"{model.__class__.__name__}-Freeze{n:02d}.pth"
    if False and model_path.exists():
        logger.info(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        logger.info("converting model from openclip model")
        model = convert_model_params_laion2b_s34b_b79k_to_CLIP512(
            openclip_model, model.__class__, convert_num_layers=n, freeze=True
        )
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(
            model.state_dict(),
            model_path,
        )

    print(torchinfo.summary(model, input_data=(images, tokens)))
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
        probs, *_ = model(images, tokens, softmax=True, normalize=True)

    print(f"model result: {probs=}")
    for idx in probs.argmax(dim=-1):
        print(sentences[idx.item()])


inference()
# %%
# # %%
# if __name__ == "__main__":
#     epochs = 3
#     batch_size = 8
#     model = CLIP(512)
#     criterion = ContrastiveLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     params = {
#         "model": model,
#         "criterion": criterion,
#     }

#     for i in range(epochs):
#         train_epoch
