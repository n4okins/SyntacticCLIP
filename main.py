# %%
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import gated_tree_clip.nn as gtcnn
import open_clip
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchinfo
from gated_tree_clip.nn.clip.convert.from_laion2b_s34b_b79k import convert_model_params_laion2b_s34b_b79k_to_CLIP512
from gated_tree_clip.utils.datasets.cc3m import DALICC3MDataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("DEBUG")
PROJECT_ROOT = initializer(globals(), logger=logger)
logger.info(f"{PROJECT_ROOT=}")


def inference():
    import requests
    from PIL import Image
    from torchvision import transforms

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

    model = gtcnn.CLIP(512)
    n = 9
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
