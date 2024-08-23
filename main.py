# %%
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel
from utils.clogging import getColoredLogger
from utils.dummy import DummyObject
from utils.initialize import initializer

# Logger Settings
logger = getColoredLogger(__name__)
logger.setLevel("DEBUG")

# Project Init
PROJECT_ROOT = initializer(globals(), logger=logger)
PROJECT_NAME = "misc"
USE_WANDB_LOG = False

# Torch distributed settings
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_CUDA_AVAILABLE = torch.cuda.is_available()
if IS_DISTRIBUTED:
    WORLD_SIZE = torch.distributed.get_world_size()
    LOCAL_RANK = torch.distributed.get_rank()
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    logger.info(f"LOCAL_RANK={LOCAL_RANK}, WORLD_SIZE={WORLD_SIZE}")

if USE_WANDB_LOG and LOCAL_RANK == 0:
    import wandb

    wandb.init(project=PROJECT_NAME, save_code=True)
else:
    wandb = DummyObject()

PROJECT_INFOMATION_DICT = dict(
    TIMESTAMP=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    PROJECT_ROOT=PROJECT_ROOT,
    PROJECT_NAME=PROJECT_NAME,
    WORLD_SIZE=WORLD_SIZE,
    LOCAL_RANK=LOCAL_RANK,
    IS_DISTRIBUTED=IS_DISTRIBUTED,
    USE_WANDB_LOG=USE_WANDB_LOG,
    TORCH_VERSION=torch.__version__,
    IS_CUDA_AVAILABLE=IS_CUDA_AVAILABLE,
    TORCH_CUDA_VERSION=torch.version.cuda,
    TORCH_CUDNN_VERSION=torch.backends.cudnn.version(),
    TORCH_DEVICE_COUNT=torch.cuda.device_count(),
    TORCH_DEVICES_INFO=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())],
)

# Print Project Information
logger.info("=" * 16 + " Project Information Begin " + "=" * 16)
for k, v in PROJECT_INFOMATION_DICT.items():
    tab = 3 - len(k) // 6
    if tab == 0:
        tab += int(len(k) % 6 == 0)
    tab += 1
    logger.info(f" | {k}" + "\t" * tab + f"{v}")
wandb.config.update({"project_information": PROJECT_INFOMATION_DICT})
logger.info("=" * 16 + " Project Information End " + "=" * 16)

# %%
# model
model = None

if model is not None and IS_DISTRIBUTED and IS_CUDA_AVAILABLE:
    TORCH_STRAEM = torch.cuda.Stream()
    TORCH_STRAEM.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(TORCH_STRAEM):
        model = model.to(LOCAL_RANK)
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        torch.cuda.current_stream().wait_stream(TORCH_STRAEM)

    logger.info("Model is DistributedDataParallel")

# %%
