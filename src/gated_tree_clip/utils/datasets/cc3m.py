# %%

import json
from pathlib import Path
from typing import Literal

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from utils.clogging import getColoredLogger

from ..magicvalues import MagicNumbers

logger = getColoredLogger(__name__)


@pipeline_def
def build_cc3m(
    path_to_cc3m_split: Path,
    num_shards: int = 2,
    shard_id: int = 0,
    shuffle: bool = False,
    dali_device: Literal["cpu", "gpu"] = "gpu",
    reader_name: str = "CC3MReader",
):
    """
    Pipeline to load the CC3M dataset.

    Args:
        path_to_cc3m_split (Path): Path to the CC3M split.
        num_shards (int, optional): Number of shards. Defaults to 2.
        shard_id (int, optional): Shard ID. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        dali_device (Literal["cpu", "gpu"], optional): DALI device. Defaults to "gpu".
        reader_name (str, optional): Name of the reader. Defaults to "CC3MReader".

    Usage:
        train_dataloader_params = dict(
            path_to_cc3m_split=CC3M_DIR / "Training",
            num_threads=4,
            batch_size=batch_size,
            num_shards=world_size,
            device_id=local_rank,
            seed=seed + local_rank,
            shard_id=local_rank,
            shuffle=True,
        )
        train_dataloader = DALIGenericIterator(
            build_cc3m(**train_dataloader_params),
            ["jpg", "json"],
            dynamic_shape=False,
            auto_reset=True,
            prepare_first_batch=False,
            reader_name="CC3MReader",
            last_batch_policy=LastBatchPolicy.DROP,
        )
        for data in train_dataloader:
            if data:
                data = data[0]
            images, metadata = data["jpg"], data["json"].numpy()
            metadata = [json.loads("".join([chr(o) for o in row.tolist() if o != 0])) for row in metadata]
            ...
    """
    # (in zsh) > for f (./path/to/*.tar) {wds2idx $f $f:r.index}
    cc3m_tarfiles = list(path_to_cc3m_split.glob("*.tar"))
    cc3m_index_files = (tar.with_suffix(".index") for tar in cc3m_tarfiles)
    cc3m_tarfiles = list(map(str, sorted(cc3m_tarfiles)))
    cc3m_index_files = list(map(str, sorted(cc3m_index_files)))
    logger.info(f"{len(cc3m_tarfiles)=}, {len(cc3m_index_files)=}")
    img_raw, info = fn.readers.webdataset(
        paths=cc3m_tarfiles,
        index_paths=cc3m_index_files,
        ext=["jpg", "json"],
        missing_component_behavior="error",
        random_shuffle=shuffle,
        name=reader_name,
    )
    img = fn.decoders.image(img_raw, device="mixed", output_type=types.RGB)
    img = fn.resize(img, device=dali_device, resize_x=224, resize_y=224)
    img = img / 255.0
    img = fn.crop_mirror_normalize(
        img,
        dtype=types.FLOAT,
        device=dali_device,
        mean=MagicNumbers.RGB_CLIP_IMAGE_MEAN,
        std=MagicNumbers.RGB_CLIP_IMAGE_STD,
    )
    return img, fn.pad(info, device="cpu")
