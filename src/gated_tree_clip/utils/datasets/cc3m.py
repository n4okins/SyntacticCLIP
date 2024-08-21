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


class DALICC3MDataLoader:
    def __init__(
        self,
        path_to_cc3m_split: Path,
        batch_size: int = 1,
        num_threads: int = 2,
        device_id: int = 0,
        seed: int = 0,
        shuffle: bool = False,
        reader_name: str = "CC3MReader",
        dali_device: Literal["cpu", "gpu"] = "gpu",
    ):
        self.path_to_cc3m_split = path_to_cc3m_split
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed
        self.dali_device = dali_device

        self.dali_iterator = DALIGenericIterator(
            self.build(
                path_to_cc3m_split,
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=device_id,
                seed=seed,
                shuffle=shuffle,
                reader_name=reader_name,
            ),
            ["jpg", "json"],
            dynamic_shape=False,
            auto_reset=True,
            prepare_first_batch=False,
            reader_name=reader_name,
            last_batch_policy=LastBatchPolicy.DROP,
        )

    @pipeline_def
    def build(self, path_to_cc3m_split: Path, shuffle: bool = True, reader_name: str = "CC3MReader"):
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
        img = fn.resize(img, device=self.dali_device, resize_x=224, resize_y=224)
        img = img / 255.0
        img = fn.crop_mirror_normalize(
            img,
            dtype=types.FLOAT,
            device=self.dali_device,
            mean=MagicNumbers.RGB_CLIP_IMAGE_MEAN,
            std=MagicNumbers.RGB_CLIP_IMAGE_STD,
        )
        return img, fn.pad(info, device="cpu")

    def __next__(self):
        data = next(self.dali_iterator)
        if data:
            data = data[0]
        images, metadata = data["jpg"], data["json"].numpy()
        metadata = [json.loads("".join([chr(o) for o in row.tolist() if o != 0])) for row in metadata]
        return images, metadata

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dali_iterator)
