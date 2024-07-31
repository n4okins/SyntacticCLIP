# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/fairseq_dropout.py

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from ..utils.clogging import getColoredLogger

logger = getColoredLogger(__name__)


class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self, name: str, retain_dropout: bool = False, retain_dropout_modules: Optional[list[str]] = None, **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} " "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info("Enabling dropout during inference for module: {}".format(name))
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))
