import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructFormer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(StructFormer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
