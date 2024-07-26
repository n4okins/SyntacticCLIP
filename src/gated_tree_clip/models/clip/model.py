import torch
import torch.nn as nn


class CLIP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(CLIP, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)