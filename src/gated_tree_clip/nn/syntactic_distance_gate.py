from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clogging import getColoredLogger

logger = getColoredLogger(__name__)


class SyntacticDistanceGate(nn.Module):
    def __init__(
        self,
        in_embed_dim: int,
        num_lookback_range: int,
        num_gate_heads: int = 2,
        *,
        tau: float = 1.0,
        dropout_p: float = 0.0,
        batch_first: bool = True,
        distance_activation_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.lookback_range = num_lookback_range
        self.batch_first = batch_first
        self.tau = tau
        self.num_gate_heads = num_gate_heads
        self.conv = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv1d(in_embed_dim, in_embed_dim, 1),
            nn.BatchNorm1d(in_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(in_embed_dim, in_embed_dim, 1),
            nn.BatchNorm1d(in_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(in_embed_dim, num_gate_heads, num_lookback_range, padding=num_lookback_range),
        )
        self.distance_activation_fn = distance_activation_fn or nn.Tanh()

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, embed_dim) or (seq_len, batch_size, embed_dim)
        # gate: (batch_size, seq_len, seq_len)
        # distance: (batch_size, seq_len, 1)
        if x.dim() == 2:
            if self.batch_first:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)

        if self.batch_first:
            x = x.transpose(1, 2)
        else:
            x = x.permute(1, 2, 0)

        batch_size, embed_dim, seq_len = x.size()

        # distance: Syntactic Distance [d_i, ...]: i番目の単語の構文距離 (構文高？)
        # distance := distance  (batch_size, seq_len, 1)
        # distance[i] = \tanh(W_D [k_{i-M}, k_{i-M+1}, ..., K_{i}]^{\top} + b_D)
        # conv_input: (batch_size, embed_dim, seq_len)
        distance = self.conv(x)
        # disttance : (batch_size, distance_dim, seq_len + lookback_range)
        distance = distance[:, :, 1 : -self.lookback_range]
        distance = self.distance_activation_fn(distance)
        distance = distance.view(batch_size * self.num_gate_heads, seq_len, 1)
        # distance: (batch_size * num_gates_heads, seq_len, 1)
        alpha = (F.hardtanh((distance - distance.transpose(2, 1)) * self.tau) + 1) / 2
        lower_tri = alpha.tril(diagonal=-1) + torch.ones_like(alpha).triu(diagonal=0)
        upper_tri = torch.ones_like(alpha).tril(diagonal=0) + alpha.triu(diagonal=1)
        gate = lower_tri * upper_tri
        distance = distance.view(batch_size, self.num_gate_heads, seq_len).transpose(1, 2)

        gate = gate.contiguous()
        distance = distance.contiguous()
        # gate := gate  (batch_size, seq_len, seq_len), 0 <= gate <= 1
        # distance := distance  (batch_size, seq_len, num_gate_heads), -1 <= distance <= 1
        return gate, distance
