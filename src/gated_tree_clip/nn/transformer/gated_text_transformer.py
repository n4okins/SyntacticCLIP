from typing import Optional

import torch

from .gated_transformer import SyntacticTransformer
from .text_transformer import TextTransformer


class SyntacticTextTransformer(TextTransformer):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_layers, batch_first)
        self.transformer = SyntacticTransformer(embed_dim, num_heads, num_layers, batch_first)

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
