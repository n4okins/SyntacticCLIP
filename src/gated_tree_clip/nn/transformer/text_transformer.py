from typing import Optional

import torch
import torch.nn as nn
from utils.clogging import getColoredLogger

from ..normalization.cast_layernorm import CastLayerNorm
from ..transformer.transformer import Transformer

logger = getColoredLogger(__name__)


class TextTransformer(nn.Module):
    """Text Transformer
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of heads
        num_layers (int): Number of layers
        vocab_size (int): Vocabulary size
        vocab_embed_dim (int): Vocabulary embedding dimension
        max_context_length (int): Maximum context length
        pad_token_id (int): Padding token ID
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        batch_first: bool = True,
        *,
        vocab_size: int = 49408,
        vocab_embed_dim: int = 512,
        max_context_length: int = 77,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.transformer = Transformer(embed_dim, num_heads, num_layers, batch_first)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.vocab_embed_dim = vocab_embed_dim
        self.max_context_length = max_context_length
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, vocab_embed_dim, padding_idx=pad_token_id)
        self.positional_embedding = nn.Parameter(torch.zeros(max_context_length, vocab_embed_dim))

        self.layernorm_post = CastLayerNorm(normalized_shape=vocab_embed_dim)

        self.attention_mask: torch.Tensor
        self.register_buffer(
            "attention_mask",
            torch.zeros(max_context_length, max_context_length).fill_(float("-inf")).triu_(1),
            persistent=False,
        )

        self.head_weight = nn.Parameter(torch.randn(vocab_embed_dim, embed_dim))

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, sequence_length]
        """
        x_ = x
        batch_size, sequence_length = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding[:sequence_length]
        x, w = self.transformer(
            x,
            attention_mask=attention_mask or self.attention_mask,
            return_weights=return_weights,
        )
        x = self.layernorm_post(x)

        # _tokens: unused
        pooled, _tokens = x[torch.arange(batch_size), x_.argmax(dim=-1)], x
        pooled = pooled @ self.head_weight

        return pooled, w
