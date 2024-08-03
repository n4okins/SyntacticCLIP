import gated_tree_clip.nn as gtcnn
import pytest
import torch
import torch.nn as nn


class TestLayers:
    def test_multihead_attention_layer(self):
        embed_dim = 4
        num_heads = 1
        bias = False
        batch_first = True
        batch_size = 2
        seq_len = 5
        x = torch.ones((batch_size, seq_len, embed_dim))
        q = k = v = x

        torch_model = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            batch_first=batch_first,
        )
        gtc_model = gtcnn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            batch_first=batch_first,
        )
        gtc_model.load_state_dict(torch_model.state_dict())

        gtc_attn_output, gtc_attn_weights = gtc_model(q, k, v)
        gtc_attn_output, gtc_attn_weights = torch_model(q, k, v)

        assert torch.allclose(gtc_attn_output, gtc_attn_output)
        assert torch.allclose(gtc_attn_weights, gtc_attn_weights)
