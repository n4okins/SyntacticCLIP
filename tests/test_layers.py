from typing import Callable, Optional

import gated_tree_clip.nn as gtcnn
import pytest
import torch
import torch.nn as nn


class TestLayers:
    @pytest.mark.parametrize(
        "batch_size, seq_len, embed_dim, num_heads, bias, batch_first",
        [
            (1, 5, 4, 1, False, False),
            (2, 5, 4, 4, False, False),
            (32, 100, 512, 32, False, False),
            (1, 5, 4, 1, True, False),
            (2, 5, 4, 4, True, False),
            (32, 100, 512, 32, True, False),
        ],
    )
    def test_multihead_attention_layer(
        self,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        bias: bool,
        batch_first: bool,
    ):
        x = torch.ones((seq_len, batch_size, embed_dim))
        if batch_first:
            x = x.permute(1, 0, 2)

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

    @pytest.mark.parametrize(
        "batch_size, seq_len, embed_dim, num_lookback_range, num_gate_heads, tau, dropout_p, batch_first, distance_activation_fn",
        [
            (1, 5, 4, 1, 1, 1.0, 0.0, False, None),
            (32, 100, 512, 32, 32, 10.0, 0.5, False, None),
            (1, 5, 4, 1, 1, 1.0, 0.0, False, torch.relu),
            (32, 100, 512, 32, 32, 10.0, 0.5, False, torch.relu),
            (1, 5, 4, 1, 1, 1.0, 0.0, False, torch.sigmoid),
            (32, 100, 512, 8, 8, 10.0, 0.0, False, torch.sigmoid),
        ],
    )
    def test_syntactic_distance_gate_layer(
        self,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_lookback_range: int,
        num_gate_heads: int,
        tau: float,
        dropout_p: float,
        batch_first: bool,
        distance_activation_fn: Optional[Callable],
    ):
        x = torch.randn((seq_len, batch_size, embed_dim))
        if batch_first:
            x = x.permute(1, 0, 2)

        gate = gtcnn.SyntacticDistanceGate(
            in_embed_dim=embed_dim,
            num_lookback_range=num_lookback_range,
            num_gate_heads=num_gate_heads,
            tau=tau,
            dropout_p=dropout_p,
            batch_first=batch_first,
            distance_activation_fn=distance_activation_fn,
        )
        gate_output, distance = gate(x)
        assert tuple(gate_output.size()) == (
            batch_size * num_gate_heads,
            seq_len,
            seq_len,
        ), f"{gate_output.size()=}, expected {(batch_size, seq_len, seq_len)}"
        assert tuple(distance.size()) == (
            batch_size,
            seq_len,
            1,
        ), f"{distance.size()=} expected {(batch_size, seq_len, 1)}"

    @pytest.mark.parametrize(
        "batch_size, seq_len, embed_dim, num_heads, batch_first, attn_gate",
        [
            (1, 5, 4, 1, False, None),
            (1, 5, 4, 2, True, None),
            (1, 5, 4, 4, False, None),
            (2, 5, 4, 1, True, None),
            (2, 5, 4, 2, False, None),
            (2, 5, 4, 4, True, None),
            (32, 100, 512, 32, True, None),
            (1, 5, 4, 1, True, torch.randn(1, 5, 5)),
            (1, 5, 4, 2, True, torch.randn(1, 5, 5)),
            (1, 5, 4, 4, False, torch.randn(1, 5, 5)),
            (2, 5, 4, 1, True, torch.randn(2, 5, 5)),
            (2, 5, 4, 2, False, torch.randn(2, 5, 5)),
            (2, 5, 4, 4, False, torch.randn(2, 5, 5)),
            (32, 100, 512, 32, False, torch.randn(32, 100, 100)),
        ],
    )
    def test_multihead_attention_with_gate(
        self,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        batch_first: bool,
        attn_gate: Optional[torch.Tensor],
    ):
        x = torch.randn((seq_len, batch_size, embed_dim))
        if batch_first:
            x = x.permute(1, 0, 2)
        q = k = v = x
        attn = gtcnn.MultiheadAttentionWithGate(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=batch_first,
        )
        attn_output, attn_weights = attn(q, k, v)
        if batch_first:
            assert tuple(attn_output.size()) == (
                batch_size,
                seq_len,
                embed_dim,
            ), f"{attn_output.size()=}, expected {(batch_size, seq_len, embed_dim)}"
            assert tuple(attn_weights.size()) == (
                batch_size,
                seq_len,
                seq_len,
            ), f"{attn_weights.size()=}, expected {(batch_size, seq_len, seq_len)}"
        else:
            assert tuple(attn_output.size()) == (
                seq_len,
                batch_size,
                embed_dim,
            ), f"{attn_output.size()=}, expected {(seq_len, batch_size, embed_dim)}"
            assert tuple(attn_weights.size()) == (
                batch_size,
                seq_len,
                seq_len,
            ), f"{attn_weights.size()=}, expected {(batch_size, seq_len, seq_len)}"
