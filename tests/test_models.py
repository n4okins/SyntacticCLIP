import torch
import pytest

import gated_tree_clip.nn as gtnn


class TestModels:
    @pytest.mark.parametrize("model", [gtnn.CLIP, gtnn.StructFormer])
    def test_model(self, model):
        model = model(10, 5)
        x = torch.randn(10)
        out = model(x)
        assert out.shape == (5,)
