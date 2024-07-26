import torch
import pytest

from gated_tree_clip.models import CLIP, StructFormer

class TestModels:
    def test_clip_model(self):
        model = CLIP(512, 256)
        x = torch.randn(1, 512)
        assert model(x).shape == (1, 256)
    
    def test_structformer_model(self):
        model = StructFormer(512, 256)
        x = torch.randn(1, 512)
        assert model(x).shape == (1, 256)