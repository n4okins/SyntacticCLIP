import gated_tree_clip.nn as gtnn
import pytest
import torch


class TestModels:
    @pytest.mark.parametrize("model", [gtnn.CLIPBase])
    def test_model(self, model): ...
