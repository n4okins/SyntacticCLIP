# %%
import gated_tree_clip.nn as gtc_nn
import torch
import torch.nn as nn
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)

x = torch.randn((2, 5, 4))
in_layer = gtc_nn.MultiheadAttention(
    embed_dim=4,
    num_heads=2,
)
for key, param in in_layer.state_dict().items():
    print(key, param.shape)

layers = nn.ModuleList(
    [
        gtc_nn.MultiheadAttention(
            embed_dim=4,
            num_heads=2,
        ) for _ in range(16)
    ]
)
attn_out, attn_weight = in_layer(x, x, x)
for layer in layers:
    attn_out, attn_weight = layer(attn_out, attn_out, attn_out)

print(f"自作: {attn_out=}")
logger.info(attn_out.shape)
logger.info(attn_weight.shape)

in_layer = gtc_nn.MultiheadAttentionWithGate(
    embed_dim=4,
    num_heads=2,
)
for key, param in in_layer.state_dict().items():
    print(key, param.shape)

layers = nn.ModuleList(
    [
        gtc_nn.MultiheadAttentionWithGate(
            embed_dim=4,
            num_heads=2,
        ) for _ in range(16)
    ]
)
attn_out, attn_weight = in_layer(x, x, x)
for layer in layers:
    attn_out, attn_weight = layer(attn_out, attn_out, attn_out)

print(f"自作 with Gate: {attn_out=}")
logger.info(attn_out.shape)
logger.info(attn_weight.shape)



in_layer = nn.MultiheadAttention(
    embed_dim=4,
    num_heads=2,
)
for key, param in in_layer.state_dict().items():
    print(key, param.shape)

attn_out, attn_weight = in_layer(x, x, x)
layers = nn.ModuleList(
    [
        nn.MultiheadAttention(
            embed_dim=4,
            num_heads=2,
        ) for _ in range(16)
    ]
)
logger.info(attn_out.shape)
logger.info(attn_weight.shape)
print(f"Torch: {attn_out=}")
# %%
