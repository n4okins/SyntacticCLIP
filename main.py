# %%
import gated_tree_clip.nn as gtc_nn
import torch
from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)

x = torch.randn((2, 5, 4))
q = k = v = x
layer = gtc_nn.MultiheadAttention(
    embed_dim=4,
    num_heads=2,
)
attn_out, attn_weight = layer(q, k, v)
logger.info(attn_out.shape)
logger.info(attn_weight.shape)


x = torch.randn((2, 5, 4))
q = k = v = x
gate = gtc_nn.SyntacticDistanceGate(
    in_embed_dim=4,
    num_gate_heads=1,
    num_lookback_range=3,
)
layer = gtc_nn.MultiheadAttentionWithGate(embed_dim=4, num_heads=2, batch_first=True)

attn_gate, distance = gate(k)
print(f"{attn_gate.size()=}, {distance.size()=}")
attn_out, attn_weight = layer(q, k, v, attn_gate=attn_gate)
logger.info(attn_out.shape)
logger.info(attn_weight.shape)

print(f"{attn_out.size()=}, {attn_weight.size()=}")

# %%
