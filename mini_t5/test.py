import torch

from modules.layer_norm import LayerNorm


x = torch.randn(
    2,
    10,
    64
)


norm = LayerNorm(64)


out = norm(x)


print(out.shape)