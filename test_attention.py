import torch

from common.nn.layers.multi_head_attention import MultiHeadAttention


model = MultiHeadAttention(
    embedding_dim=128,
    num_heads=4,
)


x = torch.randn(
    32,
    128,
    128
)


y = model(x)

print(y.shape)