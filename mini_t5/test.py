import torch

from mini_t5.modules.multi_head_attention import MultiHeadAttention



x = torch.randn(
    2,
    10,
    64
)


attention = MultiHeadAttention(
    embedding_dim=64,
    num_heads=8
)


out = attention(
    x,
    x,
    x
)


print(out.shape)