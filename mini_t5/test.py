import torch

from modules.feed_forward import FeedForward



x = torch.randn(
    2,
    10,
    64
)



ff = FeedForward(
    embedding_dim=64,
    hidden_dim=256
)



out = ff(x)


print(out.shape)