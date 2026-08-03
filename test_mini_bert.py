import torch

from mini_bert.model import MiniBERT


model = MiniBERT(
    vocab_size=1000,
    block_size=64,
    embedding_dim=128,
    num_layers=2,
)


x = torch.randint(
    0,
    1000,
    (4,64)
)


output = model(x)


print(output.shape)