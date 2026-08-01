import torch

from common.nn.layers.input_embedding import InputEmbedding


model = InputEmbedding(
    vocab_size=500,
    block_size=128,
    embedding_dim=64,
)


x = torch.randint(
    0,
    500,
    (32,128)
)


out = model(x)

print(out.shape)