import torch

from encoder import T5Encoder



model = T5Encoder(
    vocab_size=1000,
    embedding_dim=128,
    num_layers=3,
    num_heads=8,
    ff_hidden_dim=512,
    max_length=64
)



x = torch.randint(
    0,
    1000,
    (2,20)
)


out = model(x)


print(out.shape)