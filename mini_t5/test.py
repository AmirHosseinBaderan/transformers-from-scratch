import torch

from model import MiniT5



model = MiniT5(
    vocab_size=500,
    embedding_dim=128,
    num_layers=2,
    num_heads=8,
    ff_hidden_dim=512,
    max_length=64
)



encoder_input = torch.randint(
    0,
    500,
    (4,20)
)


decoder_input = torch.randint(
    0,
    500,
    (4,15)
)



output = model(
    encoder_input,
    decoder_input
)


print(output.shape)