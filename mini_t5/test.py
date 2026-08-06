import torch

from modules.encoder_block import EncoderBlock


x = torch.randn(
    2,
    20,
    128
)


block = EncoderBlock(
    embedding_dim=128,
    num_heads=8,
    ff_hidden_dim=512
)


out = block(x)


print(out.shape)