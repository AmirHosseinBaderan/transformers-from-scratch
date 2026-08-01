import torch

from common.nn.blocks.decoder_block import DecoderBlock


block = DecoderBlock(
    embedding_dim=128,
    num_heads=4,
)


x = torch.randn(
    32,
    128,
    128
)


y = block(x)


print(y.shape)