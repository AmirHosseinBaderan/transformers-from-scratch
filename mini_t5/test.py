import torch

from modules.decoder_block import DecoderBlock



decoder_input = torch.randn(
    2,
    10,
    128
)


encoder_output = torch.randn(
    2,
    15,
    128
)


block = DecoderBlock(
    embedding_dim=128,
    num_heads=8,
    ff_hidden_dim=512
)


out = block(
    decoder_input,
    encoder_output
)


print(out.shape)