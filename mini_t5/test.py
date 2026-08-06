import torch

from decoder import T5Decoder



decoder = T5Decoder(
    vocab_size=1000,
    embedding_dim=128,
    num_layers=3,
    num_heads=8,
    ff_hidden_dim=512,
    max_length=64
)



decoder_input = torch.randint(
    0,
    1000,
    (2,15)
)



encoder_output = torch.randn(
    2,
    20,
    128
)



out = decoder(
    decoder_input,
    encoder_output
)



print(out.shape)