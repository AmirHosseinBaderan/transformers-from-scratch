import torch

from mini_t5.modules.embedding import TokenEmbedding
from mini_t5.modules.positional_encoding import PositionalEncoding


batch = torch.tensor(
    [
        [1,2,3,4],
        [4,3,2,1]
    ]
)


embedding = TokenEmbedding(
    vocab_size=100,
    embedding_dim=32
)


position = PositionalEncoding(
    embedding_dim=32
)


x = embedding(batch)

print(x.shape)


x = position(x)

print(x.shape)