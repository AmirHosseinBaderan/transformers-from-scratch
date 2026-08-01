from __future__ import annotations

import torch.nn as nn

from common.nn.layers.embedding import TokenEmbedding
from common.nn.layers.positional_embedding import PositionalEmbedding


class InputEmbedding(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            block_size: int,
            embedding_dim: int,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            block_size,
            embedding_dim,
        )


    def forward(
            self,
            x,
    ):

        token_vectors = self.token_embedding(x)

        position_vectors = self.position_embedding(x)


        return token_vectors + position_vectors