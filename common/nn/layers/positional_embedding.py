from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Adds information about token positions
    inside a sequence.
    """

    def __init__(
            self,
            block_size: int,
            embedding_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim,
        )


    def forward(
            self,
            x,
    ):
        """
        x shape:

        (batch, sequence_length)

        """

        batch_size, seq_length = x.shape


        positions = torch.arange(
            seq_length,
            device=x.device,
        )


        return self.embedding(
            positions
        )