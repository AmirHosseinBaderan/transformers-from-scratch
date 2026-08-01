from __future__ import annotations

import torch.nn as nn

from common.nn.layers.input_embedding import (
    InputEmbedding,
)

from common.nn.layers.layer_norm import (
    LayerNorm,
)

from common.nn.blocks.decoder_block import (
    DecoderBlock,
)


class MiniGPT(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            block_size: int,
            embedding_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.1,
    ):
        super().__init__()


        self.embedding = InputEmbedding(
            vocab_size,
            block_size,
            embedding_dim,
        )


        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )


        self.norm = LayerNorm(
            embedding_dim
        )


        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
        )


    def forward(
            self,
            x,
    ):

        x = self.embedding(x)


        for block in self.blocks:
            x = block(x)


        x = self.norm(x)


        logits = self.lm_head(x)


        return logits