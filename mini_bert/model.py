from __future__ import annotations

import torch
import torch.nn as nn

from common.nn.blocks.encoder_block import EncoderBlock
from common.nn.layers.input_embedding import InputEmbedding
from common.nn.layers.layer_norm import LayerNorm


class MiniBERT(nn.Module):

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
            vocab_size=vocab_size,
            block_size=block_size,
            embedding_dim=embedding_dim,
        )

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(
            embedding_dim
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module,
    ):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):

        x = self.embedding(
            input_ids
        )

        for block in self.blocks:

            x = block(
                x,
                attention_mask=attention_mask,
            )

        x = self.norm(
            x
        )

        return x