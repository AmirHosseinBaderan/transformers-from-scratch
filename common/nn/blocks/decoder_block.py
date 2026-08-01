from __future__ import annotations

import torch.nn as nn

from common.nn.layers.multi_head_attention import (
    MultiHeadAttention,
)

from common.nn.layers.feed_forward import (
    FeedForward,
)

from common.nn.layers.layer_norm import (
    LayerNorm,
)



class DecoderBlock(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            dropout: float = 0.1,
    ):
        super().__init__()


        self.norm1 = LayerNorm(
            embedding_dim
        )


        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout,
        )


        self.norm2 = LayerNorm(
            embedding_dim
        )


        self.feed_forward = FeedForward(
            embedding_dim,
            dropout,
        )


    def forward(self, x):

        # Attention + Residual

        x = x + self.attention(
            self.norm1(x)
        )


        # Feed Forward + Residual

        x = x + self.feed_forward(
            self.norm2(x)
        )


        return x