from __future__ import annotations

import torch
import torch.nn as nn

from common.nn.layers.attention import SelfAttention
from common.nn.layers.feed_forward import FeedForward
from common.nn.layers.layer_norm import LayerNorm


class EncoderBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = SelfAttention(
            head_dim=embedding_dim,
            dropout=dropout,
            causal=False,
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.norm1 = LayerNorm(
            embedding_dim,
        )

        self.norm2 = LayerNorm(
            embedding_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:


        attention_output = self.attention(
            x,
            causal=False,
        )

        x = self.norm1(
            x + self.dropout(attention_output)
        )


        feed_forward_output = self.feed_forward(
            x
        )


        x = self.norm2(
            x + self.dropout(feed_forward_output)
        )


        return x