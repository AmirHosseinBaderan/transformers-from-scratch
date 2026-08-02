from __future__ import annotations

import torch
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
            use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing

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

    def _attention_forward(self, x):
        """Forward pass for attention block (used for gradient checkpointing)."""
        return x + self.attention(self.norm1(x))

    def _feed_forward_forward(self, x):
        """Forward pass for feed forward block (used for gradient checkpointing)."""
        return x + self.feed_forward(self.norm2(x))

    def forward(self, x):
        # Attention + Residual
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._attention_forward,
                x,
                use_reentrant=False,
            )
        else:
            x = x + self.attention(self.norm1(x))

        # Feed Forward + Residual
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._feed_forward_forward,
                x,
                use_reentrant=False,
            )
        else:
            x = x + self.feed_forward(self.norm2(x))

        return x
