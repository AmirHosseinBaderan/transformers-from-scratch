from __future__ import annotations

import torch
import torch.nn as nn


class MaskedLMHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.activation = nn.GELU()

        self.norm = nn.LayerNorm(
            embedding_dim,
        )

        self.decoder = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        x = self.linear(
            hidden_states
        )

        x = self.activation(
            x
        )

        x = self.norm(
            x
        )

        logits = self.decoder(
            x
        )

        return logits